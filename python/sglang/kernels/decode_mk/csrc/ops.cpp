// Copyright 2026 Fractalyze Inc. All rights reserved.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "barrier.h"
#include "decode.h"
#include "gdn.h"
#include "gemv.h"
#include "int4_gemv.h"
#include "qwen38_decode.h"
#include "qwen38_layer.h"
#include "qwen38_mtp.h"
#include "qwen38_prefill.h"
#include "qwen3omni_cp.h"
#include "sampler.h"
#include "slow_ar.h"
#include "talker_decode.h"
#include "thinker_attention.h"
#include "thinker_decode.h"
#include "thinker_moe.h"
#include "thinker_prefill.h"

namespace s2mk {
namespace {

void CheckOk(cudaError_t err, const char* what) {
  TORCH_CHECK(err == cudaSuccess, what, " failed: ", cudaGetErrorString(err));
}

// The data pointer of a contiguous CUDA tensor of `dtype`.
template <typename T>
T* Ptr(const torch::Tensor& t, torch::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must live on the GPU");
  TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype);
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  return reinterpret_cast<T*>(t.data_ptr());
}

// The device view of an ErrorRecord held in a pinned int32 CPU tensor.
ErrorRecord* MappedRecord(const torch::Tensor& error) {
  TORCH_CHECK(error.is_pinned() && error.scalar_type() == torch::kInt32 &&
                  error.numel() * sizeof(int32_t) == sizeof(ErrorRecord),
              "error must be a pinned int32 tensor of ",
              sizeof(ErrorRecord) / sizeof(int32_t), " elements");
  void* device = nullptr;
  CheckOk(MappedDevicePointer(error.data_ptr(), &device),
          "mapping the error record");
  return static_cast<ErrorRecord*>(device);
}

// Checks a [num_layers, 8] table of LayerWeights pointers.
const LayerWeights* LayerTable(const torch::Tensor& layers, const char* name) {
  static_assert(sizeof(LayerWeights) == 8 * sizeof(int64_t));
  TORCH_CHECK(layers.dim() == 2 && layers.size(1) == 8 && layers.size(0) > 0,
              name, " must be [num_layers, 8]");
  return Ptr<const LayerWeights>(layers, torch::kInt64, name);
}

void CheckNumel(const torch::Tensor& t, int64_t numel, const char* name) {
  TORCH_CHECK(t.numel() == numel, name, " must hold ", numel,
              " elements, got ", t.numel());
}

// A kInt4Zp Weight of `rows` rows of k values from (packed, scales, zeros):
// packed int32 [rows, k / 8], bf16 scales [rows, k / 32] and zero points
// int32 [rows, k / 256].
Weight Int4ZpWeightOf(const std::vector<torch::Tensor>& parts, int64_t rows,
                      int64_t k, const std::string& name) {
  TORCH_CHECK(parts.size() == 3, name, " must be (packed, scales, zeros)");
  const std::string packed_name = name + "_packed";
  const std::string scales_name = name + "_scales";
  const std::string zeros_name = name + "_zeros";
  CheckNumel(parts[0], rows * k / 8, packed_name.c_str());
  CheckNumel(parts[1], rows * k / 32, scales_name.c_str());
  CheckNumel(parts[2], rows * k / 256, zeros_name.c_str());
  Weight w{};
  w.format = WeightFormat::kInt4Zp;
  w.data = Ptr<const int32_t>(parts[0], torch::kInt32, packed_name.c_str());
  w.scales = Ptr<const __nv_bfloat16>(parts[1], torch::kBFloat16,
                                      scales_name.c_str());
  w.zeros = Ptr<const uint32_t>(parts[2], torch::kInt32, zeros_name.c_str());
  return w;
}

// A kBf16 Weight, [rows, k].
Weight Bf16WeightOf(const torch::Tensor& weight, int64_t k, const char* name) {
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == k, name, " must be [rows, ",
              k, "]");
  Weight w{};
  w.format = WeightFormat::kBf16;
  w.data = Ptr<const __nv_bfloat16>(weight, torch::kBFloat16, name);
  return w;
}

// A [rows, k] Weight in either format: a kInt4Zp (packed, scales, zeros), or
// a kBf16 weight alone.
Weight WeightOf(const std::vector<torch::Tensor>& parts, int64_t rows,
                int64_t k, const std::string& name) {
  if (parts.size() != 1) return Int4ZpWeightOf(parts, rows, k, name);
  const Weight w = Bf16WeightOf(parts[0], k, name.c_str());
  TORCH_CHECK(parts[0].size(0) == rows, name, " must have ", rows, " rows");
  return w;
}

// Qwen3.8's LM head, read by its tag: a kBf16 [vocab, kQwen38Dim] weight
// alone, or a kInt4Zp (packed, scales, zeros). Sets `vocab` to its rows.
Weight LmHeadOf(const std::vector<torch::Tensor>& parts, int* vocab) {
  TORCH_CHECK(!parts.empty() && parts[0].dim() == 2,
              "lm_head must be a [vocab, ", kQwen38Dim,
              "] bf16 weight or (packed, scales, zeros)");
  *vocab = static_cast<int>(parts[0].size(0));
  return WeightOf(parts, *vocab, kQwen38Dim, "lm_head");
}

// The SM count of `device`, the most CTAs a persistent launch may use.
int SmCount(int device) {
  int sms = 0;
  CheckOk(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
          "reading the SM count");
  return sms;
}

void CheckGrid(int64_t num_ctas) {
  TORCH_CHECK(kQHeads <= num_ctas && num_ctas <= kMaxCtas,
              "the attention split needs ", kQHeads, " to ", kMaxCtas,
              " CTAs, got ", num_ctas);
}

void CheckPrefetch(int64_t prefetch_bytes) {
  TORCH_CHECK(0 <= prefetch_bytes && prefetch_bytes <= (1 << 30) &&
                  prefetch_bytes % 16 == 0,
              "prefetch_bytes must be a multiple of 16 in [0, 2^30], got ",
              prefetch_bytes);
}

void RunGemv(const torch::Tensor& descs, int64_t num_ctas, int64_t smem_bytes) {
  TORCH_CHECK(descs.is_cuda(), "descs must live on the GPU");
  TORCH_CHECK(descs.scalar_type() == torch::kInt64, "descs must be int64");
  TORCH_CHECK(descs.dim() == 2 && descs.size(1) == 5 && descs.is_contiguous(),
              "descs must be a contiguous [count, 5] tensor");
  static_assert(sizeof(GemvDesc) == 5 * sizeof(int64_t));
  const c10::cuda::CUDAGuard guard(descs.device());
  CheckOk(LaunchGemv(
              reinterpret_cast<const GemvDesc*>(descs.data_ptr<int64_t>()),
              static_cast<int>(descs.size(0)), static_cast<int>(num_ctas),
              static_cast<int>(smem_bytes), at::cuda::getCurrentCUDAStream()),
          "gemv launch");
}

torch::Tensor RunInt4Gemv(const torch::Tensor& packed,
                          const torch::Tensor& scales,
                          const std::optional<torch::Tensor>& zeros,
                          const torch::Tensor& x, int64_t num_ctas) {
  constexpr int64_t kK = kInt4GemvK;
  TORCH_CHECK(packed.dim() == 2 && packed.size(1) == kK / 8,
              "packed must be [n, ", kK / 8, "]");
  const int64_t n = packed.size(0);
  TORCH_CHECK(num_ctas > 0 && (n + num_ctas - 1) / num_ctas <=
                                  kInt4GemvMaxRowsPerCta,
              "num_ctas ", num_ctas, " gives a CTA more than ",
              kInt4GemvMaxRowsPerCta, " rows");
  CheckNumel(scales, n * kK / 32, "scales");
  CheckNumel(x, kK, "x");
  const uint32_t* zeros_ptr = nullptr;
  if (zeros.has_value()) {
    CheckNumel(*zeros, n * kK / 256, "zeros");
    zeros_ptr = Ptr<const uint32_t>(*zeros, torch::kInt32, "zeros");
  }
  torch::Tensor y = torch::empty({n}, x.options().dtype(torch::kFloat32));
  const c10::cuda::CUDAGuard guard(x.device());
  CheckOk(LaunchInt4Gemv(Ptr<const int32_t>(packed, torch::kInt32, "packed"),
                         Ptr<const __nv_bfloat16>(scales, torch::kBFloat16,
                                                  "scales"),
                         zeros_ptr,
                         Ptr<const __nv_bfloat16>(x, torch::kBFloat16, "x"),
                         y.data_ptr<float>(), static_cast<int>(n),
                         static_cast<int>(num_ctas),
                         at::cuda::getCurrentCUDAStream()),
          "int4 gemv launch");
  return y;
}

torch::Tensor RunInt4Mma(const std::optional<torch::Tensor>& packed,
                         const std::optional<torch::Tensor>& scales,
                         const std::optional<torch::Tensor>& zeros,
                         const std::optional<torch::Tensor>& bf16,
                         const torch::Tensor& x) {
  constexpr int64_t kK = kInt4GemvK;
  TORCH_CHECK(packed.has_value() != bf16.has_value(),
              "pass an int4 W (packed, scales) or a bf16 W, not both");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kK && 0 < x.size(0) &&
                  x.size(0) <= kInt4MmaMaxTokens,
              "x must be [tokens ≤ ", kInt4MmaMaxTokens, ", ", kK, "]");
  const int64_t tokens = x.size(0);
  const int32_t* packed_ptr = nullptr;
  const __nv_bfloat16* scales_ptr = nullptr;
  const uint32_t* zeros_ptr = nullptr;
  const __nv_bfloat16* bf16_ptr = nullptr;
  int64_t n = 0;
  if (packed.has_value()) {
    TORCH_CHECK(scales.has_value(), "an int4 W needs its scales");
    TORCH_CHECK(packed->dim() == 2 && packed->size(1) == kK / 8,
                "packed must be [n, ", kK / 8, "]");
    n = packed->size(0);
    CheckNumel(*scales, n * kK / 32, "scales");
    packed_ptr = Ptr<const int32_t>(*packed, torch::kInt32, "packed");
    scales_ptr = Ptr<const __nv_bfloat16>(*scales, torch::kBFloat16, "scales");
    if (zeros.has_value()) {
      CheckNumel(*zeros, n * kK / 256, "zeros");
      zeros_ptr = Ptr<const uint32_t>(*zeros, torch::kInt32, "zeros");
    }
  } else {
    TORCH_CHECK(bf16->dim() == 2 && bf16->size(1) == kK, "bf16 must be [n, ",
                kK, "]");
    n = bf16->size(0);
    bf16_ptr = Ptr<const __nv_bfloat16>(*bf16, torch::kBFloat16, "bf16");
  }
  TORCH_CHECK(n > 0 && n % 16 == 0, "W's rows must be a multiple of 16");
  torch::Tensor y =
      torch::empty({tokens, n}, x.options().dtype(torch::kFloat32));
  const c10::cuda::CUDAGuard guard(x.device());
  CheckOk(LaunchInt4Mma(packed_ptr, scales_ptr, zeros_ptr, bf16_ptr,
                        Ptr<const __nv_bfloat16>(x, torch::kBFloat16, "x"),
                        y.data_ptr<float>(), static_cast<int>(n),
                        static_cast<int>(tokens),
                        at::cuda::getCurrentCUDAStream()),
          "int4 mma launch");
  return y;
}

// A params struct as a CPU uint8 tensor, and back: the thinker's blocks are
// built once per layer and launched, alone or stacked into a decode step,
// from these bytes.
template <typename T>
torch::Tensor StructBytes(const T& value) {
  torch::Tensor bytes = torch::empty({static_cast<int64_t>(sizeof(T))},
                                     torch::dtype(torch::kUInt8));
  std::memcpy(bytes.data_ptr(), &value, sizeof(T));
  return bytes;
}

template <typename T>
T StructFrom(const torch::Tensor& bytes, const char* name) {
  TORCH_CHECK(bytes.device().is_cpu() && bytes.scalar_type() == torch::kUInt8 &&
                  bytes.is_contiguous() && bytes.numel() == sizeof(T),
              name, " must be ", sizeof(T), " contiguous CPU bytes");
  T value;
  std::memcpy(&value, bytes.data_ptr(), sizeof(T));
  return value;
}

// A layer's paged cache from vLLM's key and value views, [blocks,
// block_size, kv_heads, head_dim] each, in any strides that keep a head's
// values contiguous.
PagedKv PagedKvOf(const torch::Tensor& key_cache,
                  const torch::Tensor& value_cache,
                  const torch::Tensor& block_table,
                  int64_t kv_heads = kThinkerKvHeads,
                  int64_t head_dim = kThinkerHeadDim) {
  for (const torch::Tensor* t : {&key_cache, &value_cache}) {
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 &&
                    t->dim() == 4,
                "the KV cache must be 4-D bf16 on the GPU");
    TORCH_CHECK(t->size(2) == kv_heads && t->size(3) == head_dim &&
                    t->stride(3) == 1,
                "the KV cache must be [blocks, block_size, ", kv_heads,
                ", ", head_dim, "] with each head contiguous");
  }
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes() &&
                  key_cache.strides() == value_cache.strides(),
              "key and value caches must share a layout");
  PagedKv kv{};
  kv.key = reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr());
  kv.value = reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr());
  kv.block_table = Ptr<const int32_t>(block_table, torch::kInt32, "block_table");
  kv.block_stride = key_cache.stride(0);
  kv.slot_stride = key_cache.stride(1);
  kv.head_stride = key_cache.stride(2);
  kv.block_size = static_cast<int>(key_cache.size(1));
  return kv;
}

torch::Tensor ThinkerAttentionParamsOf(
    const torch::Tensor& residual_in, const torch::Tensor& norm,
    const torch::Tensor& wqkv_packed, const torch::Tensor& wqkv_scales,
    const torch::Tensor& q_norm, const torch::Tensor& k_norm,
    const torch::Tensor& wo_packed, const torch::Tensor& wo_scales,
    const torch::Tensor& cos_sin, const torch::Tensor& positions,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& block_table, int64_t pos, int64_t splits, double eps,
    int64_t timeout_ns, const torch::Tensor& qkv,
    const torch::Tensor& partial_ml, const torch::Tensor& partial_o,
    const torch::Tensor& residual, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kThinkerDim, kQkv = kThinkerQkvRows,
                    kQ = kThinkerQDim, kHd = kThinkerHeadDim;
  const int64_t items = kThinkerQHeads * splits;
  TORCH_CHECK(0 < num_ctas && num_ctas <= kMaxCtas &&
                  (kQkv + num_ctas - 1) / num_ctas <= kMaxRowsPerCta,
              "num_ctas ", num_ctas, " gives a CTA too many rows");
  TORCH_CHECK(splits > 0 && items <= num_ctas, "splits ", splits, " needs ",
              items, " CTAs, have ", num_ctas);
  CheckNumel(residual_in, kD, "residual_in");
  CheckNumel(norm, kD, "norm");
  CheckNumel(wqkv_packed, kQkv * kD / 8, "wqkv_packed");
  CheckNumel(wqkv_scales, kQkv * kD / 32, "wqkv_scales");
  CheckNumel(q_norm, kHd, "q_norm");
  CheckNumel(k_norm, kHd, "k_norm");
  CheckNumel(wo_packed, kD * kQ / 8, "wo_packed");
  CheckNumel(wo_scales, kD * kQ / 32, "wo_scales");
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == kHd,
              "cos_sin must be [positions, ", kHd, "]");
  CheckNumel(positions, 3, "positions");
  CheckNumel(qkv, kQkv, "qkv");
  CheckNumel(partial_ml, items * 2, "partial_ml");
  CheckNumel(partial_o, items * kHd, "partial_o");
  CheckNumel(residual, kD, "residual");
  const PagedKv kv = PagedKvOf(key_cache, value_cache, block_table);
  TORCH_CHECK(0 <= pos && pos < block_table.numel() * kv.block_size,
              "pos ", pos, " lies past the block table");
  const auto kBf16 = torch::kBFloat16;
  ThinkerAttentionParams p{};
  p.residual_in = Ptr<const float>(residual_in, torch::kFloat32, "residual_in");
  p.norm = Ptr<const __nv_bfloat16>(norm, kBf16, "norm");
  p.wqkv_packed = Ptr<const int32_t>(wqkv_packed, torch::kInt32, "wqkv_packed");
  p.wqkv_scales = Ptr<const __nv_bfloat16>(wqkv_scales, kBf16, "wqkv_scales");
  p.q_norm = Ptr<const __nv_bfloat16>(q_norm, kBf16, "q_norm");
  p.k_norm = Ptr<const __nv_bfloat16>(k_norm, kBf16, "k_norm");
  p.wo_packed = Ptr<const int32_t>(wo_packed, torch::kInt32, "wo_packed");
  p.wo_scales = Ptr<const __nv_bfloat16>(wo_scales, kBf16, "wo_scales");
  p.cos_sin = Ptr<const __nv_bfloat16>(cos_sin, kBf16, "cos_sin");
  p.positions = Ptr<const int32_t>(positions, torch::kInt32, "positions");
  p.kv = kv;
  p.pos = static_cast<int>(pos);
  p.splits = static_cast<int>(splits);
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.qkv = Ptr<float>(qkv, torch::kFloat32, "qkv");
  p.partial_ml = Ptr<float>(partial_ml, torch::kFloat32, "partial_ml");
  p.partial_o = Ptr<float>(partial_o, torch::kFloat32, "partial_o");
  p.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  return StructBytes(p);
}

void RunThinkerAttention(const torch::Tensor& params, int64_t num_ctas) {
  CheckOk(LaunchThinkerAttention(
              StructFrom<ThinkerAttentionParams>(params, "params"),
              static_cast<int>(num_ctas), at::cuda::getCurrentCUDAStream()),
          "thinker attention launch");
}

// The widest phase of a Qwen3.8 layer gives a CTA 2 × (kQwen38Ffn /
// num_ctas) rows of the gate-up, which MlpShared::ys holds for up to
// 2 × kMaxRowsPerCta.
void CheckQwen38Grid(int64_t num_ctas) {
  constexpr int64_t kRows[] = {kQwen38QkvRows, kQwen38Dim, kQwen38Ffn};
  TORCH_CHECK(0 < num_ctas && num_ctas <= kMaxCtas, "num_ctas ", num_ctas,
              " out of range");
  for (int64_t rows : kRows) {
    TORCH_CHECK((rows + num_ctas - 1) / num_ctas <= kMaxRowsPerCta,
                "num_ctas ", num_ctas, " gives a CTA too many rows");
  }
}

Qwen38MlpParams MlpParams(const torch::Tensor& post_norm,
                          const std::vector<torch::Tensor>& w13,
                          const std::vector<torch::Tensor>& w2, double eps,
                          const torch::Tensor& act) {
  constexpr int64_t kD = kQwen38Dim, kF = kQwen38Ffn;
  CheckNumel(post_norm, kD, "post_norm");
  CheckNumel(act, kF, "act");
  Qwen38MlpParams p{};
  p.post_norm =
      Ptr<const __nv_bfloat16>(post_norm, torch::kBFloat16, "post_norm");
  p.w13 = WeightOf(w13, 2 * kF, kD, "w13");
  p.w2 = WeightOf(w2, kD, kF, "w2");
  p.eps = static_cast<float>(eps);
  p.act = Ptr<__nv_bfloat16>(act, torch::kBFloat16, "act");
  return p;
}

torch::Tensor Qwen38MlpParamsOf(const torch::Tensor& post_norm,
                                const std::vector<torch::Tensor>& w13,
                                const std::vector<torch::Tensor>& w2,
                                double eps, const torch::Tensor& act) {
  return StructBytes(MlpParams(post_norm, w13, w2, eps, act));
}

torch::Tensor Qwen38LayerParamsOf(
    const torch::Tensor& residual_in, const torch::Tensor& input_norm,
    const std::vector<torch::Tensor>& wqkv, const torch::Tensor& q_norm,
    const torch::Tensor& k_norm, const std::vector<torch::Tensor>& wo,
    const torch::Tensor& cos_sin, const torch::Tensor& positions,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& block_table, int64_t pos, int64_t splits,
    const torch::Tensor& post_norm, const std::vector<torch::Tensor>& w13,
    const std::vector<torch::Tensor>& w2, double eps, int64_t timeout_ns,
    const torch::Tensor& qkv, const torch::Tensor& partial_ml,
    const torch::Tensor& partial_o, const torch::Tensor& act,
    const torch::Tensor& hidden, const torch::Tensor& residual,
    const torch::Tensor& sync, const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kQwen38Dim, kQkv = kQwen38QkvRows,
                    kQ = kQwen38QDim, kHd = kQwen38HeadDim;
  const int64_t items = kQwen38QHeads * splits;
  CheckQwen38Grid(num_ctas);
  TORCH_CHECK(splits > 0 && items <= num_ctas, "splits ", splits, " needs ",
              items, " CTAs, have ", num_ctas);
  CheckNumel(residual_in, kD, "residual_in");
  CheckNumel(input_norm, kD, "input_norm");
  CheckNumel(q_norm, kHd, "q_norm");
  CheckNumel(k_norm, kHd, "k_norm");
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == kQwen38Rotary,
              "cos_sin must be [positions, ", kQwen38Rotary, "]");
  CheckNumel(positions, 3, "positions");
  CheckNumel(qkv, kQkv, "qkv");
  CheckNumel(partial_ml, items * 2, "partial_ml");
  CheckNumel(partial_o, items * kHd, "partial_o");
  CheckNumel(hidden, kD, "hidden");
  CheckNumel(residual, kD, "residual");
  const PagedKv kv =
      PagedKvOf(key_cache, value_cache, block_table, kQwen38KvHeads, kHd);
  TORCH_CHECK(0 <= pos && pos < block_table.numel() * kv.block_size,
              "pos ", pos, " lies past the block table");
  const auto kBf16 = torch::kBFloat16;
  Qwen38LayerParams p{};
  p.residual_in = Ptr<const float>(residual_in, torch::kFloat32, "residual_in");
  p.input_norm = Ptr<const __nv_bfloat16>(input_norm, kBf16, "input_norm");
  p.wqkv = WeightOf(wqkv, kQkv, kD, "wqkv");
  p.q_norm = Ptr<const __nv_bfloat16>(q_norm, kBf16, "q_norm");
  p.k_norm = Ptr<const __nv_bfloat16>(k_norm, kBf16, "k_norm");
  p.wo = WeightOf(wo, kD, kQ, "wo");
  p.cos_sin = Ptr<const __nv_bfloat16>(cos_sin, kBf16, "cos_sin");
  p.positions = Ptr<const int32_t>(positions, torch::kInt32, "positions");
  p.kv = kv;
  p.pos = static_cast<int>(pos);
  p.splits = static_cast<int>(splits);
  p.mlp = MlpParams(post_norm, w13, w2, eps, act);
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.qkv = Ptr<float>(qkv, torch::kFloat32, "qkv");
  p.partial_ml = Ptr<float>(partial_ml, torch::kFloat32, "partial_ml");
  p.partial_o = Ptr<float>(partial_o, torch::kFloat32, "partial_o");
  p.hidden = Ptr<float>(hidden, torch::kFloat32, "hidden");
  p.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  return StructBytes(p);
}

void RunQwen38Layer(const torch::Tensor& params, int64_t num_ctas) {
  CheckOk(LaunchQwen38Layer(StructFrom<Qwen38LayerParams>(params, "params"),
                            static_cast<int>(num_ctas),
                            at::cuda::getCurrentCUDAStream()),
          "Qwen3.8 layer launch");
}

// Every layer's block params stacked on the device, [num_layers, bytes].
template <typename T>
const T* StackedParams(const torch::Tensor& stacked, const char* name) {
  TORCH_CHECK(stacked.is_cuda() && stacked.scalar_type() == torch::kUInt8 &&
                  stacked.is_contiguous() && stacked.dim() == 2 &&
                  stacked.size(1) == sizeof(T),
              name, " must be [layers, ", sizeof(T), "] uint8 on the GPU");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(stacked.data_ptr()) % alignof(T) == 0,
              name, " is misaligned");
  return reinterpret_cast<const T*>(stacked.data_ptr());
}

void RunBarrierProbe(int64_t rounds, int64_t skip_cta, int64_t skip_barrier,
                     int64_t step, int64_t timeout_ns, const torch::Tensor& sync,
                     const torch::Tensor& error, const torch::Tensor& slots,
                     const torch::Tensor& mismatches, int64_t num_ctas) {
  TORCH_CHECK(slots.numel() == 2 * num_ctas * kProbeWordsPerCta,
              "slots must hold 2 × num_ctas × PROBE_WORDS_PER_CTA");
  BarrierProbeArgs args{};
  args.rounds = static_cast<int>(rounds);
  args.skip_cta = static_cast<int>(skip_cta);
  args.skip_barrier = static_cast<int>(skip_barrier);
  args.step = static_cast<int>(step);
  args.timeout_ns = timeout_ns;
  args.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  args.error = MappedRecord(error);
  args.slots = Ptr<int>(slots, torch::kInt32, "slots");
  args.mismatches = Ptr<unsigned>(mismatches, torch::kInt32, "mismatches");
  const c10::cuda::CUDAGuard guard(sync.device());
  CheckOk(LaunchBarrierProbe(args, static_cast<int>(num_ctas),
                             at::cuda::getCurrentCUDAStream()),
          "barrier probe launch");
}

torch::Tensor GdnParamsOf(
    const torch::Tensor& residual_in, const torch::Tensor& norm,
    const std::vector<torch::Tensor>& in_proj, const torch::Tensor& gates,
    const torch::Tensor& conv, const torch::Tensor& a_log,
    const torch::Tensor& dt_bias, const torch::Tensor& out_norm,
    const std::vector<torch::Tensor>& out_proj, double eps, int64_t timeout_ns,
    const torch::Tensor& conv_state, const torch::Tensor& state,
    const torch::Tensor& mixed,
    const torch::Tensor& z, const torch::Tensor& beta,
    const torch::Tensor& decay, const torch::Tensor& core,
    const torch::Tensor& residual, const torch::Tensor& sync,
    const torch::Tensor& error) {
  constexpr int64_t kD = kGdnDim, kIn = kGdnInt4Rows, kV = kGdnValueDim,
                    kH = kGdnVHeads, kHd = kGdnHeadDim, kC = kGdnConvDim,
                    kW = kGdnConvWidth;
  CheckNumel(residual_in, kD, "residual_in");
  CheckNumel(norm, kD, "norm");
  CheckNumel(conv, kC * kW, "conv");
  CheckNumel(a_log, kH, "a_log");
  CheckNumel(dt_bias, kH, "dt_bias");
  CheckNumel(out_norm, kHd, "out_norm");
  CheckNumel(conv_state, kC * (kW - 1), "conv_state");
  CheckNumel(state, kH * kHd * kHd, "state");
  CheckNumel(mixed, kC, "mixed");
  CheckNumel(z, kV, "z");
  CheckNumel(beta, kH, "beta");
  CheckNumel(decay, kH, "decay");
  CheckNumel(core, kV, "core");
  CheckNumel(residual, kD, "residual");
  const auto kBf16 = torch::kBFloat16;
  const auto kF32 = torch::kFloat32;
  GdnParams p{};
  p.residual_in = Ptr<const float>(residual_in, kF32, "residual_in");
  p.norm = Ptr<const __nv_bfloat16>(norm, kBf16, "norm");
  p.in_proj = Int4ZpWeightOf(in_proj, kIn, kD, "in_proj");
  CheckNumel(gates, kGdnGateRows * kD, "gates");
  p.gates = Bf16WeightOf(gates, kD, "gates");
  p.conv = Ptr<const __nv_bfloat16>(conv, kBf16, "conv");
  p.a_log = Ptr<const float>(a_log, kF32, "a_log");
  p.dt_bias = Ptr<const float>(dt_bias, kF32, "dt_bias");
  p.out_norm = Ptr<const __nv_bfloat16>(out_norm, kBf16, "out_norm");
  p.out_proj = WeightOf(out_proj, kD, kV, "out_proj");
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.conv_state = Ptr<float>(conv_state, kF32, "conv_state");
  p.state = Ptr<float>(state, kF32, "state");
  p.mixed = Ptr<float>(mixed, kF32, "mixed");
  p.z = Ptr<float>(z, kF32, "z");
  p.beta = Ptr<float>(beta, kF32, "beta");
  p.decay = Ptr<float>(decay, kF32, "decay");
  p.core = Ptr<float>(core, kF32, "core");
  p.residual = Ptr<float>(residual, kF32, "residual");
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  return StructBytes(p);
}

void CheckGdnGrid(int64_t num_ctas) {
  const int sms = SmCount(at::cuda::current_device());
  TORCH_CHECK(kGdnMinCtas <= num_ctas && num_ctas <= sms, "num_ctas ",
              num_ctas, " must be in [", kGdnMinCtas, ", ", sms, "]");
}

void RunGdn(const torch::Tensor& params, int64_t num_ctas) {
  CheckGdnGrid(num_ctas);
  CheckOk(LaunchGdn(StructFrom<GdnParams>(params, "params"),
                    static_cast<int>(num_ctas),
                    at::cuda::getCurrentCUDAStream()),
          "gdn launch");
}


torch::Tensor RunQwen38Projection(const std::vector<torch::Tensor>& weight,
                                  const torch::Tensor& x, bool whole_k,
                                  int64_t num_ctas) {
  TORCH_CHECK(x.dim() == 2 && 0 < x.size(0) &&
                  x.size(0) <= kQwen38PrefillMaxTokens,
              "x must be [tokens ≤ ", kQwen38PrefillMaxTokens, ", k]");
  const int64_t k = x.size(1);
  TORCH_CHECK(k == kQwen38Dim || k == kQwen38QDim || k == kQwen38Ffn,
              "k must be one of the prefill's projection widths, got ", k);
  TORCH_CHECK(weight.size() == 3, "weight must be (packed, scales, zeros)");
  const int64_t n = weight[0].size(0);
  TORCH_CHECK(n > 0 && n % 16 == 0, "W's rows must be a multiple of 16");
  const Weight w = Int4ZpWeightOf(weight, n, k, "weight");
  const int sms = SmCount(at::cuda::current_device());
  TORCH_CHECK(0 < num_ctas && num_ctas <= sms, "num_ctas ", num_ctas,
              " must be in [1, ", sms, "]");
  torch::Tensor y =
      torch::empty({x.size(0), n}, x.options().dtype(torch::kFloat32));
  const c10::cuda::CUDAGuard guard(x.device());
  CheckOk(LaunchQwen38Projection(
              w, static_cast<int>(n), static_cast<int>(k),
              Ptr<const __nv_bfloat16>(x, torch::kBFloat16, "x"),
              static_cast<int>(x.size(0)), whole_k, y.data_ptr<float>(),
              static_cast<int>(num_ctas), at::cuda::getCurrentCUDAStream()),
          "Qwen3.8 projection launch");
  return y;
}

void RunQwen38Prefill(
    const torch::Tensor& linear, const torch::Tensor& linear_mlp,
    const torch::Tensor& full, int64_t tokens, const torch::Tensor& pos0,
    const torch::Tensor& positions, const torch::Tensor& final_norm,
    const std::vector<torch::Tensor>& lm_head, double eps, int64_t timeout_ns,
    const torch::Tensor& residual, const torch::Tensor& h,
    const torch::Tensor& proj, const torch::Tensor& beta,
    const torch::Tensor& gate, const torch::Tensor& query,
    const torch::Tensor& key, const torch::Tensor& value,
    const torch::Tensor& core, const torch::Tensor& attn,
    const torch::Tensor& act, const torch::Tensor& partial_ml,
    const torch::Tensor& partial_o, const std::optional<torch::Tensor>& logits,
    const std::optional<torch::Tensor>& hidden,
    const std::optional<torch::Tensor>& profile, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kQwen38Dim, kMax = kQwen38PrefillMaxTokens;
  const int64_t num_layers = linear.size(0) + full.size(0);
  TORCH_CHECK(full.size(0) == num_layers / kQwen38FullInterval,
              "every ", kQwen38FullInterval, "th of ", num_layers,
              " layers is full attention, got ", full.size(0));
  TORCH_CHECK(linear_mlp.size(0) == linear.size(0),
              "every linear-attention layer needs its MLP");
  const int sms = SmCount(at::cuda::current_device());
  TORCH_CHECK(kQwen38PrefillMinCtas <= num_ctas && num_ctas <= sms,
              "num_ctas ", num_ctas, " must be in [", kQwen38PrefillMinCtas,
              ", ", sms, "]");
  TORCH_CHECK(0 < tokens && tokens <= kMax, "tokens ", tokens, " out of [1, ",
              kMax, "]");
  CheckNumel(pos0, 1, "pos0");
  CheckNumel(positions, 3 * tokens, "positions");
  CheckNumel(final_norm, kD, "final_norm");
  // The workspace holds a whole chunk, whatever this one's tokens.
  CheckNumel(residual, kMax * kD, "residual");
  CheckNumel(h, kMax * kD, "h");
  CheckNumel(proj, kMax * kGdnInt4Rows, "proj");
  CheckNumel(beta, kMax * kGdnVHeads, "beta");
  CheckNumel(gate, kMax * kGdnVHeads, "gate");
  CheckNumel(query, kMax * kGdnKeyDim, "query");
  CheckNumel(key, kMax * kGdnKeyDim, "key");
  CheckNumel(value, kMax * kGdnValueDim, "value");
  CheckNumel(core, kMax * kGdnValueDim, "core");
  CheckNumel(attn, kMax * kQwen38QDim, "attn");
  CheckNumel(act, kMax * kQwen38Ffn, "act");
  const int64_t max_spans = partial_ml.numel() / (kMax * kQwen38QHeads * 2);
  CheckNumel(partial_ml, kMax * kQwen38QHeads * max_spans * 2, "partial_ml");
  CheckNumel(partial_o, kMax * kQwen38QHeads * max_spans * kQwen38HeadDim,
             "partial_o");
  TORCH_CHECK(max_spans > 0, "partial_ml holds no span");
  Qwen38PrefillParams p{};
  p.linear = StackedParams<GdnParams>(linear, "linear");
  p.linear_mlp = StackedParams<Qwen38MlpParams>(linear_mlp, "linear_mlp");
  p.full = StackedParams<Qwen38LayerParams>(full, "full");
  p.num_layers = static_cast<int>(num_layers);
  p.tokens = static_cast<int>(tokens);
  p.pos0 = Ptr<const int32_t>(pos0, torch::kInt32, "pos0");
  p.positions = Ptr<const int32_t>(positions, torch::kInt32, "positions");
  p.final_norm =
      Ptr<const __nv_bfloat16>(final_norm, torch::kBFloat16, "final_norm");
  p.lm_head = LmHeadOf(lm_head, &p.vocab);
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  const auto kF32 = torch::kFloat32;
  const auto kBf16 = torch::kBFloat16;
  p.residual = Ptr<float>(residual, kF32, "residual");
  p.h = Ptr<__nv_bfloat16>(h, kBf16, "h");
  p.proj = Ptr<float>(proj, kF32, "proj");
  p.beta = Ptr<float>(beta, kF32, "beta");
  p.gate = Ptr<float>(gate, kF32, "gate");
  p.query = Ptr<float>(query, kF32, "query");
  p.key = Ptr<float>(key, kF32, "key");
  p.value = Ptr<float>(value, kF32, "value");
  p.core = Ptr<float>(core, kF32, "core");
  p.attn = Ptr<__nv_bfloat16>(attn, kBf16, "attn");
  p.act = Ptr<__nv_bfloat16>(act, kBf16, "act");
  p.partial_ml = Ptr<float>(partial_ml, kF32, "partial_ml");
  p.partial_o = Ptr<float>(partial_o, kF32, "partial_o");
  p.max_spans = static_cast<int>(max_spans);
  if (logits.has_value()) {
    CheckNumel(*logits, p.vocab, "logits");
    p.logits = Ptr<float>(*logits, kF32, "logits");
  }
  if (hidden.has_value()) {
    CheckNumel(*hidden, (num_layers + 1) * tokens * kD, "hidden");
    p.hidden = Ptr<float>(*hidden, kF32, "hidden");
  }
  if (profile.has_value()) {
    CheckNumel(*profile,
               num_ctas * Qwen38PrefillBarriers(p.num_layers) * 2, "profile");
    p.profile = Ptr<int64_t>(*profile, torch::kInt64, "profile");
  }
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  const c10::cuda::CUDAGuard guard(residual.device());
  CheckOk(LaunchQwen38Prefill(p, static_cast<int>(num_ctas),
                              at::cuda::getCurrentCUDAStream()),
          "Qwen3.8 prefill launch");
}

// The params a decode step and a verify step share: the stacked layers, the
// embedding, the final norm, the LM head, and the barrier.
Qwen38DecodeParams Qwen38StepParams(
    const torch::Tensor& linear, const torch::Tensor& linear_mlp,
    const torch::Tensor& full, const torch::Tensor& embed,
    const torch::Tensor& final_norm, const std::vector<torch::Tensor>& lm_head,
    double eps, int64_t timeout_ns, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kQwen38Dim;
  const int64_t num_layers = linear.size(0) + full.size(0);
  TORCH_CHECK(full.size(0) == num_layers / kQwen38FullInterval,
              "every ", kQwen38FullInterval, "th of ", num_layers,
              " layers is full attention, got ", full.size(0));
  TORCH_CHECK(linear_mlp.size(0) == linear.size(0),
              "every linear-attention layer needs its MLP");
  CheckGdnGrid(num_ctas);
  CheckQwen38Grid(num_ctas);
  TORCH_CHECK(embed.dim() == 2 && embed.size(1) == kD, "embed must be [vocab, ",
              kD, "]");
  CheckNumel(final_norm, kD, "final_norm");
  Qwen38DecodeParams p{};
  p.linear = StackedParams<GdnParams>(linear, "linear");
  p.linear_mlp = StackedParams<Qwen38MlpParams>(linear_mlp, "linear_mlp");
  p.full = StackedParams<Qwen38LayerParams>(full, "full");
  p.num_layers = static_cast<int>(num_layers);
  p.embed = Ptr<const __nv_bfloat16>(embed, torch::kBFloat16, "embed");
  p.final_norm =
      Ptr<const __nv_bfloat16>(final_norm, torch::kBFloat16, "final_norm");
  p.lm_head = LmHeadOf(lm_head, &p.vocab);
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  return p;
}

void RunQwen38Decode(
    const torch::Tensor& linear, const torch::Tensor& linear_mlp,
    const torch::Tensor& full, const torch::Tensor& token,
    const torch::Tensor& pos, const torch::Tensor& positions,
    const torch::Tensor& embed, const torch::Tensor& final_norm,
    const std::vector<torch::Tensor>& lm_head, double eps, int64_t timeout_ns,
    const torch::Tensor& residual, const torch::Tensor& logits,
    const std::optional<torch::Tensor>& hidden, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kQwen38Dim;
  Qwen38DecodeParams p =
      Qwen38StepParams(linear, linear_mlp, full, embed, final_norm, lm_head,
                       eps, timeout_ns, sync, error, num_ctas);
  CheckNumel(token, 1, "token");
  CheckNumel(pos, 1, "pos");
  CheckNumel(positions, 3, "positions");
  CheckNumel(residual, kD, "residual");
  CheckNumel(logits, p.vocab, "logits");
  p.token = Ptr<const int32_t>(token, torch::kInt32, "token");
  p.pos = Ptr<const int32_t>(pos, torch::kInt32, "pos");
  p.positions = Ptr<const int32_t>(positions, torch::kInt32, "positions");
  p.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  p.logits = Ptr<float>(logits, torch::kFloat32, "logits");
  if (hidden.has_value()) {
    CheckNumel(*hidden, (p.num_layers + 1) * kD, "hidden");
    p.hidden = Ptr<float>(*hidden, torch::kFloat32, "hidden");
  }
  const c10::cuda::CUDAGuard guard(residual.device());
  CheckOk(LaunchQwen38Decode(p, static_cast<int>(num_ctas),
                             at::cuda::getCurrentCUDAStream()),
          "Qwen3.8 decode launch");
}

void RunQwen38Verify(
    const torch::Tensor& linear, const torch::Tensor& linear_mlp,
    const torch::Tensor& full, const torch::Tensor& tokens,
    const torch::Tensor& pos, const torch::Tensor& slot, int64_t num_slots,
    const torch::Tensor& embed, const torch::Tensor& final_norm,
    const std::vector<torch::Tensor>& lm_head, double eps, int64_t splits,
    int64_t timeout_ns, const torch::Tensor& residual,
    const torch::Tensor& logits, const torch::Tensor& final_hidden,
    const torch::Tensor& mixed, const torch::Tensor& z,
    const torch::Tensor& beta, const torch::Tensor& decay,
    const torch::Tensor& core, const torch::Tensor& qkv,
    const torch::Tensor& partial_ml, const torch::Tensor& partial_o,
    const torch::Tensor& act, const torch::Tensor& sync,
    const torch::Tensor& error, int64_t num_ctas) {
  constexpr int64_t kD = kQwen38Dim, kHd = kQwen38HeadDim;
  const int64_t n = tokens.numel();
  TORCH_CHECK(0 < n && n <= kQwen38MaxVerifyTokens, "a verify step takes 1 to ",
              kQwen38MaxVerifyTokens, " tokens, got ", n);
  TORCH_CHECK(num_slots > n, "num_slots ", num_slots, " must exceed the ", n,
              " tokens");
  const int64_t items = kQwen38QHeads * splits;
  Qwen38VerifyParams v{};
  v.step = Qwen38StepParams(linear, linear_mlp, full, embed, final_norm,
                            lm_head, eps, timeout_ns, sync, error, num_ctas);
  CheckNumel(pos, 1, "pos");
  CheckNumel(slot, 1, "slot");
  CheckNumel(residual, n * kD, "residual");
  CheckNumel(logits, n * v.step.vocab, "logits");
  CheckNumel(final_hidden, n * kD, "final_hidden");
  CheckNumel(mixed, n * kGdnConvDim, "mixed");
  CheckNumel(z, n * kGdnValueDim, "z");
  CheckNumel(beta, n * kGdnVHeads, "beta");
  CheckNumel(decay, n * kGdnVHeads, "decay");
  CheckNumel(core, n * kGdnValueDim, "core");
  CheckNumel(qkv, n * kQwen38QkvRows, "qkv");
  CheckNumel(partial_ml, n * items * 2, "partial_ml");
  CheckNumel(partial_o, n * items * kHd, "partial_o");
  CheckNumel(act, n * kQwen38Ffn, "act");
  v.step.token = Ptr<const int32_t>(tokens, torch::kInt32, "tokens");
  v.step.pos = Ptr<const int32_t>(pos, torch::kInt32, "pos");
  v.step.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  v.step.logits = Ptr<float>(logits, torch::kFloat32, "logits");
  v.num_tokens = static_cast<int>(n);
  v.num_slots = static_cast<int>(num_slots);
  v.slot = Ptr<const int32_t>(slot, torch::kInt32, "slot");
  v.mixed = Ptr<float>(mixed, torch::kFloat32, "mixed");
  v.z = Ptr<float>(z, torch::kFloat32, "z");
  v.beta = Ptr<float>(beta, torch::kFloat32, "beta");
  v.decay = Ptr<float>(decay, torch::kFloat32, "decay");
  v.core = Ptr<float>(core, torch::kFloat32, "core");
  v.qkv = Ptr<float>(qkv, torch::kFloat32, "qkv");
  v.partial_ml = Ptr<float>(partial_ml, torch::kFloat32, "partial_ml");
  v.partial_o = Ptr<float>(partial_o, torch::kFloat32, "partial_o");
  v.act = Ptr<__nv_bfloat16>(act, torch::kBFloat16, "act");
  v.final_hidden =
      Ptr<__nv_bfloat16>(final_hidden, torch::kBFloat16, "final_hidden");
  const c10::cuda::CUDAGuard guard(residual.device());
  CheckOk(LaunchQwen38Verify(v, static_cast<int>(num_ctas),
                             at::cuda::getCurrentCUDAStream()),
          "Qwen3.8 verify launch");
}

torch::Tensor Qwen38MtpParamsOf(
    const torch::Tensor& token, const torch::Tensor& pos,
    const torch::Tensor& hidden_in, const torch::Tensor& embed,
    const torch::Tensor& embed_norm, const torch::Tensor& hidden_norm,
    const torch::Tensor& fc, const torch::Tensor& layer,
    const torch::Tensor& final_norm, const std::vector<torch::Tensor>& lm_head,
    double eps, int64_t timeout_ns, const torch::Tensor& residual,
    const torch::Tensor& hidden_out, const std::optional<torch::Tensor>& logits,
    const torch::Tensor& sync, const torch::Tensor& error) {
  constexpr int64_t kD = kQwen38Dim;
  const auto kBf16 = torch::kBFloat16;
  CheckNumel(token, 1, "token");
  CheckNumel(pos, 1, "pos");
  CheckNumel(hidden_in, kD, "hidden_in");
  TORCH_CHECK(embed.dim() == 2 && embed.size(1) == kD, "embed must be [vocab, ",
              kD, "]");
  CheckNumel(embed_norm, kD, "embed_norm");
  CheckNumel(hidden_norm, kD, "hidden_norm");
  CheckNumel(fc, kD * 2 * kD, "fc");
  CheckNumel(final_norm, kD, "final_norm");
  CheckNumel(residual, kD, "residual");
  CheckNumel(hidden_out, kD, "hidden_out");
  Qwen38MtpParams p{};
  p.token = Ptr<const int32_t>(token, torch::kInt32, "token");
  p.pos = Ptr<const int32_t>(pos, torch::kInt32, "pos");
  p.hidden_in = Ptr<const __nv_bfloat16>(hidden_in, kBf16, "hidden_in");
  p.embed = Ptr<const __nv_bfloat16>(embed, kBf16, "embed");
  p.embed_norm = Ptr<const __nv_bfloat16>(embed_norm, kBf16, "embed_norm");
  p.hidden_norm = Ptr<const __nv_bfloat16>(hidden_norm, kBf16, "hidden_norm");
  p.fc = Bf16WeightOf(fc, 2 * kD, "fc");
  p.layer = StructFrom<Qwen38LayerParams>(layer, "layer");
  for (const Weight* w : {&p.layer.wqkv, &p.layer.wo, &p.layer.mlp.w13,
                          &p.layer.mlp.w2}) {
    TORCH_CHECK(w->format == WeightFormat::kBf16,
                "the MTP head reads its layer's projections as bf16");
  }
  p.final_norm = Ptr<const __nv_bfloat16>(final_norm, kBf16, "final_norm");
  p.lm_head = LmHeadOf(lm_head, &p.vocab);
  p.eps = static_cast<float>(eps);
  p.timeout_ns = timeout_ns;
  p.residual = Ptr<float>(residual, torch::kFloat32, "residual");
  p.hidden_out = Ptr<__nv_bfloat16>(hidden_out, kBf16, "hidden_out");
  if (logits.has_value()) {
    CheckNumel(*logits, p.vocab, "logits");
    p.logits = Ptr<float>(*logits, torch::kFloat32, "logits");
  }
  p.sync = Ptr<unsigned>(sync, torch::kInt32, "sync");
  p.error = MappedRecord(error);
  return StructBytes(p);
}

void RunQwen38Mtp(const torch::Tensor& params, int64_t num_ctas) {
  CheckGdnGrid(num_ctas);
  CheckQwen38Grid(num_ctas);
  CheckOk(LaunchQwen38Mtp(StructFrom<Qwen38MtpParams>(params, "params"),
                          static_cast<int>(num_ctas),
                          at::cuda::getCurrentCUDAStream()),
          "Qwen3.8 MTP launch");
}

}  // namespace
}  // namespace s2mk

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("run_gemv", &s2mk::RunGemv, "Runs a device-resident GEMV sequence.");
  m.def("gemv_smem_bytes", &s2mk::GemvSmemBytes,
        "Dynamic shared memory for a GEMV sequence.");
  m.def("run_int4_gemv", &s2mk::RunInt4Gemv,
        "y = W x for an int4 W, symmetric or with zero points.",
        py::arg("packed"), py::arg("scales"), py::arg("zeros"), py::arg("x"),
        py::arg("num_ctas"));
  m.def("run_int4_mma", &s2mk::RunInt4Mma,
        "y = X W^T on the prefills' tensor-core tile, for an int4 W, "
        "symmetric or with zero points, or a bf16 W.",
        py::arg("packed"), py::arg("scales"), py::arg("zeros"),
        py::arg("bf16"), py::arg("x"));
  m.def("thinker_attention_params", &s2mk::ThinkerAttentionParamsOf,
        "The thinker attention block's launch params, as CPU bytes.",
        py::arg("residual_in"), py::arg("norm"), py::arg("wqkv_packed"),
        py::arg("wqkv_scales"), py::arg("q_norm"), py::arg("k_norm"),
        py::arg("wo_packed"), py::arg("wo_scales"), py::arg("cos_sin"),
        py::arg("positions"), py::arg("key_cache"), py::arg("value_cache"),
        py::arg("block_table"), py::arg("pos"), py::arg("splits"),
        py::arg("eps"), py::arg("timeout_ns"), py::arg("qkv"),
        py::arg("partial_ml"), py::arg("partial_o"), py::arg("residual"),
        py::arg("sync"), py::arg("error"), py::arg("num_ctas"));
  m.def("run_thinker_attention", &s2mk::RunThinkerAttention,
        "One token through Qwen3-Omni's thinker attention block, one launch.",
        py::arg("params"), py::arg("num_ctas"));
  m.def("qwen38_layer_params", &s2mk::Qwen38LayerParamsOf,
        "The Qwen3.8 full-attention layer's launch params, as CPU bytes.",
        py::arg("residual_in"), py::arg("input_norm"), py::arg("wqkv"),
        py::arg("q_norm"), py::arg("k_norm"), py::arg("wo"), py::arg("cos_sin"),
        py::arg("positions"), py::arg("key_cache"), py::arg("value_cache"),
        py::arg("block_table"), py::arg("pos"), py::arg("splits"),
        py::arg("post_norm"), py::arg("w13"), py::arg("w2"), py::arg("eps"),
        py::arg("timeout_ns"), py::arg("qkv"), py::arg("partial_ml"),
        py::arg("partial_o"), py::arg("act"), py::arg("hidden"),
        py::arg("residual"), py::arg("sync"), py::arg("error"),
        py::arg("num_ctas"));
  m.def("run_qwen38_layer", &s2mk::RunQwen38Layer,
        "One token through a Qwen3.8 full-attention layer, one launch.",
        py::arg("params"), py::arg("num_ctas"));
  m.def("run_barrier_probe", &s2mk::RunBarrierProbe,
        "Runs the grid-barrier probe.", py::arg("rounds"),
        py::arg("skip_cta"), py::arg("skip_barrier"), py::arg("step"),
        py::arg("timeout_ns"), py::arg("sync"), py::arg("error"),
        py::arg("slots"), py::arg("mismatches"), py::arg("num_ctas"));
  m.def("gdn_params", &s2mk::GdnParamsOf,
        "The Qwen3.8 linear-attention layer's launch params, as CPU bytes.",
        py::arg("residual_in"), py::arg("norm"), py::arg("in_proj"),
        py::arg("gates"), py::arg("conv"), py::arg("a_log"),
        py::arg("dt_bias"), py::arg("out_norm"), py::arg("out_proj"),
        py::arg("eps"), py::arg("timeout_ns"),
        py::arg("conv_state"), py::arg("state"), py::arg("mixed"),
        py::arg("z"), py::arg("beta"), py::arg("decay"), py::arg("core"),
        py::arg("residual"), py::arg("sync"), py::arg("error"));
  m.def("run_gdn", &s2mk::RunGdn,
        "One token through a Qwen3.8 linear-attention layer, one launch.",
        py::arg("params"), py::arg("num_ctas"));
  m.def("qwen38_mlp_params", &s2mk::Qwen38MlpParamsOf,
        "A Qwen3.8 layer's dense MLP block params, as CPU bytes.",
        py::arg("post_norm"), py::arg("w13"), py::arg("w2"), py::arg("eps"),
        py::arg("act"));
  m.def("run_qwen38_decode", &s2mk::RunQwen38Decode,
        "One decode step of Qwen3.8-27B's language model, one launch.",
        py::arg("linear"), py::arg("linear_mlp"), py::arg("full"),
        py::arg("token"), py::arg("pos"), py::arg("positions"),
        py::arg("embed"), py::arg("final_norm"), py::arg("lm_head"),
        py::arg("eps"), py::arg("timeout_ns"), py::arg("residual"),
        py::arg("logits"), py::arg("hidden"), py::arg("sync"),
        py::arg("error"), py::arg("num_ctas"));
  m.def("run_qwen38_prefill", &s2mk::RunQwen38Prefill,
        "One prefill chunk of Qwen3.8-27B's language model, one launch.",
        py::arg("linear"), py::arg("linear_mlp"), py::arg("full"),
        py::arg("tokens"), py::arg("pos0"), py::arg("positions"),
        py::arg("final_norm"), py::arg("lm_head"), py::arg("eps"),
        py::arg("timeout_ns"), py::arg("residual"), py::arg("h"),
        py::arg("proj"), py::arg("beta"), py::arg("gate"), py::arg("query"),
        py::arg("key"), py::arg("value"), py::arg("core"), py::arg("attn"),
        py::arg("act"), py::arg("partial_ml"), py::arg("partial_o"),
        py::arg("logits"), py::arg("hidden"),
        py::arg("profile"), py::arg("sync"), py::arg("error"),
        py::arg("num_ctas"));
  m.def("run_qwen38_projection", &s2mk::RunQwen38Projection,
        "y = X W^T for an int4 W with zero points, on the Qwen3.8 prefill's "
        "dense phase.",
        py::arg("weight"), py::arg("x"), py::arg("whole_k"),
        py::arg("num_ctas"));
  m.def("qwen38_prefill_barriers", &s2mk::Qwen38PrefillBarriers,
        "Grid barriers a Qwen3.8 prefill launch over `num_layers` layers "
        "takes.",
        py::arg("num_layers"));
  m.def("run_qwen38_verify", &s2mk::RunQwen38Verify,
        "Up to QWEN38_MAX_VERIFY_TOKENS consecutive Qwen3.8-27B tokens, one "
        "launch.",
        py::arg("linear"), py::arg("linear_mlp"), py::arg("full"),
        py::arg("tokens"), py::arg("pos"), py::arg("slot"),
        py::arg("num_slots"), py::arg("embed"), py::arg("final_norm"),
        py::arg("lm_head"), py::arg("eps"), py::arg("splits"),
        py::arg("timeout_ns"), py::arg("residual"), py::arg("logits"),
        py::arg("final_hidden"), py::arg("mixed"), py::arg("z"),
        py::arg("beta"), py::arg("decay"), py::arg("core"), py::arg("qkv"),
        py::arg("partial_ml"), py::arg("partial_o"), py::arg("act"),
        py::arg("sync"), py::arg("error"), py::arg("num_ctas"));
  m.def("qwen38_mtp_params", &s2mk::Qwen38MtpParamsOf,
        "The Qwen3.8 MTP head's launch params, as CPU bytes.",
        py::arg("token"), py::arg("pos"), py::arg("hidden_in"),
        py::arg("embed"), py::arg("embed_norm"), py::arg("hidden_norm"),
        py::arg("fc"), py::arg("layer"), py::arg("final_norm"),
        py::arg("lm_head"), py::arg("eps"), py::arg("timeout_ns"),
        py::arg("residual"), py::arg("hidden_out"), py::arg("logits"),
        py::arg("sync"), py::arg("error"));
  m.def("run_qwen38_mtp", &s2mk::RunQwen38Mtp,
        "One step of Qwen3.8-27B's MTP head, one launch.", py::arg("params"),
        py::arg("num_ctas"));
  m.attr("QWEN38_MAX_VERIFY_TOKENS") = py::int_(s2mk::kQwen38MaxVerifyTokens);
  m.attr("ERROR_RECORD_WORDS") =
      py::int_(sizeof(s2mk::ErrorRecord) / sizeof(int32_t));
  m.attr("SYNC_WORDS") = py::int_(s2mk::kSyncWords);
  m.attr("PROBE_WORDS_PER_CTA") = py::int_(s2mk::kProbeWordsPerCta);
  m.attr("PROFILE_HEADER_WORDS") = py::int_(s2mk::kProfileHeader);
  m.attr("MAX_HISTORY") = py::int_(s2mk::kMaxHistory);
  m.attr("CP_PROFILE_WORDS") = py::int_(s2mk::kCpProfileWords);
  m.def("decode_profile_words", &s2mk::DecodeProfileWords,
        "Profile words per CTA for a decode launch of `num_steps` steps.",
        py::arg("num_steps"));
  m.def("profile_words", &s2mk::ProfileWords,
        "Profile stamp words per CTA for a launch of `num_layers` layers.",
        py::arg("num_layers"));
}
