// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_QWEN38_PREFILL_H_
#define S2MK_CSRC_QWEN38_PREFILL_H_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "barrier.h"
#include "gdn.h"
#include "qwen38_decode.h"
#include "qwen38_layer.h"
#include "weight.h"

namespace s2mk {

// A chunk of a prompt's tokens: 8 n-tiles of mma.m16n8k16, the delta rule's
// chunk and an attention query block. A launch takes a prompt of any number
// of chunks, each phase over all of them.
constexpr int kQwen38PrefillChunk = 64;

// The chunks of `tokens` tokens, the last one partial.
__host__ __device__ constexpr int Qwen38PrefillChunks(int tokens) {
  return (tokens + kQwen38PrefillChunk - 1) / kQwen38PrefillChunk;
}

// Grid barriers a linear-attention layer and a full-attention layer take, one
// after each of their phases, the MLP's included.
constexpr int kQwen38PrefillLinearBarriers = 10;
constexpr int kQwen38PrefillFullBarriers = 9;

// The delta rule's intra-chunk matrices of a launch of `tokens` tokens, in
// floats: T = (I + A)⁻¹ and M, kQwen38PrefillChunk² each, for every (chunk,
// value head). They sit in the act workspace, which no phase reads between
// the down projection and the next layer's gate-up.
__host__ __device__ constexpr int64_t Qwen38PrefillDeltaFloats(int tokens) {
  return int64_t{Qwen38PrefillChunks(tokens)} * kGdnVHeads * 2 *
         kQwen38PrefillChunk * kQwen38PrefillChunk;
}

// Cache positions an attention partial covers when a launch takes one chunk:
// a full layer splits each token's keys into spans of this many, merged
// after, so a long cache spreads over the grid. A launch of several chunks
// gives each (chunk, query head) a CTA over its keys whole, one span.
constexpr int kQwen38PrefillAttnSpan = 512;

// The int4 GEMM's tiles (qwen38_prefill_gemm.cuh): kQwen38GemmTileRows rows
// over k-tiles of kQwen38GemmTileK columns.
constexpr int kQwen38GemmTileRows = 128;
constexpr int kQwen38GemmTileK = 128;
// The grid the GEMM's work balances for: an RTX 5090's SMs. Any grid gets
// the same splits, so the same sums; only their balance differs.
constexpr int kQwen38GemmBalancedCtas = 170;
constexpr int kQwen38GemmMaxSplits = 8;

// The k splits each row tile of an n × k int4 GEMM takes, 1 for none. Whole
// k, each CTA takes a run of 32-row blocks; a CTA's k-tile over r rows costs
// a quarter of a full tile's fixed plus r / kQwen38GemmTileRows of the rest.
// Split, a round takes whole tiles and costs about two k-tiles to fold. The
// cheaper wins for the busiest of kQwen38GemmBalancedCtas CTAs, in
// sixteenths of a full tile's k-tile.
__host__ __device__ constexpr int Qwen38GemmSplits(int n, int k) {
  const int tiles = n / kQwen38GemmTileRows;
  const int k_tiles = k / kQwen38GemmTileK;
  const int blocks = n / 32;
  const int rows = (blocks + kQwen38GemmBalancedCtas - 1) /
                   kQwen38GemmBalancedCtas * 32;
  int best = 1;
  int best_load = k_tiles * (4 + 12 * rows / kQwen38GemmTileRows);
  for (int splits = 2; splits <= kQwen38GemmMaxSplits; ++splits) {
    const int per_round = kQwen38GemmBalancedCtas / splits;
    const int rounds = (tiles + per_round - 1) / per_round;
    const int load = rounds * ((k_tiles + splits - 1) / splits + 2) * 16;
    if (load < best_load) {
      best = splits;
      best_load = load;
    }
  }
  return best;
}

// Bytes of x a band of chunks may read, half an RTX 5090's L2, so a
// band's x stays there while every row tile works over it.
constexpr int64_t kQwen38GemmBandBytes = int64_t{48} << 20;

// The chunks each band of an int4 GEMM over `chunks` chunks at k columns
// takes, the last band the rest: the fewest bands whose x fits
// kQwen38GemmBandBytes, as even as they go. Each band reads every weight
// tile again, so x that fits whole keeps one band.
__host__ __device__ constexpr int Qwen38GemmBandChunks(int k, int chunks) {
  const int64_t chunk_bytes = int64_t{kQwen38PrefillChunk} * k * 2;
  const int64_t most = kQwen38GemmBandBytes / chunk_bytes;
  const int widest = most > 1 ? static_cast<int>(most) : 1;
  const int bands = (chunks + widest - 1) / widest;
  return (chunks + bands - 1) / bands;
}

// The GEMM workspace an n × k int4 GEMM takes: a partial of each split of a
// tile that splits, kQwen38GemmTileRows × kQwen38PrefillChunk floats each, and
// two counter words a tile. Only a launch of one chunk splits.
__host__ __device__ constexpr int64_t Qwen38GemmPartialFloats(int n, int k) {
  const int splits = Qwen38GemmSplits(n, k);
  return splits == 1 ? 0
                     : int64_t{n} * splits * kQwen38PrefillChunk;
}

__host__ __device__ constexpr int Qwen38GemmCounters(int n) {
  return 2 * (n / kQwen38GemmTileRows);
}

// The prefill's int4 GEMMs, n × k: in_proj, q k v and gate, gate-up, then the
// projections back to the residual from attention and from the MLP.
constexpr int kQwen38PrefillGemms[][2] = {
    {kGdnInt4Rows, kQwen38Dim},
    {kQwen38QkvRows, kQwen38Dim},
    {2 * kQwen38Ffn, kQwen38Dim},
    {kQwen38Dim, kQwen38QDim},
    {kQwen38Dim, kQwen38Ffn},
};
static_assert(kGdnValueDim == kQwen38QDim, "out_proj is o_proj's shape");

// The GEMM workspace the largest of the prefill's GEMMs takes.
constexpr int64_t kQwen38PrefillGemmPartialFloats = [] {
  int64_t most = 0;
  for (const auto& gemm : kQwen38PrefillGemms) {
    const int64_t floats = Qwen38GemmPartialFloats(gemm[0], gemm[1]);
    most = floats > most ? floats : most;
  }
  return most;
}();
constexpr int kQwen38PrefillGemmCounters = [] {
  int most = 0;
  for (const auto& gemm : kQwen38PrefillGemms) {
    const int counters = Qwen38GemmCounters(gemm[0]);
    most = counters > most ? counters : most;
  }
  return most;
}();

// Grid barriers a prefill launch over `num_layers` layers takes.
__host__ __device__ constexpr int Qwen38PrefillBarriers(int num_layers) {
  const int full = num_layers / kQwen38FullInterval;
  return full * kQwen38PrefillFullBarriers +
         (num_layers - full) * kQwen38PrefillLinearBarriers;
}

// A prompt's pass through Qwen3.8-27B's language model in one launch, layer
// by layer: each layer's phases run over all `tokens` prompt tokens on tensor
// cores (the int4 projections in qwen38_prefill_gemm.cuh, the bf16 ones in
// prefill_dense.cuh), each projection one GEMM over every token. Then, when
// `logits` is set, the final norm and the LM head for the last token.
//
// The tokens sit at cache positions pos0[0] on. Each linear-attention layer
// runs the chunked gated delta rule from its conv and recurrent states, its
// chunks in order, and leaves them where token-by-token decode would; each
// full-attention layer writes the tokens' keys and values to its paged cache
// and attends causally over the cache. So a Qwen38DecodeParams over the same
// layer params decodes on from the state the prompt leaves.
//
// The layers' weights, states and caches come from the decode's block params
// (qwen38_decode.h); their workspace and residual fields are unused.
//
// Every sum has one order, whichever CTA takes it, so the result is bitwise
// deterministic and the same at every CTA count.
struct Qwen38PrefillParams {
  const GdnParams* linear;  // [linear layers], on the device
  const Qwen38MlpParams* linear_mlp;  // [linear layers], on the device
  const Qwen38LayerParams* full;  // [full layers], on the device
  int num_layers;
  int tokens;  // ≥ 1; the workspace holds them
  const int32_t* pos0;  // [1]: the first token's cache position
  // [3, tokens]: each token's M-RoPE positions, temporal, height, width.
  const int32_t* positions;

  const __nv_bfloat16* final_norm;  // [kQwen38Dim], applied as 1 + w
  Weight lm_head;  // kBf16 or kInt4Zp [vocab, kQwen38Dim]
  int vocab;
  float eps;
  int64_t timeout_ns;

  // The tokens' embeddings in, as fp32; each layer's residual after.
  float* residual;  // [tokens, kQwen38Dim]
  // Workspace.
  __nv_bfloat16* h;  // [tokens, kQwen38Dim]: a norm's output
  // [tokens, kGdnInt4Rows]: the linear layers' q, k, v (before the conv)
  // and z; the full layers' q, k, v and gate rows (kQwen38QkvRows).
  float* proj;
  float* beta;  // [tokens, kGdnVHeads]: sigmoid(b)
  // [tokens, kGdnVHeads]: g, the log decay; the delta rule's chunk phase
  // replaces each chunk's with their running sum.
  float* gate;
  // [tokens, kGdnKHeads, kGdnHeadDim]: after the conv, SiLU and L2 norm, the
  // query also scaled by kGdnHeadDim^(-1/2); rounded to tf32.
  float* query;
  float* key;
  float* value;  // [tokens, kGdnValueDim]: after the conv and SiLU
  float* core;  // [tokens, kGdnValueDim]: the delta rule's readout
  // [tokens, kQwen38QDim]: the gated RMSNorm's output (linear layers), or
  // the gated attention output (full layers).
  __nv_bfloat16* attn;
  // [tokens, kQwen38Ffn]: SiLU(gate) × up; during a linear layer's delta
  // rule, its intra-chunk matrices, Qwen38PrefillDeltaFloats(tokens) floats,
  // so it holds at least twice that many bf16s.
  __nv_bfloat16* act;
  // Each (token, query head)'s attention partial over each span of its keys
  // (kQwen38PrefillAttnSpan): [tokens, kQwen38QHeads, spans] of (running max,
  // sum), and of kQwen38HeadDim unnormalized outputs, where spans cover every
  // position the launch reaches: one chunk's spans of kQwen38PrefillAttnSpan,
  // or one for several chunks.
  float* partial_ml;
  float* partial_o;
  // The int4 GEMM's workspace: kQwen38PrefillGemmPartialFloats floats and
  // kQwen38PrefillGemmCounters counters, zero before the first launch, as
  // every launch leaves them.
  float* gemm_partials;
  unsigned* gemm_counters;

  // Outputs.
  float* logits;  // [vocab]: the last token's; null skips the LM head
  // [num_layers + 1, tokens, kQwen38Dim]: the residual entering each layer
  // and the last layer's output; null skips it.
  float* hidden;
  // [tokens, kQwen38Dim]: the last layer's output alone; null skips it.
  float* last_hidden;
  // [num_ctas, Qwen38PrefillBarriers(num_layers), 2]: each CTA's globaltimer
  // on arriving at and leaving every grid barrier; null skips it.
  int64_t* profile;
  unsigned* sync;  // kSyncWords words
  ErrorRecord* error;  // device view of the host-mapped record
};

// The fewest CTAs a prefill launch may use: one per (value head, half of its
// state columns) for the delta rule. A split GEMM's tile needs a CTA a split.
constexpr int kQwen38PrefillMinCtas = 2 * kGdnVHeads;
static_assert(kQwen38PrefillMinCtas >= kQwen38GemmMaxSplits);

// A plain launch of `num_ctas` CTAs, one per SM at most and at least
// kQwen38PrefillMinCtas.
cudaError_t LaunchQwen38Prefill(const Qwen38PrefillParams& params,
                                int num_ctas, cudaStream_t stream);

// One of the prefill's int4 projections alone, on the GEMM the prefill runs
// it with: y = x W^T in fp32 for `tokens` bf16 rows x ([tokens, k]) and an
// n-row kInt4Zp W of width k, kQwen38Dim, kQwen38QDim or kQwen38Ffn, n a
// multiple of kQwen38GemmTileRows, on at least Qwen38GemmSplits(n, k) CTAs.
// y is [tokens, n].
// `partials` holds Qwen38GemmPartialFloats(n, k) floats and `counters`
// Qwen38GemmCounters(n), zero before the first launch, as each leaves them.
cudaError_t LaunchQwen38Projection(const Weight& w, int n, int k,
                                   const __nv_bfloat16* x, int tokens,
                                   float* y, float* partials,
                                   unsigned* counters, int num_ctas,
                                   cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_QWEN38_PREFILL_H_
