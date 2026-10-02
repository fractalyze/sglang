// Copyright 2026 Fractalyze Inc. All rights reserved.

#ifndef S2MK_CSRC_SAMPLER_H_
#define S2MK_CSRC_SAMPLER_H_

#include <cuda_runtime.h>

#include <cstdint>

namespace s2mk {

// ref/sampler.py holds the same numbers.
constexpr int kImEndId = 151645;
constexpr int kSemanticBeginId = 151678;
constexpr int kSemanticEndId = 155773;
// <|im_end|> and the semantic tokens, in ascending vocabulary order.
constexpr int kNumSampleable = 1 + kSemanticEndId - kSemanticBeginId + 1;
constexpr int kGraphTopK = 30;
constexpr int kRasLookback = 4;
// The longest repetition window the sampler keeps.
constexpr int kMaxHistory = 64;

// Production's per-request sampling settings (ref.sampler.SamplingParams).
struct SamplingParams {
  int64_t seed;
  float temperature;
  float top_p;
  int top_k;
  float repetition_penalty;
  float ras_temperature;
  float ras_top_p;
  int history_len;  // at most kMaxHistory
};

// Draws one token per case with the megakernel's sampler, one CTA per case,
// for the tier-0 test.
struct SamplerProbeArgs {
  SamplingParams params;
  const float* logits;  // [cases, kNumSampleable]
  const int64_t* history;  // [cases, kMaxHistory]: oldest first
  const int64_t* history_count;  // [cases]
  const int64_t* sample_step;  // [cases]
  int64_t* tokens;  // [cases]: vocabulary ids
};

cudaError_t LaunchSamplerProbe(const SamplerProbeArgs& args, int cases,
                               cudaStream_t stream);

}  // namespace s2mk

#endif  // S2MK_CSRC_SAMPLER_H_
