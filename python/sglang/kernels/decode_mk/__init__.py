# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""decode-mk's Qwen3.8-27B megakernels: batch-1 decode, the 64-token prompt
prefill, and the verify step and MTP head of speculative decoding, for one
RTX 5090 (sm_120a) on the int4 checkpoint cyankiwi/Qwen3.8-27B-AWQ-INT4.

Vendored from fractalyze/decode-mk at the commit VENDORED names, by
scripts/sync_decode_mk.py; edit them there, not here. _ext.py, SGLang's own,
JIT-builds csrc/ on first use. sglang/srt/models/qwen3_5_decode_mk.py serves
the model on them behind the SGLANG_DECODE_MK_* switches.
"""
