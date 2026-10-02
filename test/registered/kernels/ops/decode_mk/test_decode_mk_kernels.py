# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""The vendored decode-mk kernels (sglang/kernels/decode_mk) as SGLang builds
them, on random Qwen3.8 weights over the first 8 layers (two periods of three
linear layers and a full one): a decode step's layers and logits against an
fp32 PyTorch model on the dequantized weights, bounded by the same model in
bf16; the verify step against decode steps, bit for bit; and the prefill's
logits against the fp32 model. decode-mk's own tier-0 tests hold the kernels
at all 64 layers and more shapes; these hold the vendored copy and its
build."""

import dataclasses

import pytest
import torch

from sglang.kernels.decode_mk.qwen38_decode import (
    Qwen38Decoder,
    Qwen38State,
    Qwen38Weights,
    reference_step,
)
from sglang.kernels.decode_mk.qwen38_layer import cos_sin_table
from sglang.kernels.decode_mk.qwen38_prefill import Qwen38Prefiller
from sglang.kernels.decode_mk.qwen38_verify import (
    MAX_TOKENS,
    Qwen38Verifier,
    SlottedState,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="sm_120a kernels: needs an RTX 5090, which CI does not have",
)

LAYERS = 8
MAX_POSITIONS = 256
# decode-mk's error budget: the kernel's relative L2 error to fp32 within
# 1.25 × that of the same model run in bf16.
FACTOR = 1.25

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="the kernels are built for sm_120a",
)


@pytest.fixture(scope="module")
def weights():
    w = Qwen38Weights.random(seed=0)
    return dataclasses.replace(w, layers=w.layers[:LAYERS])


@pytest.fixture(scope="module")
def cos_sin():
    return cos_sin_table(MAX_POSITIONS)


def _tokens(vocab: int, n: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(vocab, (n,), generator=gen).tolist()


def _rel_l2(x: torch.Tensor, ref: torch.Tensor) -> float:
    x, ref = x.float(), ref.float()
    return float((x - ref).norm() / ref.norm())


def _within_budget(name, got, budget, truth) -> None:
    error, allowed = _rel_l2(got, truth), _rel_l2(budget, truth)
    assert error <= FACTOR * allowed, (
        f"{name}: error {error:.3g} exceeds {FACTOR} × bf16's {allowed:.3g}"
    )


def _positions(pos: int) -> torch.Tensor:
    return torch.tensor([pos] * 3, dtype=torch.int32, device="cuda")


def test_decode_steps_within_bf16_budget(weights, cos_sin):
    state = Qwen38State.zeros(LAYERS, MAX_POSITIONS)
    truth_state, budget_state = state.clone(), state.clone()
    decoder = Qwen38Decoder(weights, state, cos_sin)
    hidden = torch.empty(LAYERS + 1, weights.embed.shape[1], device="cuda")
    for pos, token in enumerate(_tokens(weights.vocab, 4, seed=1)):
        truth, truth_hidden = reference_step(
            weights, truth_state, token, pos, _positions(pos), cos_sin, torch.float32
        )
        budget, budget_hidden = reference_step(
            weights, budget_state, token, pos, _positions(pos), cos_sin, torch.bfloat16
        )
        logits = decoder.step(token, pos, hidden=hidden)
        for i in range(1, LAYERS + 1):
            _within_budget(
                f"step {pos}, layer {i - 1}",
                hidden[i],
                budget_hidden[i],
                truth_hidden[i],
            )
        _within_budget(f"step {pos}, logits", logits, budget, truth)


@pytest.mark.parametrize("n", range(1, MAX_TOKENS + 1))
def test_verify_step_is_decode_steps(weights, cos_sin, n):
    """The verify step speculative decoding rests on: n tokens in one launch
    give each token's logits and states exactly as n decode steps do."""
    start = 5
    prefix = _tokens(weights.vocab, start, seed=2)
    tokens = _tokens(weights.vocab, n, seed=3)

    state = Qwen38State.zeros(LAYERS, MAX_POSITIONS)
    decoder = Qwen38Decoder(weights, state, cos_sin)
    for pos, token in enumerate(prefix):
        decoder.step(token, pos)
    slotted = SlottedState.of(state.clone(), MAX_TOKENS + 1)
    expected, states = [], []
    for t, token in enumerate(tokens):
        expected.append(decoder.step(token, start + t).clone())
        states.append([s.clone() for s in state.linear])

    verifier = Qwen38Verifier(weights, slotted, cos_sin)
    scalar = lambda v: torch.tensor([v], dtype=torch.int32, device="cuda")
    verifier.launch(
        torch.tensor(tokens, dtype=torch.int32, device="cuda"), scalar(start), scalar(0)
    )
    torch.cuda.synchronize()
    assert torch.equal(verifier.logits[:n], torch.stack(expected))
    for t in range(n):
        for j, s in enumerate(states[t]):
            assert torch.equal(slotted.conv[j][t + 1], s.conv)
            assert torch.equal(slotted.recurrent[j][t + 1], s.recurrent)


def test_prefill_within_bf16_budget(weights, cos_sin):
    """A 70-token prompt, a full 64-token chunk and a partial one, against
    the fp32 model fed the same tokens one at a time."""
    prompt = _tokens(weights.vocab, 70, seed=4)
    truth_state = Qwen38State.zeros(LAYERS, MAX_POSITIONS)
    budget_state = truth_state.clone()
    for pos, token in enumerate(prompt):
        truth, _ = reference_step(
            weights, truth_state, token, pos, _positions(pos), cos_sin, torch.float32
        )
        budget, _ = reference_step(
            weights, budget_state, token, pos, _positions(pos), cos_sin, torch.bfloat16
        )
    decoder = Qwen38Decoder(weights, Qwen38State.zeros(LAYERS, MAX_POSITIONS), cos_sin)
    logits = Qwen38Prefiller(decoder).run(prompt)
    _within_budget("prompt logits", logits, budget, truth)
