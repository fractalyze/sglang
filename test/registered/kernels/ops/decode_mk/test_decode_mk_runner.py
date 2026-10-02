# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""DecodeMkRunner (sglang/srt/models/qwen3_5_decode_mk.py), the request
state SGLang drives the kernels through, on random Qwen3.8 weights over the
first 8 layers: its greedy replies against decode-mk's own Qwen38Generator, a
prompt in chunks against the prompt whole, a request after another, and
speculative decoding against plain decoding, token for token."""

import dataclasses

import pytest
import torch

from sglang.kernels.decode_mk.qwen38_decode import Qwen38Weights
from sglang.kernels.decode_mk.qwen38_mtp import MtpWeights
from sglang.kernels.decode_mk.qwen38_prefill import MAX_TOKENS as PREFILL_TOKENS
from sglang.kernels.decode_mk.qwen38_spec import MAX_DRAFTS, Qwen38Generator
from sglang.srt.models.qwen3_5_decode_mk import DecodeMkRunner
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="sm_120a kernels: needs an RTX 5090, which CI does not have",
)

LAYERS = 8
MAX_POSITIONS = 512
REPLY = 24

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="the kernels are built for sm_120a",
)


@pytest.fixture(scope="module")
def weights():
    w = Qwen38Weights.random(seed=0)
    return dataclasses.replace(w, layers=w.layers[:LAYERS])


@pytest.fixture(scope="module")
def mtp():
    return MtpWeights.random(seed=7)


def _prompt(vocab: int, n: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(vocab, (n,), generator=gen).tolist()


def _ids(tokens: list[int]) -> torch.Tensor:
    return torch.tensor(tokens, device="cuda")


def _plain(runner: DecodeMkRunner, prompt: list[int], chunks=None) -> list[int]:
    """The greedy reply as SGLang drives a request without speculation: the
    prompt in `chunks` (default whole), then one decode step a token."""
    bounds = [0, *(chunks or []), len(prompt)]
    for begin, end in zip(bounds, bounds[1:]):
        logits = runner.extend(_ids(prompt[begin:end]), begin)
    reply = [int(logits.argmax())]
    while len(reply) < REPLY:
        pos = len(prompt) + len(reply) - 1
        logits = runner.decode(_ids(reply[-1:]), _ids([pos]))
        reply.append(int(logits.argmax()))
    return reply


def _speculative(runner: DecodeMkRunner, prompt: list[int], chunks=None) -> list[int]:
    """The greedy reply as DECODE_MK_MTP's worker drives one."""
    bounds = [0, *(chunks or []), len(prompt)]
    for begin, end in zip(bounds, bounds[1:]):
        logits = runner.extend(_ids(prompt[begin:end]), begin)
    reply = [int(logits.argmax())]
    while len(reply) < REPLY:
        pos = len(prompt) + len(reply) - 1
        reply += runner.speculate(reply[-1], pos)
    return reply[:REPLY]


@pytest.fixture(scope="module")
def prompt(weights):
    return _prompt(weights.vocab, 2 * PREFILL_TOKENS + 9, seed=1)


@pytest.fixture(scope="module")
def plain(weights):
    return DecodeMkRunner(weights, MAX_POSITIONS, prefill=False)


@pytest.fixture(scope="module")
def prefilled(weights):
    return DecodeMkRunner(weights, MAX_POSITIONS, prefill=True)


def test_reply_is_decode_mks(weights, plain, prompt):
    """Token by token, the prompt leaves the states decode-mk's generator's
    verify steps leave, which decode steps then continue bit for bit."""
    generator = Qwen38Generator(weights, MAX_POSITIONS)
    expected = generator.generate(prompt, REPLY).tokens
    assert _plain(plain, prompt) == expected


@pytest.mark.parametrize("runner", ["plain", "prefilled"])
def test_chunked_prompt_is_the_whole_prompt(request, runner, prompt):
    """SGLang's chunked prefill hands the prompt over in pieces, each from
    the position the last one ended at; at the prefill kernel's own chunk
    boundaries that is the same launches."""
    runner = request.getfixturevalue(runner)
    whole = _plain(runner, prompt)
    assert _plain(runner, prompt, chunks=[PREFILL_TOKENS, 2 * PREFILL_TOKENS]) == whole


def test_a_request_starts_afresh(weights, plain, prompt):
    first = _plain(plain, prompt)
    _plain(plain, _prompt(weights.vocab, 40, seed=2))
    assert _plain(plain, prompt) == first


@pytest.mark.parametrize("drafts", range(1, MAX_DRAFTS + 1))
@pytest.mark.parametrize("prefill", [False, True])
def test_speculation_is_plain_decoding(weights, mtp, prompt, drafts, prefill, request):
    """Each cycle emits the accepted drafts and the model's own next token,
    all of them what plain decoding from the same prompt states emits."""
    plain = request.getfixturevalue("prefilled" if prefill else "plain")
    expected = _plain(plain, prompt)
    runner = DecodeMkRunner(weights, MAX_POSITIONS, prefill, mtp, drafts)
    assert _speculative(runner, prompt) == expected
    assert _speculative(runner, prompt, chunks=[PREFILL_TOKENS]) == expected


def test_speculation_after_another_request(weights, mtp, plain, prompt):
    expected = _plain(plain, prompt)
    runner = DecodeMkRunner(weights, MAX_POSITIONS, False, mtp, MAX_DRAFTS)
    _speculative(runner, _prompt(weights.vocab, 40, seed=2))
    assert _speculative(runner, prompt) == expected


def test_extend_refuses_a_prompt_past_the_caches(weights, plain):
    with pytest.raises(ValueError, match="past the kernels'"):
        plain.extend(_ids(_prompt(weights.vocab, 8, seed=3)), MAX_POSITIONS - 4)
