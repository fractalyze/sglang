# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""DecodeMkRunner (sglang/srt/models/qwen3_5_decode_mk.py), the request
state SGLang drives the kernels through, on random Qwen3.8 weights over the
first 8 layers, token for token against decode-mk's own API: its greedy
replies against Qwen38Generator's, and on the prefill kernel against
Qwen38Prefiller's over the whole prompt, past one launch's tokens too; a
prompt in chunks against the prompt whole; a request after another; and
speculative decoding against the verify step fed one token a step from the
prompt's states, which is the text decode-mk's start/cycle emits."""

import dataclasses

import pytest
import torch

from sglang.kernels.decode_mk.qwen38_decode import (
    Qwen38Decoder,
    Qwen38State,
    Qwen38Weights,
)
from sglang.kernels.decode_mk.qwen38_layer import cos_sin_table
from sglang.kernels.decode_mk.qwen38_mtp import MtpWeights
from sglang.kernels.decode_mk.qwen38_prefill import CHUNK
from sglang.kernels.decode_mk.qwen38_prefill import MAX_TOKENS as PREFILL_TOKENS
from sglang.kernels.decode_mk.qwen38_prefill import Qwen38Prefiller
from sglang.kernels.decode_mk.qwen38_spec import MAX_DRAFTS, Qwen38Generator
from sglang.kernels.decode_mk.qwen38_verify import MAX_TOKENS as VERIFY_TOKENS
from sglang.kernels.decode_mk.qwen38_verify import Qwen38Verifier, SlottedState
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
# A prompt past one prefill launch's tokens, which runs as two launches.
LONG_PROMPT = PREFILL_TOKENS + 2 * CHUNK + 9
LONG_POSITIONS = LONG_PROMPT + REPLY + VERIFY_TOKENS

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


def _prefilled_reply(weights, prompt: list[int], positions: int) -> list[int]:
    """decode-mk's greedy reply on the prefill kernel: Qwen38Prefiller.run
    over the whole prompt, then one decode step a token."""
    decoder = Qwen38Decoder(
        weights, Qwen38State.zeros(LAYERS, positions), cos_sin_table(positions)
    )
    reply = [int(Qwen38Prefiller(decoder).run(prompt).argmax())]
    while len(reply) < REPLY:
        pos = len(prompt) + len(reply) - 1
        reply.append(int(decoder.step(reply[-1], pos).argmax()))
    return reply


def _verified_reply(weights, prompt: list[int], prefill: bool) -> list[int]:
    """decode-mk's greedy reply from the verify step fed one token a step,
    after the prompt on slot 0's states: on the prefill kernel, or one decode
    step a token. Speculative decoding's text is this one at every draft
    count, a token's logits being the same bits whatever drafts follow it."""
    slotted = SlottedState.of(
        Qwen38State.zeros(LAYERS, MAX_POSITIONS), VERIFY_TOKENS + 1
    )
    cos_sin = cos_sin_table(MAX_POSITIONS)
    verifier = Qwen38Verifier(weights, slotted, cos_sin)
    state = Qwen38State(
        [slotted.linear(j, 0) for j in range(len(slotted.conv))], slotted.caches
    )
    decoder = Qwen38Decoder(weights, state, cos_sin)
    if prefill:
        logits = Qwen38Prefiller(decoder).run(prompt)
    else:
        for pos, token in enumerate(prompt):
            logits = decoder.step(token, pos)
    reply = [int(logits.argmax())]
    while len(reply) < REPLY:
        i = len(reply) - 1
        logits = verifier.step([reply[-1]], len(prompt) + i, i % slotted.slots)
        reply.append(int(logits[0].argmax()))
    return reply


@pytest.fixture(scope="module")
def prompt(weights):
    return _prompt(weights.vocab, 2 * CHUNK + 9, seed=1)


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


@pytest.mark.parametrize("length", [2 * CHUNK + 9, LONG_PROMPT])
def test_prefilled_reply_is_decode_mks(weights, length):
    """The prefill kernel runs the whole prompt in the launches decode-mk's
    own prefill makes of it, so the states, and the reply, are its bits."""
    prompt = _prompt(weights.vocab, length, seed=4)
    runner = DecodeMkRunner(weights, LONG_POSITIONS, prefill=True)
    assert _plain(runner, prompt) == _prefilled_reply(weights, prompt, LONG_POSITIONS)


def test_chunked_prompt_is_the_whole_prompt(plain, prompt):
    """SGLang's chunked prefill hands the prompt over in pieces, each from
    the position the last one ended at; one decode step a token, that is the
    same steps. The prefill kernel takes the whole prompt, which
    decode_mk_hook.py makes SGLang hand over."""
    whole = _plain(plain, prompt)
    assert _plain(plain, prompt, chunks=[CHUNK, 2 * CHUNK]) == whole


@pytest.mark.parametrize("runner", ["plain", "prefilled"])
def test_a_request_starts_afresh(weights, runner, prompt, request):
    runner = request.getfixturevalue(runner)
    first = _plain(runner, prompt)
    _plain(runner, _prompt(weights.vocab, 40, seed=2))
    assert _plain(runner, prompt) == first


@pytest.mark.parametrize("drafts", range(1, MAX_DRAFTS + 1))
@pytest.mark.parametrize("prefill", [False, True])
def test_speculation_is_decode_mks(weights, mtp, prompt, drafts, prefill):
    """Each cycle emits the accepted drafts and the verify step's own next
    token, the text decode-mk's verify step emits fed one token a step."""
    expected = _verified_reply(weights, prompt, prefill)
    runner = DecodeMkRunner(weights, MAX_POSITIONS, prefill, mtp, drafts)
    assert _speculative(runner, prompt) == expected
    if not prefill:
        assert _speculative(runner, prompt, chunks=[CHUNK]) == expected


def test_speculation_after_another_request(weights, mtp, prompt):
    expected = _verified_reply(weights, prompt, prefill=False)
    runner = DecodeMkRunner(weights, MAX_POSITIONS, False, mtp, MAX_DRAFTS)
    _speculative(runner, _prompt(weights.vocab, 40, seed=2))
    assert _speculative(runner, prompt) == expected


def test_extend_refuses_a_prompt_past_the_caches(weights, plain):
    with pytest.raises(ValueError, match="past the kernels'"):
        plain.extend(_ids(_prompt(weights.vocab, 8, seed=3)), MAX_POSITIONS - 4)
