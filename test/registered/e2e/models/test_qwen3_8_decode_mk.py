# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Qwen3.8-27B served on decode-mk's megakernels behind the SGLANG_DECODE_MK_*
switches, on the int4 checkpoint cyankiwi/Qwen3.8-27B-AWQ-INT4 (6e134ba) named
by SGLANG_DECODE_MK_QWEN38_CHECKPOINT, on one RTX 5090.

Each switch set's greedy replies, token for token, against decode-mk's own
API on the same weights: Qwen38Generator for the prompt fed token by token,
the prefill kernel and decode steps for SGLANG_DECODE_MK_PREFILL=1, and the
same with SGLANG_DECODE_MK_MTP=1, whose speculation must accept drafts and
change no token. Also: a sampled request under MTP, an image refused, and a
server that would batch refused at startup."""

import gc
import os
import subprocess
import sys
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(
    est_time=900,
    stage="base-b",
    runner_config="1-gpu-large",
    disabled="needs an RTX 5090 and the Qwen3.8-27B int4 checkpoint",
)

CHECKPOINT = os.environ.get("SGLANG_DECODE_MK_QWEN38_CHECKPOINT")
CONTEXT = 4096
REPLY = 64
PROMPTS = [
    "Explain in detail how a lighthouse keeps ships safe at night, step by step.",
    "Summarize these readings, then name the warmest site:\n"
    + "\n".join(
        f"Reading {i}: site {'ABCDE'[i % 5]} read {10 + (7 * i) % 23} C at "
        f"{i % 24:02d}:{(13 * i) % 60:02d}."
        for i in range(30)
    ),
]
SERVER_ARGS = [
    "--max-running-requests",
    "1",
    "--disable-radix-cache",
    "--disable-cuda-graph",
    "--context-length",
    str(CONTEXT),
]
MTP_ARGS = ["--speculative-algorithm", "DECODE_MK_MTP", "--disable-overlap-schedule"]


def _prompt_ids() -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        for prompt in PROMPTS
    ]
    return [tokenizer.encode(text, add_special_tokens=False) for text in texts]


def _references(prompts: list[list[int]]) -> dict[str, list[list[int]]]:
    """decode-mk's greedy replies, on the GPU the servers take after: the
    prompt token by token (as verify steps, which decode-mk holds bitwise
    equal), and on the prefill kernel."""
    from sglang.kernels.decode_mk import qwen38_checkpoint
    from sglang.kernels.decode_mk.qwen38_decode import (
        LAYERS,
        Qwen38Decoder,
        Qwen38State,
    )
    from sglang.kernels.decode_mk.qwen38_layer import cos_sin_table
    from sglang.kernels.decode_mk.qwen38_prefill import Qwen38Prefiller
    from sglang.kernels.decode_mk.qwen38_spec import Qwen38Generator

    weights = qwen38_checkpoint.load(CHECKPOINT)
    generator = Qwen38Generator(weights, CONTEXT)
    tokens = [generator.generate(p, REPLY).tokens for p in prompts]
    del generator
    prefilled = []
    for prompt in prompts:
        decoder = Qwen38Decoder(
            weights, Qwen38State.zeros(LAYERS, CONTEXT), cos_sin_table(CONTEXT)
        )
        reply = [int(Qwen38Prefiller(decoder).run(prompt).argmax())]
        while len(reply) < REPLY:
            pos = len(prompt) + len(reply) - 1
            reply.append(int(decoder.step(reply[-1], pos).argmax()))
        prefilled.append(reply)
        del decoder
    del weights
    gc.collect()
    torch.cuda.empty_cache()
    return {"tokens": tokens, "prefill": prefilled}


@unittest.skipUnless(
    CHECKPOINT
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0),
    "set SGLANG_DECODE_MK_QWEN38_CHECKPOINT on an RTX 5090",
)
class TestQwen38DecodeMk(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = _prompt_ids()
        cls.references = _references(cls.prompts)

    def _serve(self, switches: dict[str, str], extra_args=()):
        env = {**os.environ, **switches}
        process = popen_launch_server(
            CHECKPOINT,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[*SERVER_ARGS, *extra_args],
            env=env,
        )
        self.addCleanup(kill_process_tree, process.pid)

    def _generate(self, input_ids, **sampling):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": REPLY,
                    "ignore_eos": True,
                    "temperature": 0,
                    **sampling,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def _assert_replies(self, reference: str):
        replies = [self._generate(p) for p in self.prompts]
        for reply, expected in zip(replies, self.references[reference]):
            self.assertEqual(reply["output_ids"], expected)
        return replies

    def test_decode(self):
        self._serve({"SGLANG_DECODE_MK_DECODE": "1"})
        self._assert_replies("tokens")
        image = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json={
                "model": CHECKPOINT,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            },
                            {"type": "text", "text": "Describe the image."},
                        ],
                    }
                ],
                "max_tokens": 8,
            },
            timeout=60,
        )
        self.assertEqual(image.status_code, 400)

    def test_decode_with_prefill(self):
        self._serve({"SGLANG_DECODE_MK_DECODE": "1", "SGLANG_DECODE_MK_PREFILL": "1"})
        self._assert_replies("prefill")

    def test_mtp(self):
        self._serve(
            {"SGLANG_DECODE_MK_DECODE": "1", "SGLANG_DECODE_MK_MTP": "1"}, MTP_ARGS
        )
        replies = self._assert_replies("tokens")
        for reply in replies:
            self.assertGreater(reply["meta_info"]["spec_accept_length"], 1.2)
        sampled = self._generate(self.prompts[0], temperature=1.0, top_p=0.95)
        self.assertEqual(len(sampled["output_ids"]), REPLY)

    def test_mtp_with_prefill(self):
        self._serve(
            {
                "SGLANG_DECODE_MK_DECODE": "1",
                "SGLANG_DECODE_MK_PREFILL": "1",
                "SGLANG_DECODE_MK_MTP": "1",
            },
            [*MTP_ARGS, "--speculative-num-steps", "3"],
        )
        self._assert_replies("prefill")

    def test_batching_is_refused(self):
        launch = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                CHECKPOINT,
                *SERVER_ARGS,
                "--max-running-requests",
                "2",
            ],
            env={**os.environ, "SGLANG_DECODE_MK_DECODE": "1"},
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertNotEqual(launch.returncode, 0)
        self.assertIn("--max-running-requests 1", launch.stderr)


if __name__ == "__main__":
    unittest.main()
