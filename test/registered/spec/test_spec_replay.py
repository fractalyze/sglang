"""
Tests the speculative replay table (SGLANG_SIMULATE_ACC_REPLAY_PATH): a replaying
request emits exactly its continuation whatever the target and drafter predict,
in rounds of the fixed accept length, and a request not in the file is untouched.
"""

import json
import os
import tempfile
import unittest

import torch

from sglang.srt.speculative.spec_replay import SpecReplay
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")

_N = 6  # verify window: last bonus + 5 drafts
_PROMPT = [11, 12, 13, 14]
_CONT = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
_OTHER = [21, 22, 23]


def _table(accept_len: int = 3, num_rows: int = 4) -> SpecReplay:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "replay.json")
        with open(path, "w") as f:
            json.dump({"continuations": [{"input_ids": _PROMPT, "output_ids": _CONT}]}, f)
        return SpecReplay.from_file(path, accept_len, _N, num_rows, "cpu")


def _real_accept(bs: int, num_correct: int):
    """What the target's own sampling returns: junk predictions, `num_correct` drafts plus the bonus."""
    predict = torch.full((bs * _N,), 7, dtype=torch.int32)
    accept_index = torch.full((bs, _N), -1, dtype=torch.int32)
    for i in range(bs):
        accept_index[i, : num_correct + 1] = i * _N + torch.arange(num_correct + 1, dtype=torch.int32)
    accept_lens = torch.full((bs,), num_correct + 1, dtype=torch.int32)
    return predict, accept_lens, accept_index


class TestSpecReplay(unittest.TestCase):
    def test_first_token_replays_only_registered_prompts(self):
        replay = _table()
        replay.register(rows=[2, 0], prompts=[_PROMPT, _OTHER])
        rows = torch.tensor([2, 0])
        seq_lens = torch.tensor([len(_PROMPT), len(_OTHER)])
        out = replay.first_tokens(torch.tensor([5, 6], dtype=torch.int64), rows, seq_lens)
        self.assertEqual(out.tolist(), [_CONT[0], 6])

    def test_drafts_keep_node_zero_and_replace_the_rest(self):
        replay = _table()
        replay.register(rows=[1, 3], prompts=[_PROMPT, _OTHER])
        drafts = torch.arange(2 * _N, dtype=torch.int64) + 500
        s = len(_PROMPT) + 2  # two continuation tokens already committed
        out = replay.drafts(drafts, torch.tensor([1, 3]), torch.tensor([s, 9])).view(2, _N)
        self.assertEqual(out[0, 0].item(), 500)
        self.assertEqual(out[0, 1:].tolist(), _CONT[3 : 3 + _N - 1])
        self.assertEqual(out[1].tolist(), list(range(506, 512)))

    def test_accept_takes_fixed_length_from_continuation(self):
        replay = _table(accept_len=4)
        replay.register(rows=[0, 1], prompts=[_PROMPT, _OTHER])
        predict, accept_lens, accept_index = _real_accept(bs=2, num_correct=1)
        s = len(_PROMPT)
        predict, accept_lens, accept_index = replay.accept(
            predict, accept_lens, accept_index, torch.tensor([0, 1]), torch.tensor([s, 3])
        )
        self.assertEqual(accept_lens.tolist(), [4, 2])
        self.assertEqual(accept_index[0].tolist(), [0, 1, 2, 3, -1, -1])
        self.assertEqual(accept_index[1].tolist(), [6, 7, -1, -1, -1, -1])
        self.assertEqual(predict[accept_index[0, :4].long()].tolist(), _CONT[1:5])
        self.assertEqual(predict[accept_index[1, :2].long()].tolist(), [7, 7])

    def test_positions_past_the_continuation_repeat_its_last_token(self):
        replay = _table()
        replay.register(rows=[0], prompts=[_PROMPT])
        s = len(_PROMPT) + len(_CONT) - 2
        out = replay.drafts(torch.zeros(_N, dtype=torch.int64), torch.tensor([0]), torch.tensor([s]))
        self.assertEqual(out[1:].tolist(), [_CONT[-1]] * (_N - 1))

    def test_rounds_emit_the_continuation_whatever_the_target_predicts(self):
        for accept_len in (1, 3, _N):
            replay = _table(accept_len=accept_len)
            replay.register(rows=[2], prompts=[_PROMPT])
            rows = torch.tensor([2])
            seq_len = len(_PROMPT)
            emitted = replay.first_tokens(torch.tensor([9]), rows, torch.tensor([seq_len])).tolist()
            rounds = 0
            while len(emitted) < len(_CONT):
                predict, accept_lens, accept_index = _real_accept(bs=1, num_correct=rounds % _N)
                predict, accept_lens, accept_index = replay.accept(
                    predict, accept_lens, accept_index, rows, torch.tensor([seq_len])
                )
                emitted += predict[accept_index[0, : accept_lens[0]].long()].tolist()
                seq_len += accept_lens[0].item()
                rounds += 1
            self.assertEqual(emitted[: len(_CONT)], _CONT)
            self.assertEqual(rounds, -(-(len(_CONT) - 1) // accept_len))

    def test_reregistering_a_row_for_an_unknown_prompt_stops_its_replay(self):
        replay = _table()
        replay.register(rows=[1], prompts=[_PROMPT])
        replay.register(rows=[1], prompts=[_OTHER])
        out = replay.first_tokens(torch.tensor([42]), torch.tensor([1]), torch.tensor([3]))
        self.assertEqual(out.tolist(), [42])


if __name__ == "__main__":
    unittest.main()
