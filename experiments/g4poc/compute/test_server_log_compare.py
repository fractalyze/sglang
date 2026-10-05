import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server_log_compare as slc  # noqa: E402


def _dec(ts: str, n: int, tput: float) -> str:
    return (f"[2026-10-06 {ts}] Decode batch, #running-req: {n}, #full token: 9, full token usage: 0.00, "
            f"cuda graph: True, gen throughput (token/s): {tput}, #queue-req: 0")


def _pre(ts: str, new: int) -> str:
    return (f"[2026-10-06 {ts}] Prefill batch, #new-seq: 1, #new-token: {new}, #cached-token: 0, "
            f"full token usage: 0.00, #running-req: 0, #queue-req: 0")


class ServerLogCompareTest(absltest.TestCase):
    def test_counts_only_the_timed_window(self):
        lines = [_pre("03:00:00", 999),  # t0, inside the warm-up
                 _pre("03:04:00", 2048), _pre("03:05:00", 1000),  # 240 s and 300 s: timed
                 _pre("03:12:00", 999)]  # 720 s: after the window
        lines += [_dec("03:06:00", 8, 640.0)] * slc.MIN_STEPS + [_dec("03:06:00", 4, 400.0)]
        s = slc.point_stats("\n".join(lines))
        self.assertEqual(s["prefill_passes"], 2)
        self.assertEqual(s["new_tokens"], 3048)
        self.assertEqual(s["decode_steps_per_s"], {8: 80.0})  # batch 4 has too few lines

    def test_points_take_the_last_flush_segments_in_order(self):
        log = "\n".join(["start", slc.FLUSH, _pre("01:00:00", 1), slc.FLUSH, _pre("02:00:00", 1),
                         _pre("02:05:00", 7), slc.FLUSH, _pre("03:00:00", 1), _pre("03:04:30", 5)])
        res = slc.points(log, ["C48", "C64"])
        self.assertEqual(res["C48"]["new_tokens"], 7)
        self.assertEqual(res["C64"]["new_tokens"], 5)


if __name__ == "__main__":
    absltest.main()
