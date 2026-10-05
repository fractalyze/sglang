import os
import sys

import msgspec
from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_offer  # noqa: E402
from gate import config, loadgen  # noqa: E402
from workload.schema import Session, Turn  # noqa: E402


def _session(i: int, thinks) -> Session:
    return Session(session_id=f"s{i}", language="en", system="", history=[],
                   turns=[Turn(user="u", think_s=t) for t in thinks], source="test")


class PlanOfferTest(absltest.TestCase):
    def test_turns_follow_the_previous_reply_and_stop_at_the_window_end(self):
        sessions = [_session(0, [0, 10, 10])]
        start = loadgen.SessionStart(0.0, 0, 0, "n0")
        spans, done = plan_offer.replay_times([start], sessions, 1.0, 100.0, 2.0)
        self.assertEqual(done, [2.0, 14.0, 26.0])  # sends at 0, 12, 24
        self.assertEqual(spans, [(0.0, 26.0)])
        spans, done = plan_offer.replay_times([start], sessions, 1.0, 20.0, 2.0)
        self.assertEqual(done, [2.0, 14.0])  # turn 2 falls due at 24, past the end
        self.assertEqual(spans, [(0.0, 14.0)])

    def test_mid_conversation_start_and_think_scale(self):
        sessions = [_session(0, [0, 10, 10])]
        start = loadgen.SessionStart(5.0, 0, 1, "n0")
        _, done = plan_offer.replay_times([start], sessions, 2.0, 100.0, 1.0)
        self.assertEqual(done, [6.0, 27.0])  # turn 1 sent at its start, turn 2 after 10 x 2 s

    def test_offered_uses_the_sweeps_plan(self):
        sessions = [_session(i, [0] + [20.0] * 4) for i in range(50)]
        load = msgspec.structs.replace(config.PTHINK30, concurrency=8, name="pthink30-C8")
        o = plan_offer.offered(sessions, load, "sweep-8", 3.0)
        self.assertEqual(o["plan_digest"], loadgen.plan_digest(loadgen.plan_open(sessions, load, "sweep-8")))
        self.assertGreater(o["live_sessions"], 0)
        self.assertGreater(o["turns_per_s"], 0)


if __name__ == "__main__":
    absltest.main()
