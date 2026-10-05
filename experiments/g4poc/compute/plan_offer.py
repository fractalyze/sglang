"""The load a poisson session plan offers, read off the plan alone (no server).

`gate sweep` seeds each concurrency's plan separately (seed sweep-<C>), so the arrival draw, not only C, sets the
offered load: pthink30 at C80 offers ~66 live sessions, about what C64 offers, and C96 ~100. This replays a plan's
turn schedule as `loadgen.session` sends it, with a constant E2E, and reports the mean live sessions over the timed
window and the turns finished per second, the numbers a measured point reports as `sessions_active_mean` and
`requests_per_s`. Run from experiments/g4poc on the host that has the session file:

  python compute/plan_offer.py --load pthink30 --concurrency 64,72,80,96 [--e2e 6]
"""

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import msgspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gate import config, loadgen, metrics  # noqa: E402
from workload.schema import Session  # noqa: E402


def replay_times(plan: Sequence[loadgen.SessionStart], sessions: Sequence[Session], think_scale: float,
                 t_end: float, e2e_s: float) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Session spans (first send, last reply) and every turn's reply time, each turn taking e2e_s."""
    spans, done = [], []
    for st in plan:
        s = sessions[st.session_index]
        due, first, last = st.t_start, None, None
        for k in range(st.first_turn, len(s.turns)):
            if k > st.first_turn:
                due = last + s.turns[k].think_s * think_scale
            if due >= t_end:
                break
            first = due if first is None else first
            last = due + e2e_s
            done.append(last)
        if first is not None:
            spans.append((first, last))
    return spans, done


def offered(sessions: Sequence[Session], load: config.SessionLoad, seed: str, e2e_s: float) -> Dict:
    plan = loadgen.plan_open(sessions, load, seed)
    w0, w1 = load.warmup_s, load.warmup_s + load.window_s
    spans, done = replay_times(plan, sessions, load.think_scale, w1, e2e_s)
    return {"concurrency": load.concurrency, "plan_digest": loadgen.plan_digest(plan),
            "live_sessions": metrics.occupancy_in_window(spans, w0, w1)["mean"],
            "turns_per_s": sum(1 for t in done if w0 <= t < w1) / load.window_s,
            "arrivals_in_window": sum(1 for st in plan if w0 <= st.t_start < w1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--load", default="pthink30")
    p.add_argument("--concurrency", required=True)
    p.add_argument("--e2e", type=float, default=6.0, help="mean E2E (s) every turn is given")
    p.add_argument("--seed-prefix", default="sweep-", help="gate sweep seeds concurrency C with sweep-<C>")
    p.add_argument("--sessions", default=config.SESSIONS_PATH)
    args = p.parse_args()
    with open(args.sessions, "rb") as f:
        sessions = [msgspec.json.decode(line, type=Session) for line in f]
    base = config.LOADS[args.load]
    for c in (int(x) for x in args.concurrency.split(",")):
        load = msgspec.structs.replace(base, concurrency=c, name=f"{args.load}-C{c}")
        o = offered(sessions, load, f"{args.seed_prefix}{c}", args.e2e)
        print(f"C{c:<4d} plan {o['plan_digest'][:12]}  live {o['live_sessions']:6.1f}  "
              f"turns/s {o['turns_per_s']:.3f}  arrivals in window {o['arrivals_in_window']}")


if __name__ == "__main__":
    main()
