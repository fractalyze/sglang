"""Multi-turn greedy exactness of a HiCache ref against a device-only control.

Each session starts from a role-play guard item (persona + ~5K-token history); every later turn
appends the previous turn's output ids and a fixed user follow-up. Turns run in turn-major order at
concurrency 1, so batch composition never differs between runs. On a device pool smaller than all
sessions together (e.g. `--max-total-tokens 16384`), a session's prefix is evicted to the host pool
between its turns and the next turn loads it back; on the control it stays a device hit. Both then
prefill only the new suffix on the same cached prefix, so a correct load-back gives identical
greedy tokens and identical `cached_tokens`.

`--template-cut N` drops the last N tokens of the previous prompt before appending its output, the
way a chat template re-renders a past assistant turn without the generation prompt's tail (Gemma-4:
`<|turn>model\n<reply>` vs the prompt's `<|turn>model\n<|channel>thought\n<channel|>`, N = 4). The
next turn then matches N tokens short of the cached prompt, the case SGLANG_OPT_SWA_PREFILL_WINDOW_MARGIN
exists for.

  python -m hicache.exactness_mt run --ref mem-hc-fix-smallpool --sessions 4 --turns 3
  python -m hicache.exactness_mt compare <control exactness_mt.json> <candidate exactness_mt.json>
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import requests

from gate import config, fidelity, hostwatch, rpquality, runner, server
from workload import schema

FOLLOW_UPS = (
    "Stay in character and tell me what you notice about the room around us.",
    "Before we go on, remind me what you promised me earlier, in your own words.",
    "Describe what happens next, and how it makes you feel.",
)


def generate(url: str, input_ids: List[int], output_len: int) -> Dict:
    r = requests.post(
        f"{url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": output_len,
                "ignore_eos": True,
                "temperature": 0,
            },
        },
        timeout=3600,
    )
    r.raise_for_status()
    out = r.json()
    return {
        "output_ids": out["output_ids"],
        "cached_tokens": out["meta_info"].get("cached_tokens"),
    }


def run_sessions(
    url: str,
    sessions: List[Dict],
    turns: int,
    output_len: int,
    tokenizer,
    template_cut: int = 0,
) -> List[Dict]:
    follow_ups = [
        tokenizer.encode(f"\n\nUser: {m}\n\nAssistant:", add_special_tokens=False)
        for m in FOLLOW_UPS
    ]
    prompts = {s["seed"]: list(s["input_ids"]) for s in sessions}
    rows = []
    for turn in range(turns):
        for s in sessions:
            seed = s["seed"]
            out = generate(url, prompts[seed], output_len)
            rows.append(
                {"seed": seed, "turn": turn, "prompt_len": len(prompts[seed]), **out}
            )
            print(
                f"turn {turn} session {seed}: prompt {len(prompts[seed])} cached {out['cached_tokens']}",
                file=sys.stderr,
                flush=True,
            )
            prompts[seed] = (
                prompts[seed][: len(prompts[seed]) - template_cut]
                + out["output_ids"]
                + follow_ups[turn % len(follow_ups)]
            )
    return rows


def compare(control: Dict, candidate: Dict) -> Dict:
    rows = []
    for a, b in zip(control["rows"], candidate["rows"]):
        assert (a["seed"], a["turn"]) == (b["seed"], b["turn"]), (a["seed"], b["seed"])
        x, y = a["output_ids"], b["output_ids"]
        first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
        if first is None and len(x) != len(y):
            first = min(len(x), len(y))
        rows.append(
            {
                "seed": a["seed"],
                "turn": a["turn"],
                "match": first is None,
                "first_mismatch": first,
                "cached_control": a["cached_tokens"],
                "cached_candidate": b["cached_tokens"],
            }
        )
    return {
        "n": len(rows),
        "exact": sum(r["match"] for r in rows),
        "same_cached_tokens": sum(
            r["cached_control"] == r["cached_candidate"] for r in rows
        ),
        "rows": rows,
    }


def _run(a) -> None:
    ref = server.load_ref(a.ref)
    out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"exactmt-{a.ref}"))
    os.makedirs(out_dir)
    tokenizer = runner.load_tokenizer()
    items = rpquality.build_items(schema.read(config.SESSIONS_PATH), tokenizer)[
        : a.sessions
    ]
    sessions = [{"seed": it["id"], "input_ids": it["input_ids"]} for it in items]
    res: Dict = {
        "ref": ref,
        "turns": a.turns,
        "output_len": a.output_len,
        "template_cut": a.template_cut,
    }
    with (
        hostwatch.host_lock(),
        server.Server(
            ref,
            os.path.join(out_dir, "server.log"),
            extra_args=runner.server_extra_args(ref),
        ) as srv,
    ):
        res["commit"] = srv.commit
        requests.post(f"{srv.url}/flush_cache", timeout=60)
        res["rows"] = run_sessions(
            srv.url, sessions, a.turns, a.output_len, tokenizer, a.template_cut
        )
    res["host"] = srv.host_summary
    fidelity.save_json(os.path.join(out_dir, "exactness_mt.json"), res)
    print(f"run dir: {out_dir}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ref", required=True)
    r.add_argument("--sessions", type=int, default=4)
    r.add_argument("--turns", type=int, default=3)
    r.add_argument("--output-len", type=int, default=128)
    r.add_argument("--template-cut", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("control")
    c.add_argument("candidate")
    a = ap.parse_args()
    if a.cmd == "run":
        _run(a)
    else:
        with open(a.control) as f, open(a.candidate) as g:
            print(json.dumps(compare(json.load(f), json.load(g)), indent=1))


if __name__ == "__main__":
    main()
