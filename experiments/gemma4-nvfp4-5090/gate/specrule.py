"""`gate spec-run` / `gate spec-reevaluate`: the speculative-decoding rule (stats.spec_verdict).

1. Record: one control lifetime decodes every timed prompt free-running; its
   outputs become the run's replay file.
2. Pairs (ABBA): each leg serves its ref from the ref's tree plus
   gate/spec_replay.patch, with the replay env. It times the gate's workloads,
   in which every stream must emit exactly the recorded text, then measures
   free-running acceptance on the hidden set. Pair 0 also runs the fidelity
   passes. Prompts outside the replay file (warm-up, hidden set, fidelity)
   decode normally.
3. Verdict: round time from the replay legs' paired gains, tau from the hidden
   set, and quality from a quality-compare JSON, which gates only a tau gain.
"""

import asyncio
import hashlib
import json
import os
from typing import Dict, List, Optional

from gate import checkpoint, client, config, fidelity, gpu, hostwatch, prompts, runner, server, stats

OVERLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_replay.patch")
REPLAY_FILE = "replay.json"


def replay_key(input_ids: List[int]) -> str:
    """The key the server's replay table files a prompt under (sglang.srt.speculative.spec_replay.prompt_key)."""
    return hashlib.sha256(",".join(map(str, input_ids)).encode()).hexdigest()


def accept_len(ref: Dict) -> int:
    return config.SPEC_REPLAY_ACCEPT_LEN if stats.is_speculative(ref) else 1


def replay_mode(replay_path: str) -> runner.LegMode:
    return runner.LegMode(
        overlay=OVERLAY,
        env={"SGLANG_SIMULATE_ACC_REPLAY_PATH": replay_path,
             "SGLANG_SIMULATE_ACC_REPLAY_LEN": str(config.SPEC_REPLAY_ACCEPT_LEN)},
        fixed_seeds={"W32": config.SPEC_REPLAY_W32_SEED},
        tau_pass=True,
    )


def record(ref: Dict, run_dir: str, corpus, bos: int) -> str:
    """The control's free-running outputs on every replayed prompt, written as the replay file."""
    rec_dir = os.path.join(run_dir, "record")
    os.makedirs(rec_dir)
    by_wl = runner.timed_prompts(corpus, bos, "record", replay_mode(""))
    events: List[Dict] = []
    gpu.wait_quiet(None, events)
    entries = []
    # The hook tree without the replay env: the hook is inert, so this is the control as gated.
    with server.Server(ref, os.path.join(rec_dir, "server.log"), overlay=OVERLAY) as srv:
        runner.warm_up(srv, corpus, bos, "record")
        for wl in config.WORKLOADS:
            for ps in by_wl[wl.name]:
                srv.flush_cache()
                res = asyncio.run(client.run_batch(srv.url, ps, wl.decode_tokens))
                entries += [{"workload": wl.name, "input_ids": p, "output_ids": s["output_ids"]}
                            for p, s in zip(ps, res["streams"])]
        commit = srv.commit
    path = os.path.join(run_dir, REPLAY_FILE)
    runner._write(path, {"ref": ref["name"], "commit": commit, "events": events, "host": srv.host_summary,
                         "continuations": entries})
    return path


def replay_text_exact(leg: Dict, replay: Dict, by_wl: Dict) -> Dict:
    """Every timed stream emitted exactly the recorded continuation of its prompt."""
    want = {replay_key(e["input_ids"]): e["output_ids"] for e in replay["continuations"]}
    mismatched = []
    for wl, reps in leg["workloads"].items():
        for r, (rep, ps) in enumerate(zip(reps, by_wl[wl])):
            for i, (s, p) in enumerate(zip(rep["streams"], ps)):
                if s["output_ids"] != want.get(replay_key(p)):
                    mismatched.append(f"{wl}/{r}/{i}")
    return {"ok": not mismatched, "mismatched": mismatched[:20], "n_mismatched": len(mismatched)}


def tau_consistency(legs: List[Dict]) -> Dict:
    """Hidden-set rounds per prompt across one arm's legs (deterministic greedy: should not move)."""
    per_prompt = {}
    for leg in legs:
        for row in leg["hidden_tau"]:
            per_prompt.setdefault(row["id"], set()).add(row["verify_ct"])
    moved = sorted(k for k, v in per_prompt.items() if len(v) > 1)
    return {"n_prompts_with_differing_rounds": len(moved), "prompts": moved}


def run(control_name: str, candidate_name: str, n_pairs: int, label: str, notes: str, decide_on: str,
        quality_compare: Optional[str]) -> Dict:
    control_ref, candidate_ref = server.load_ref(control_name), server.load_ref(candidate_name)
    exp_id = runner.new_exp_id(label)
    run_dir = os.path.join(config.RUNS_DIR, exp_id)
    os.makedirs(run_dir)
    tok = prompts.load_tokenizer()
    corpus = prompts.load_corpus(tok)
    nonce = os.urandom(4).hex()
    meta = {"exp_id": exp_id, "rule": stats.SPEC_RULE, "control": control_ref, "candidate": candidate_ref,
            "n_pairs": n_pairs, "notes": notes, "decide_on": decide_on, "nonce": nonce,
            "quality_compare": quality_compare, "overlay": {"patch": OVERLAY, "sha": server.overlay_digest(OVERLAY)},
            "accept_len": {"control": accept_len(control_ref), "candidate": accept_len(candidate_ref)},
            "corpus_digest": prompts.corpus_digest(corpus), "fidelity_prompts_digest": fidelity.prompts_digest(),
            "environment": runner.environment(), "checkpoint": checkpoint.verify(),
            "config": {"workloads": [{"name": w.name, "concurrency": w.concurrency, "prompt": w.prompt_tokens,
                                      "decode": w.decode_tokens, "reps": w.reps_per_leg,
                                      "fixed_prompt_seed": w.fixed_prompt_seed} for w in config.WORKLOADS],
                       "w32_replay_seed": config.SPEC_REPLAY_W32_SEED,
                       "tau": {"max_new": config.TAU_MAX_NEW_TOKENS, "concurrency": config.TAU_CONCURRENCY}}}
    runner._write(os.path.join(run_dir, "meta.json"), meta)
    refs = {"control": control_ref, "candidate": candidate_ref}
    legs: Dict[str, List[Dict]] = {"control": [], "candidate": []}
    with hostwatch.host_lock():
        mode = replay_mode(record(control_ref, run_dir, corpus, tok.bos_token_id))
        for k, order in enumerate(runner.abba_order(n_pairs)):
            for role in order:
                leg_dir = os.path.join(run_dir, f"pair{k}-{role}")
                leg = runner.run_leg_with_retries(refs[role], f"{nonce}/pair{k}", leg_dir, corpus, tok.bos_token_id,
                                                  with_fidelity=(k == 0), mode=mode)
                leg["pair"], leg["role"] = k, role
                runner._write(os.path.join(leg_dir, "leg.json"), leg)
                legs[role].append(leg)
    report = evaluate(meta, legs, run_dir, corpus, tok.bos_token_id)
    runner._write(os.path.join(run_dir, "report.json"), report)
    runner.append_ledger(meta, report, run_dir)
    return report


def _quality_pass(path: Optional[str]) -> Optional[bool]:
    return None if path is None else fidelity.load_json(path)["pass"]


def evaluate(meta: Dict, legs: Dict[str, List[Dict]], run_dir: str, corpus, bos: int) -> Dict:
    control, candidate = legs["control"], legs["candidate"]
    replay = fidelity.load_json(os.path.join(run_dir, REPLAY_FILE))
    by_wl = runner.timed_prompts(corpus, bos, "", replay_mode(""))
    summary = stats.summarize_pairs([stats.leg_sums(l) for l in control], [stats.leg_sums(l) for l in candidate])
    tau = stats.tau_ratio(control[0]["hidden_tau"], candidate[0]["hidden_tau"])
    quality_pass = _quality_pass(meta.get("quality_compare"))
    noise = runner._load_noise_file().get("spec_replay_per_pair_log_sigma", {})
    timing = stats.spec_verdict(summary, tau, quality_pass, noise, meta["accept_len"], meta["decide_on"])
    fid = runner.fidelity_report(legs)
    exact = {f"pair{l['pair']}-{l['role']}": replay_text_exact(l, replay, by_wl) for ls in legs.values() for l in ls}
    args_diff = runner.server_arg_diff(control[0], candidate[0], meta["candidate"].get("server_args", []) +
                                       meta["control"].get("server_args", []))
    w_c, w_k = control[0]["weights_at_load"], candidate[0]["weights_at_load"]
    weights_match = w_c.get("checksum") == w_k.get("checksum") if w_c["ok"] and w_k["ok"] else None
    integrity = {
        "legs_ok": all(l["integrity"]["ok"] for ls in legs.values() for l in ls),
        "per_leg": [{"pair": l["pair"], "role": l["role"], **l["integrity"],
                     "decode_steps": l["decode_steps_in_window"]["graph_steps"]} for ls in legs.values() for l in ls],
        "replay_text_exact": all(e["ok"] for e in exact.values()),
        "replay_text_per_leg": exact,
        "replay_control_matches_ref": replay["ref"] == meta["control"]["name"],
        "timed_streams_full_length": all(runner.timed_streams_full_length(l, meta) for ls in legs.values() for l in ls),
        "tau_consistency": {role: tau_consistency(ls) for role, ls in legs.items()},
        "undeclared_server_arg_diffs": args_diff["undeclared"],
        "weights_match_control": weights_match,
    }
    integrity["ok"] = (
        integrity["legs_ok"]
        and integrity["replay_text_exact"]
        and integrity["replay_control_matches_ref"]
        and integrity["timed_streams_full_length"]
        and not args_diff["undeclared"]
        and (weights_match is not False or meta["candidate"]["weight_layout_change"])
        and meta["checkpoint"]["matches_hf_revision"]
    )
    fid_pass = fid.get("candidate", {}).get("verdict", {}) or {}
    return {
        "exp_id": meta["exp_id"],
        "rule": stats.SPEC_RULE,
        "control": meta["control"]["name"],
        "candidate": meta["candidate"]["name"],
        "summary": summary,
        "noise_used": noise,
        "timing": timing,
        "quality_pass": quality_pass,
        "fidelity": fid,
        "server_arg_diff": args_diff,
        "integrity": integrity,
        "verdict": {
            "integrity_ok": integrity["ok"],
            "fidelity_pass": fid_pass.get("pass"),
            "timing_promote": timing["promote"],
            "timing_rule": stats.SPEC_RULE,
            "promote": bool(integrity["ok"] and fid_pass.get("pass") and timing["promote"]),
        },
    }


def reevaluate(run_dir: str, quality_compare: Optional[str] = None, decide_on: Optional[str] = None) -> Dict:
    """Recomputes a spec-run's report from its saved legs (the old report is kept as report.v<n>.json)."""
    meta = fidelity.load_json(os.path.join(run_dir, "meta.json"))
    if quality_compare is not None:
        meta["quality_compare"] = quality_compare
    if decide_on is not None:
        meta["decide_on"] = decide_on
    legs: Dict[str, List[Dict]] = {"control": [], "candidate": []}
    for k in range(meta["n_pairs"]):
        for role in ("control", "candidate"):
            legs[role].append(fidelity.load_json(os.path.join(run_dir, f"pair{k}-{role}", "leg.json")))
    tok = prompts.load_tokenizer()
    report = evaluate(meta, legs, run_dir, prompts.load_corpus(tok), tok.bos_token_id)
    old = os.path.join(run_dir, "report.json")
    n = 1
    while os.path.exists(os.path.join(run_dir, f"report.v{n}.json")):
        n += 1
    os.rename(old, os.path.join(run_dir, f"report.v{n}.json"))
    report["reevaluated_from"] = f"report.v{n}.json"
    report["reevaluated_by_harness"] = server.harness_commit()
    runner._write(old, report)
    meta["notes"] = f"{meta['notes']} [re-evaluated: {report['reevaluated_from']} superseded]"
    runner.append_ledger(meta, report, run_dir)
    return report


def summary_line(report: Dict) -> Dict:
    t = report["timing"]
    return {"exp_id": report["exp_id"], "round_time": t["round_time"], "tau": {k: t["tau"][k] for k in (
        "control", "candidate", "ratio", "ci95", "significant")}, "counted_tau": t["counted_tau"],
            "decomposed": t["decomposed"], "checks": t["checks"], "verdict": report["verdict"],
            "integrity": {k: v for k, v in report["integrity"].items() if k not in ("per_leg", "replay_text_per_leg")}}


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
