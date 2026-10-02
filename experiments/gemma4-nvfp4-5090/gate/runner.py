"""`gate run`: ABBA legs, integrity checks, fidelity, verdict, report and ledger line."""

import asyncio
import json
import logging
import os
import platform
import secrets
import subprocess
import time
from typing import Dict, List, Optional

from gate import client, config, fidelity, gpu, hostwatch, prompts, server, stats

log = logging.getLogger(__name__)

# server_info keys that legitimately differ between two launches of one config.
_VOLATILE_SERVER_KEYS = {"random_seed", "version", "internal_states", "max_total_num_tokens", "pid"}


def new_exp_id(label: str) -> str:
    return f"{label}-{time.strftime('%Y%m%d-%H%M%S')}-{platform.node()}-{secrets.token_hex(3)}"


def environment() -> Dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            return repr(e)

    versions = run([config.VENV_PYTHON, "-c", (
        "import torch, sgl_kernel, flashinfer, transformers, triton;"
        "print(torch.__version__, torch.version.cuda, sgl_kernel.__version__, flashinfer.__version__,"
        " transformers.__version__, triton.__version__)")]).split()
    keys = ("torch", "cuda", "sgl_kernel", "flashinfer", "transformers", "triton")
    return {
        "host": platform.node(),
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,vbios_version,power.limit", "--format=csv,noheader"]),
        "versions": dict(zip(keys, versions)),
        "model_id": config.MODEL_ID,
        "model_revision": config.MODEL_REVISION,
        "model_dir": config.MODEL_DIR,
        "harness": server.harness_commit(),
    }


def _timed_workloads(srv: server.Server, corpus, bos: int, seed: str) -> Dict:
    out = {}
    for wl in config.WORKLOADS:
        reps = []
        for rep in range(wl.reps_per_leg):
            srv.flush_cache()
            ps = prompts.timing_prompts(corpus, bos, f"{seed}/{wl.name}/{rep}", wl.concurrency, wl.prompt_tokens)
            reps.append(asyncio.run(client.run_batch(srv.url, ps, wl.decode_tokens)))
        out[wl.name] = reps
    return out


def warm_up(srv: server.Server, corpus, bos: int, seed: str) -> None:
    """Runs every timed shape once so JIT, autotune and allocator growth stay outside the window."""
    for wl in config.WORKLOADS:
        ps = prompts.timing_prompts(corpus, bos, f"warmup/{seed}/{wl.name}", wl.concurrency, wl.prompt_tokens)
        asyncio.run(client.run_batch(srv.url, ps, wl.decode_tokens))
    srv.flush_cache()


def run_leg(ref: Dict, pair_seed: str, leg_dir: str, corpus, bos: int, with_fidelity: bool) -> Dict:
    os.makedirs(leg_dir, exist_ok=True)
    events: List[Dict] = []
    leg = {"ref": ref["name"], "events": events, "started": time.time()}
    leg["gpu_before_launch"] = gpu.wait_quiet(None, events)
    with server.Server(ref, os.path.join(leg_dir, "server.log")) as srv:
        leg["commit"] = srv.commit
        leg["server_cmd"] = srv.command()
        leg["weights_at_load"] = srv.weight_checksum()
        leg["server_info"] = srv.server_info()
        warm_up(srv, corpus, bos, pair_seed)
        leg["gpu_before_window"] = gpu.wait_quiet(srv.proc.pid, events)
        offset = srv.log_offset()
        with gpu.Telemetry(srv.proc.pid) as tel:
            t0 = time.time()
            leg["workloads"] = _timed_workloads(srv, corpus, bos, pair_seed)
            leg["window_s"] = time.time() - t0
        leg["telemetry"] = tel.summary()
        leg["decode_steps_in_window"] = srv.decode_steps_since(offset)
        leg["backends"] = srv.backend_report()
        if with_fidelity:
            leg["fidelity_outputs"] = fidelity.run(srv.url)
        leg["weights_at_end"] = srv.weight_checksum()
        leg["host_preflight"] = srv.preflight
    leg["host"] = srv.host_summary
    leg["integrity"] = leg_integrity(leg)
    return leg


def leg_integrity(leg: Dict) -> Dict:
    steps = leg["decode_steps_in_window"]
    cached = sum(s["cached_tokens"] for reps in leg["workloads"].values() for r in reps for s in r["streams"])
    prompt = sum(s["prompt_tokens"] for reps in leg["workloads"].values() for r in reps for s in r["streams"])
    w_load, w_end = leg["weights_at_load"], leg["weights_at_end"]
    checks = {
        "no_eager_decode_in_window": steps["eager_steps"] == 0 and steps["graph_steps"] > 0,
        "no_foreign_gpu_process_in_window": not leg["telemetry"]["foreign_seen"],
        "no_thermal_throttle_in_window": not leg["telemetry"]["thermal_or_hw_throttle_samples"],
        "weights_unchanged_during_leg": (not w_load["ok"]) or w_load == w_end,
        "host_watchdog_not_tripped": leg["host"]["tripped"] is None,
    }
    return {"checks": checks, "ok": all(checks.values()), "prefix_cache_hit_rate": cached / max(prompt, 1)}


def abba_order(n_pairs: int) -> List[List[str]]:
    return [["control", "candidate"] if k % 2 == 0 else ["candidate", "control"] for k in range(n_pairs)]


def timed_output_agreement(control: Dict, candidate: Dict) -> Dict:
    """Token agreement of the outputs produced inside the two timed windows of a pair (same prompts)."""
    rates = []
    for wl in control["workloads"]:
        for rc, rk in zip(control["workloads"][wl], candidate["workloads"][wl]):
            for sc, sk in zip(rc["streams"], rk["streams"]):
                rates.append(fidelity.token_match_rate(sc["output_ids"] or [], sk["output_ids"] or []))
    return {"min": min(rates), "mean": sum(rates) / len(rates), "n_streams": len(rates)}


def server_arg_diff(control: Dict, candidate: Dict, declared: List[str]) -> Dict:
    a, b = control["server_info"], candidate["server_info"]
    diff = sorted(k for k in set(a) | set(b) if k not in _VOLATILE_SERVER_KEYS and a.get(k) != b.get(k))
    declared_keys = {d.lstrip("-").replace("-", "_") for d in declared if d.startswith("--")}
    undeclared = [k for k in diff if k not in declared_keys]
    return {"differing_keys": diff, "undeclared": undeclared}


def run_gate(control_name: str, candidate_name: str, n_pairs: int, label: str, notes: str = "") -> Dict:
    control_ref, candidate_ref = server.load_ref(control_name), server.load_ref(candidate_name)
    exp_id = new_exp_id(label)
    run_dir = os.path.join(config.RUNS_DIR, exp_id)
    os.makedirs(run_dir)
    tok = prompts.load_tokenizer()
    corpus = prompts.load_corpus(tok)
    nonce = secrets.token_hex(4)
    meta = {"exp_id": exp_id, "control": control_ref, "candidate": candidate_ref, "n_pairs": n_pairs,
            "notes": notes, "nonce": nonce, "corpus_digest": prompts.corpus_digest(corpus),
            "fidelity_prompts_digest": fidelity.prompts_digest(), "environment": environment(),
            "config": {"workloads": [{"name": w.name, "concurrency": w.concurrency, "prompt": w.prompt_tokens,
                                      "decode": w.decode_tokens, "reps": w.reps_per_leg} for w in config.WORKLOADS]}}
    _write(os.path.join(run_dir, "meta.json"), meta)
    refs = {"control": control_ref, "candidate": candidate_ref}
    legs: Dict[str, List[Dict]] = {"control": [], "candidate": []}
    with hostwatch.host_lock():
        for k, order in enumerate(abba_order(n_pairs)):
            for role in order:
                leg_dir = os.path.join(run_dir, f"pair{k}-{role}")
                leg = run_leg(refs[role], f"{nonce}/pair{k}", leg_dir, corpus, tok.bos_token_id,
                              with_fidelity=(k == 0))
                leg["pair"], leg["role"] = k, role
                _write(os.path.join(leg_dir, "leg.json"), leg)
                legs[role].append(leg)
    report = evaluate(meta, legs)
    _write(os.path.join(run_dir, "report.json"), report)
    append_ledger(meta, report, run_dir)
    return report


def evaluate(meta: Dict, legs: Dict[str, List[Dict]]) -> Dict:
    control, candidate = legs["control"], legs["candidate"]
    summary = stats.summarize_pairs([stats.leg_sums(l) for l in control], [stats.leg_sums(l) for l in candidate])
    noise = _load_noise()
    timing = stats.timing_verdict(summary, noise)
    agreement = [timed_output_agreement(c, k) for c, k in zip(control, candidate)]
    thresholds = fidelity.load_json(fidelity.THRESHOLDS_PATH) if os.path.exists(fidelity.THRESHOLDS_PATH) else None
    fid = {}
    if os.path.exists(fidelity.REFERENCE_PATH):
        reference = fidelity.load_json(fidelity.REFERENCE_PATH)
        for role, ls in legs.items():
            cmp = fidelity.compare(reference, ls[0]["fidelity_outputs"])
            fid[role] = {"compare": cmp, "verdict": fidelity.verdict(cmp, thresholds) if thresholds else None}
    args_diff = server_arg_diff(control[0], candidate[0], meta["candidate"].get("server_args", []) +
                                meta["control"].get("server_args", []))
    w_c, w_k = control[0]["weights_at_load"], candidate[0]["weights_at_load"]
    weights_match = w_c.get("checksum") == w_k.get("checksum") if w_c["ok"] and w_k["ok"] else None
    integrity = {
        "legs_ok": all(l["integrity"]["ok"] for ls in legs.values() for l in ls),
        "per_leg": [{"pair": l["pair"], "role": l["role"], **l["integrity"]} for ls in legs.values() for l in ls],
        "timed_output_agreement_min": min(a["min"] for a in agreement),
        "undeclared_server_arg_diffs": args_diff["undeclared"],
        "weights_match_control": weights_match,
    }
    integrity["ok"] = (
        integrity["legs_ok"]
        and integrity["timed_output_agreement_min"] >= config.TOKEN_MATCH_MIN
        and not args_diff["undeclared"]
        and (weights_match is not False or meta["candidate"]["weight_layout_change"])
    )
    fid_pass = fid.get("candidate", {}).get("verdict", {}) or {}
    return {
        "exp_id": meta["exp_id"],
        "control": meta["control"]["name"],
        "candidate": meta["candidate"]["name"],
        "summary": summary,
        "noise_used": noise,
        "timing": timing,
        "timed_output_agreement": agreement,
        "fidelity": fid,
        "server_arg_diff": args_diff,
        "integrity": integrity,
        "verdict": {
            "integrity_ok": integrity["ok"],
            "fidelity_pass": fid_pass.get("pass"),
            "timing_promote": timing["promote"],
            "promote": bool(integrity["ok"] and fid_pass.get("pass") and timing["promote"]),
        },
    }


NOISE_PATH = os.path.join(config.REFERENCE_DIR, "noise.json")


def _load_noise() -> Dict[str, float]:
    if not os.path.exists(NOISE_PATH):
        return {}
    return fidelity.load_json(NOISE_PATH)["per_pair_log_sigma"]


def save_noise_from(report_path: str) -> Dict:
    report = fidelity.load_json(report_path)
    if report["control"] != report["candidate"]:
        raise ValueError("noise must come from an A/A run (control == candidate)")
    noise = {"from_exp": report["exp_id"], "per_pair_log_sigma": report["summary"]["per_pair_log_sigma"],
             "bars": {m: stats.promotion_bar(s) for m, s in report["summary"]["per_pair_log_sigma"].items()}}
    fidelity.save_json(NOISE_PATH, noise)
    return noise


def append_ledger(meta: Dict, report: Dict, run_dir: str) -> None:
    os.makedirs(os.path.dirname(config.LEDGER), exist_ok=True)
    o = report["summary"]["overall"]
    line = {
        "run": meta["exp_id"],
        "time": time.strftime("%Y-%m-%dT%H:%M"),
        "host": meta["environment"]["host"],
        "status": "completed",
        "validity": "valid" if report["integrity"]["ok"] else "contaminated",
        "control": meta["control"]["name"],
        "candidate": meta["candidate"]["name"],
        "control_commit": meta["control"]["commit"],
        "candidate_commit": meta["candidate"]["commit"],
        "harness_commit": meta["environment"]["harness"]["commit"],
        "n_pairs": meta["n_pairs"],
        "metrics": {k: o[k] for k in ("w8_composite", "w8_prefill_gain", "w8_decode_gain", "w1_tpot_gain",
                                      "w1_tpot_control_ms", "w1_tpot_candidate_ms", "w32_tput_gain",
                                      "w32_tput_control_tok_s", "w32_tput_candidate_tok_s")},
        "verdict": report["verdict"],
        "output": os.path.relpath(os.path.join(run_dir, "report.json"), config.ROOT),
        "note": meta["notes"],
    }
    with open(config.LEDGER, "a") as f:
        f.write(json.dumps(line) + "\n")


def _write(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
