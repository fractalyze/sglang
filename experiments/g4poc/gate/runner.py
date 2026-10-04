"""`gate run`: ABBA legs at a fixed offered load, integrity, fidelity, verdict, report and ledger.

Differences from the gemma4nv gate, all forced by the workload: the prefix cache
stays on and is never flushed inside the window (multi-turn reuse is part of the
system under test, and its hit rate is reported per leg); nonces are per session,
not per request; each leg replays the same fixed plan of session arrivals rather
than fixed batches; and a decode step outside a CUDA graph is reported, not
refused (the running batch legitimately exceeds the captured sizes at this
concurrency). Host safety, quiescence, weights, server-arg and fidelity checks
are unchanged.
"""

import asyncio
import json
import logging
import os
import platform
import secrets
import subprocess
import time
from typing import Dict, List, Optional, Sequence

import msgspec

from gate import checkpoint, config, fidelity, gpu, hostwatch, loadgen, metrics, server, stats
from workload import schema

log = logging.getLogger(__name__)

_VOLATILE_SERVER_KEYS = {"random_seed", "version", "internal_states", "max_total_num_tokens", "max_req_input_len",
                         "pid", "startup_time"}
_FLAG_DERIVED_PATHS = {
    "chunked_prefill_size": ("cuda_graph_config.prefill.max_bs", "cuda_graph_config.prefill.bs"),
    "mem_fraction_static": ("max_running_requests",),
    "cuda_graph_max_bs_decode": ("cuda_graph_config.decode.max_bs", "cuda_graph_config.decode.bs"),
    "cuda_graph_max_bs": ("cuda_graph_config.decode.max_bs", "cuda_graph_config.decode.bs",
                          "cuda_graph_config.prefill.max_bs", "cuda_graph_config.prefill.bs"),
}
# Every leg serves Prometheus counters (retractions, cached tokens); both arms get it.
METRICS_ARGS = ["--enable-metrics"]
# Max p99 lateness of the client's sends against the plan: past it the client, not the
# server, shaped the load.
MAX_CLIENT_LAG_P99_S = 1.0
WARMUP_LOAD = config.SessionLoad(name="warmup", concurrency=8, warmup_s=0.0, window_s=30.0, expected_session_s=30.0,
                                 think_scale=0.05, drain_timeout_s=300.0)


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


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.MODEL_DIR)


def server_extra_args(ref: Dict) -> List[str]:
    return [a for a in METRICS_ARGS if a not in ref.get("server_args", [])]


def warm_up(srv: server.Server, sessions: Sequence[schema.Session], tok, seed: str) -> None:
    """A short replay so JIT, autotune and allocator growth stay outside the window; then the cache is flushed."""
    asyncio.run(loadgen.replay(srv.url, sessions, WARMUP_LOAD, tok, f"warmup/{seed}", sample_metrics=False))
    srv.flush_cache()


def replay_leg(srv: server.Server, sessions, tok, load: config.SessionLoad, pair_seed: str, nonce_at: str) -> Dict:
    """The timed replay plus everything measured over it; the cache starts empty (flushed after warm-up)."""
    offset = srv.log_offset()
    rep = asyncio.run(loadgen.replay(srv.url, sessions, load, tok, pair_seed, nonce_at=nonce_at,
                                     keep_outputs=load.mode == "scripted"))
    w0, w1 = load.warmup_s, load.warmup_s + load.window_s
    with open(srv.log_path, errors="replace") as f:
        f.seek(offset)
        log_text = f.read()
    delta = metrics.window_counter_delta(rep["metric_samples"], w0, w1)
    return {
        "replay": rep,
        "summary": metrics.summarize(rep["records"], load.warmup_s, load.window_s),
        "gauges": metrics.gauge_summary(rep["metric_samples"], w0, w1),
        "counter_delta": delta,
        "retractions": metrics.retractions(delta, metrics.retractions_from_log(log_text)),
        "decode_steps": srv.decode_steps_since(offset),
    }


def run_leg(ref: Dict, pair_seed: str, leg_dir: str, sessions, tok, load: config.SessionLoad, with_fidelity: bool,
            nonce_at: str = "start") -> Dict:
    os.makedirs(leg_dir, exist_ok=True)
    events: List[Dict] = []
    leg = {"ref": ref["name"], "events": events, "started": time.time(), "load": msgspec.to_builtins(load)}
    leg["gpu_before_launch"] = gpu.wait_quiet(None, events)
    with server.Server(ref, os.path.join(leg_dir, "server.log"), extra_args=server_extra_args(ref)) as srv:
        leg["commit"] = srv.commit
        leg["server_cmd"] = srv.command()
        leg["weights_at_load"] = srv.weight_checksum()
        leg["server_info"] = srv.server_info()
        warm_up(srv, sessions, tok, pair_seed)
        leg["gpu_before_window"] = gpu.wait_quiet(srv.proc.pid, events)
        srv._watchdog.set_phase("timed")
        with gpu.Telemetry(srv.proc.pid) as tel:
            leg.update(replay_leg(srv, sessions, tok, load, pair_seed, nonce_at))
        srv._watchdog.set_phase("post")
        leg["telemetry"] = tel.summary()
        leg["backends"] = srv.backend_report()
        if with_fidelity and os.path.exists(fidelity.PROMPTS_PATH):
            leg["fidelity_outputs"], leg["fidelity_forced"] = fidelity_passes(srv)
        leg["weights_at_end"] = srv.weight_checksum()
        leg["host_preflight"] = srv.preflight
    leg["host"] = srv.host_summary
    leg["integrity"] = leg_integrity(leg)
    return leg


def fidelity_passes(srv: server.Server):
    """Free-running then teacher-forced fidelity, with the radix cache flushed before each."""
    srv.flush_cache()
    outputs = fidelity.run(srv.url)
    srv.flush_cache()
    forced = fidelity.run_forced(srv.url, fidelity.load_json(fidelity.REFERENCE_PATH))
    return outputs, forced


_LEG_RETRIES = 2


def run_leg_with_retries(ref: Dict, pair_seed: str, leg_dir: str, sessions, tok, load, with_fidelity: bool,
                         nonce_at: str = "start") -> Dict:
    """Reruns a leg the host disturbed (watchdog kill, busy-host refusal); earlier attempts are kept."""
    attempts = []
    for attempt in range(_LEG_RETRIES + 1):
        d = leg_dir if attempt == 0 else f"{leg_dir}-retry{attempt}"
        try:
            leg = run_leg(ref, pair_seed, d, sessions, tok, load, with_fidelity, nonce_at)
        except (RuntimeError, hostwatch.HostUnsafe) as e:
            attempts.append({"dir": d, "error": repr(e)[:500]})
            continue
        checks = leg["integrity"]["checks"]
        disturbed = not (checks["host_watchdog_not_tripped"] and checks["host_load_quiet_in_window"])
        if not disturbed or attempt == _LEG_RETRIES:
            leg["attempts"] = attempts
            return leg
        attempts.append({"dir": d, "error": "host disturbed the timed window", "host": leg["host"]})
    raise RuntimeError(f"leg failed {len(attempts)} times: {attempts}")


def leg_integrity(leg: Dict) -> Dict:
    s = leg["summary"]
    w_load, w_end = leg["weights_at_load"], leg["weights_at_end"]
    checks = {
        "no_failed_requests": s["n_failed"] == 0 and leg["replay"]["abandoned"] == 0,
        "offered_load_kept": s["client_lag_p99_s"] <= MAX_CLIENT_LAG_P99_S,
        "no_foreign_gpu_process_in_window": not leg["telemetry"]["foreign_seen"],
        "no_thermal_throttle_in_window": not leg["telemetry"]["thermal_or_hw_throttle_samples"],
        "weights_unchanged_during_leg": (not w_load["ok"]) or w_load == w_end,
        "host_watchdog_not_tripped": leg["host"]["tripped"] is None,
        "host_load_quiet_in_window": leg["host"]["peaks_by_phase"].get("timed", {}).get("peak_load1", 0)
        <= config.MAX_START_LOAD1,
    }
    steps = leg["decode_steps"]
    return {"checks": checks, "ok": all(checks.values()),
            "reported": {"prefix_cache_hit_rate": s["prefix_cache_hit_rate"],
                         "retracted_requests": leg["retractions"]["requests"],
                         "eager_decode_steps": steps["eager_steps"], "graph_decode_steps": steps["graph_steps"]}}


def abba_order(n_pairs: int) -> List[List[str]]:
    return [["control", "candidate"] if k % 2 == 0 else ["candidate", "control"] for k in range(n_pairs)]


def _window_outputs(leg: Dict) -> Dict:
    w0 = leg["load"]["warmup_s"]
    w1 = w0 + leg["load"]["window_s"]
    return {(r["nonce"], r["turn"]): r["output_ids"] for r in leg["replay"]["records"]
            if r["ok"] and w0 <= r["t_send"] < w1}


def timed_output_agreement(control: Dict, candidate: Dict) -> Dict:
    """Token agreement of the replies both arms produced for the same (session, turn) in the window."""
    a, b = _window_outputs(control), _window_outputs(candidate)
    common = sorted(set(a) & set(b))
    rates = [fidelity.token_match_rate(a[k], b[k]) for k in common]
    if not rates:
        return {"min": float("nan"), "mean": float("nan"), "n_streams": 0, "n_identical": 0}
    return {"min": min(rates), "mean": sum(rates) / len(rates), "n_streams": len(rates),
            "n_identical": sum(1 for r in rates if r == 1.0)}


def _flatten(d: Dict, prefix: str = "") -> Dict:
    out = {}
    for k, v in d.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, path + "."))
        else:
            out[path] = v
    return out


def _covered(path: str, prefixes) -> bool:
    return any(path == p or path.startswith(p + ".") for p in prefixes)


def _launch_flags(cmd: Optional[str]) -> List[str]:
    return [t for t in (cmd or "").split() if t.startswith("--")]


def agreement_threshold(aa_pair_means: List[float]) -> float:
    n = len(aa_pair_means)
    mean = sum(aa_pair_means) / n
    sd = (sum((x - mean) ** 2 for x in aa_pair_means) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return min(aa_pair_means) - max(config.NOISE_SIGMAS * sd, config.AGREEMENT_MIN_MARGIN)


def agreement_check(meta: Dict, agreement: List[Dict], noise: Dict) -> Dict:
    """Hard for A/A and numerics_unchanged candidates in scripted mode; reported otherwise."""
    means = [a["mean"] for a in agreement if a["n_streams"]]
    mean = sum(means) / len(means) if means else float("nan")
    hard = meta["load"]["mode"] == "scripted" and (
        meta["control"]["name"] == meta["candidate"]["name"] or bool(meta["candidate"].get("numerics_unchanged")))
    calib = noise.get("timed_output_agreement")
    threshold = calib["threshold"] if calib else config.TIMED_OUTPUT_AGREEMENT_MIN
    above = mean >= threshold
    return {"hard": hard, "mean": mean, "threshold": threshold, "calibrated_from": calib and calib["from_exp"],
            "above_threshold": above, "ok": above or not hard}


def server_arg_diff(control: Dict, candidate: Dict, declared: List[str]) -> Dict:
    """server_info paths that differ between the arms, and those no declared flag accounts for."""
    a, b = control["server_info"], candidate["server_info"]
    fa = _flatten({k: v for k, v in a.items() if k not in _VOLATILE_SERVER_KEYS | {"launch_command"}})
    fb = _flatten({k: v for k, v in b.items() if k not in _VOLATILE_SERVER_KEYS | {"launch_command"}})
    diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    declared_keys = {d.lstrip("-").replace("-", "_") for d in declared if d.startswith("--")}
    covered = set(declared_keys)
    for key in declared_keys:
        covered.update(_FLAG_DERIVED_PATHS.get(key, ()))
    undeclared = [k for k in diff if not _covered(k, covered)]
    flags_a, flags_b = _launch_flags(a.get("launch_command")), _launch_flags(b.get("launch_command"))
    undeclared += sorted(f"launch_command:{f}" for f in set(flags_a) ^ set(flags_b)
                         if f.lstrip("-").replace("-", "_") not in declared_keys)
    return {"differing_keys": diff, "undeclared": undeclared}


def run_gate(control_name: str, candidate_name: str, n_pairs: int, label: str, notes: str = "",
             decide_on: str = stats.DEFAULT_DECIDING_METRIC, load: config.SessionLoad = config.GATED_LOAD,
             sessions_path: str = config.SESSIONS_PATH, nonce_at: str = "start") -> Dict:
    control_ref, candidate_ref = server.load_ref(control_name), server.load_ref(candidate_name)
    exp_id = new_exp_id(label)
    run_dir = os.path.join(config.RUNS_DIR, exp_id)
    os.makedirs(run_dir)
    tok = load_tokenizer()
    sessions = schema.read(sessions_path)
    nonce = secrets.token_hex(4)
    meta = {"exp_id": exp_id, "control": control_ref, "candidate": candidate_ref, "n_pairs": n_pairs,
            "notes": notes, "decide_on": decide_on, "nonce": nonce, "load": msgspec.to_builtins(load),
            "nonce_at": nonce_at, "sessions": {"path": sessions_path, "sha256": _sha256(sessions_path),
                                               "n": len(sessions)},
            "fidelity_prompts_digest": fidelity.prompts_digest() if os.path.exists(fidelity.PROMPTS_PATH) else None,
            "environment": environment(), "checkpoint": checkpoint.verify()}
    _write(os.path.join(run_dir, "meta.json"), meta)
    refs = {"control": control_ref, "candidate": candidate_ref}
    legs: Dict[str, List[Dict]] = {"control": [], "candidate": []}
    with hostwatch.host_lock():
        for k, order in enumerate(abba_order(n_pairs)):
            for role in order:
                leg_dir = os.path.join(run_dir, f"pair{k}-{role}")
                leg = run_leg_with_retries(refs[role], f"{nonce}/pair{k}", leg_dir, sessions, tok, load,
                                           with_fidelity=(k == 0), nonce_at=nonce_at)
                leg["pair"], leg["role"] = k, role
                _write(os.path.join(leg_dir, "leg.json"), leg)
                legs[role].append(leg)
    report = evaluate(meta, legs)
    _write(os.path.join(run_dir, "report.json"), report)
    append_ledger(meta, report, run_dir)
    return report


def evaluate(meta: Dict, legs: Dict[str, List[Dict]]) -> Dict:
    control, candidate = legs["control"], legs["candidate"]
    summary = stats.summarize_pairs(control, candidate)
    noise_file = _load_noise_file()
    noise = noise_file.get("per_pair_log_sigma", {})
    timing = stats.timing_verdict(summary, noise, meta.get("decide_on", stats.DEFAULT_DECIDING_METRIC))
    agreement = [timed_output_agreement(c, k) for c, k in zip(control, candidate)]
    fid = fidelity_report(legs)
    args_diff = server_arg_diff(control[0], candidate[0], meta["candidate"].get("server_args", []) +
                                meta["control"].get("server_args", []))
    w_c, w_k = control[0]["weights_at_load"], candidate[0]["weights_at_load"]
    weights_match = w_c.get("checksum") == w_k.get("checksum") if w_c["ok"] and w_k["ok"] else None
    plans = {l["replay"]["plan_digest"] + "/" + str(l["pair"]) for ls in legs.values() for l in ls}
    integrity = {
        "legs_ok": all(l["integrity"]["ok"] for ls in legs.values() for l in ls),
        "per_leg": [{"pair": l["pair"], "role": l["role"], **l["integrity"]} for ls in legs.values() for l in ls],
        "same_plan_within_pairs": len(plans) == len(control),
        "timed_output_agreement_check": agreement_check(meta, agreement, noise_file),
        "undeclared_server_arg_diffs": args_diff["undeclared"],
        "weights_match_control": weights_match,
    }
    integrity["ok"] = (
        integrity["legs_ok"]
        and integrity["same_plan_within_pairs"]
        and integrity["timed_output_agreement_check"]["ok"]
        and not args_diff["undeclared"]
        and (weights_match is not False or meta["candidate"]["weight_layout_change"])
        and meta["checkpoint"]["matches_pin"]
    )
    fid_pass = (fid.get("candidate", {}).get("verdict") or {}).get("pass")
    return {
        "exp_id": meta["exp_id"],
        "control": meta["control"]["name"],
        "candidate": meta["candidate"]["name"],
        "load": meta["load"],
        "summary": summary,
        "noise_used": noise,
        "timing": timing,
        "timed_output_agreement": agreement,
        "fidelity": fid,
        "server_arg_diff": args_diff,
        "integrity": integrity,
        "verdict": {
            "integrity_ok": integrity["ok"],
            "fidelity_pass": fid_pass,
            "timing_promote": timing["promote"],
            "promote": bool(integrity["ok"] and fid_pass and timing["promote"]),
        },
    }


def fidelity_report(legs: Dict[str, List[Dict]]) -> Dict:
    """Each arm's pair-0 fidelity passes against the pinned reference, with the verdict."""
    thresholds = fidelity.load_json(fidelity.THRESHOLDS_PATH) if os.path.exists(fidelity.THRESHOLDS_PATH) else None
    fid = {}
    if os.path.exists(fidelity.REFERENCE_PATH):
        reference = fidelity.load_json(fidelity.REFERENCE_PATH)
        ref_forced = fidelity.load_json(fidelity.REFERENCE_FORCED_PATH)
        for role, ls in legs.items():
            if "fidelity_outputs" not in ls[0]:
                continue
            cmp = fidelity.compare(reference, ls[0]["fidelity_outputs"])
            cmp_forced = fidelity.compare_forced(reference, ref_forced, ls[0]["fidelity_forced"])
            fid[role] = {"compare": cmp, "compare_forced": cmp_forced,
                         "verdict": fidelity.verdict(cmp, cmp_forced, thresholds) if thresholds else None}
    return fid


NOISE_PATH = os.path.join(config.REFERENCE_DIR, "noise.json")


def _load_noise_file() -> Dict:
    return fidelity.load_json(NOISE_PATH) if os.path.exists(NOISE_PATH) else {}


def save_noise_from(report_path: str) -> Dict:
    report = fidelity.load_json(report_path)
    if report["control"] != report["candidate"]:
        raise ValueError("noise must come from an A/A run (control == candidate)")
    noise = {"from_exp": report["exp_id"], "load": report["load"]["name"],
             "per_pair_log_sigma": report["summary"]["per_pair_log_sigma"],
             "bars": {m: stats.promotion_bar(s) for m, s in report["summary"]["per_pair_log_sigma"].items()}}
    pair_means = [a["mean"] for a in report["timed_output_agreement"] if a["n_streams"]]
    if pair_means:
        noise["timed_output_agreement"] = {"from_exp": report["exp_id"], "pair_means": pair_means,
                                           "threshold": agreement_threshold(pair_means)}
    fidelity.save_json(NOISE_PATH, noise)
    return noise


def _ledger_harness(harness: Dict) -> str:
    return harness["commit"] or f"tree-sha256:{harness.get('tree_sha256')}"


def append_ledger(meta: Dict, report: Dict, run_dir: str) -> None:
    os.makedirs(os.path.dirname(config.LEDGER), exist_ok=True)
    pooled = report["summary"]["pooled"]
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
        "harness_commit": _ledger_harness(meta["environment"]["harness"]),
        "load": meta["load"]["name"],
        "n_pairs": meta["n_pairs"],
        "metrics": {**report["summary"]["overall"],
                    **{f"{role}_{k}": pooled[role][k] for role in ("control", "candidate")
                       for k in ("e2e_p90_s", "output_tok_s_per_gpu", "prefix_cache_hit_rate")}},
        "decided_on": report["timing"]["decided_on"],
        "verdict": report["verdict"],
        "output": os.path.relpath(os.path.join(run_dir, "report.json"), config.ROOT),
        "note": meta["notes"],
    }
    with open(config.LEDGER, "a") as f:
        f.write(json.dumps(line) + "\n")


def _sha256(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _write(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
