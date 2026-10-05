"""g4poc gate.

  gate run --control base --candidate <ref> [--pairs 4] [--load L64] [--decide-on METRIC] [--nonce-at start]
  gate set-noise --report <A/A report.json>
  gate smoke --ref base [--load L8-smoke]          # one server, one replay, metrics printed (no verdict)
  gate sweep --ref base [--load inflight|L64] --concurrency 8,16,32,64   # capacity at every SLO and price
  gate calibrate --ref base                        # fidelity reference + KL thresholds (once per baseline)
  gate quality --ref <ref> [--gsm8k-n N|all] [--set-baseline]
  gate quality-compare --control <quality.json> --candidate <quality.json>
  gate rp-quality --ref <ref> [--set-baseline]      # role-play reference consistency
  gate rp-judge --candidate <rp.json> (--judge-ref <ref> | --judge-url URL --judge-model M)
  gate pd-measure --ref base [--prefill-lens 5120,10240] [--batches 8,16,32,64]
  gate pd-model --pd <pd.json> --sweep <sweep.json> [--slo 10] [--link-gbps 10,25,100] [--sessions 2200]
  gate prebuild --ref base                         # first launch: JIT + autotune, capped and watched
  gate pin-checkpoint --source "<how the FP8 files were made>"   # once per served checkpoint

Every server launch holds the host lock and runs inside the memory-capped scope.
"""

import argparse
import json
import os
import sys

import msgspec

from gate import checkpoint, config, fidelity, hostwatch, metrics, pd, quality, rpquality, runner, server, stats
from workload import schema

QUALITY_BASELINE = os.path.join(config.REFERENCE_DIR, "quality_baseline.json")
RP_BASELINE = os.path.join(config.REFERENCE_DIR, "rp_baseline.json")


def _out_dir(label: str) -> str:
    d = os.path.join(config.RUNS_DIR, runner.new_exp_id(label))
    os.makedirs(d)
    return d


def _server(ref_name: str, out_dir: str) -> server.Server:
    ref = server.load_ref(ref_name)
    return server.Server(ref, os.path.join(out_dir, "server.log"), extra_args=runner.server_extra_args(ref))


def _load(name: str, concurrency: int = 0) -> config.SessionLoad:
    load = config.LOADS[name]
    if not concurrency:
        return load
    return msgspec.structs.replace(load, concurrency=concurrency, name=f"{name}-C{concurrency}")


def _replay_summary(srv, sessions, tok, load, seed: str, nonce_at: str) -> dict:
    leg = runner.replay_leg(srv, sessions, tok, load, seed, nonce_at)
    return {"load": msgspec.to_builtins(load), "summary": leg["summary"], "retractions": leg["retractions"],
            "gauges": leg["gauges"], "counter_delta": leg["counter_delta"], "decode_steps": leg["decode_steps"],
            "plan_digest": leg["replay"]["plan_digest"], "abandoned": leg["replay"]["abandoned"],
            "failed": leg["replay"]["failed"]}


def _smoke(args) -> None:
    out_dir = _out_dir(f"smoke-{args.ref}")
    tok, sessions = runner.load_tokenizer(), schema.read(args.sessions)
    load = _load(args.load)
    with hostwatch.host_lock(), _server(args.ref, out_dir) as srv:
        runner.warm_up(srv, sessions, tok, "smoke")
        res = _replay_summary(srv, sessions, tok, load, "smoke", args.nonce_at)
        res["server_info"] = srv.server_info()
        res["backends"] = srv.backend_report()
    res["host"] = srv.host_summary
    res["meets_slo"] = {f"{slo:g}": metrics.meets_slo(res["summary"], slo) for slo in config.SLOS_E2E_P90_S}
    res["usd_per_mtok_output"] = metrics.cost_table(res["summary"]["output_tok_s_per_gpu"])
    res["usd_per_mtok_total"] = metrics.cost_table(res["summary"]["total_tok_s_per_gpu"])
    fidelity.save_json(os.path.join(out_dir, "smoke.json"), res)
    print(json.dumps({k: res[k] for k in ("summary", "retractions", "meets_slo", "usd_per_mtok_output",
                                          "usd_per_mtok_total")}, indent=1))
    print(f"run dir: {out_dir}", file=sys.stderr)


def _sweep(args) -> None:
    out_dir = _out_dir(f"sweep-{args.ref}")
    tok, sessions = runner.load_tokenizer(), schema.read(args.sessions)
    points = []
    with hostwatch.host_lock(), _server(args.ref, out_dir) as srv:
        runner.warm_up(srv, sessions, tok, "sweep")
        for c in (int(x) for x in args.concurrency.split(",")):
            srv.flush_cache()
            load = _load(args.load, c)
            res = _replay_summary(srv, sessions, tok, load, f"sweep-{c}", args.nonce_at)
            points.append({"load": load.name, **res})
            fidelity.save_json(os.path.join(out_dir, "points.json"), points)
            if res["summary"]["e2e_p90_s"] > 2 * max(config.SLOS_E2E_P90_S):
                break
        info = srv.server_info()
    res = {"ref": args.ref, "points": points, "capacity": metrics.capacity_at_slos(points), "server_info": info,
           "host": srv.host_summary}
    fidelity.save_json(os.path.join(out_dir, "sweep.json"), res)
    print(json.dumps(res["capacity"], indent=1))
    print(f"run dir: {out_dir}", file=sys.stderr)


def _calibrate(args) -> None:
    if os.path.exists(fidelity.REFERENCE_PATH) and not args.force:
        sys.exit(f"{fidelity.REFERENCE_PATH} exists; the reference is pinned (use --force to replace it)")
    out_dir = _out_dir("calibrate")
    with hostwatch.host_lock(), _server(args.ref, out_dir) as srv:
        reference = fidelity.run(srv.url, concurrency=fidelity.REFERENCE_CONCURRENCY)
        srv.flush_cache()
        serial = fidelity.run(srv.url, concurrency=fidelity.CALIBRATION_CONCURRENCY)
        srv.flush_cache()
        repeat = fidelity.run(srv.url, concurrency=fidelity.REFERENCE_CONCURRENCY)
        srv.flush_cache()
        ref_forced = fidelity.run_forced(srv.url, reference, concurrency=fidelity.REFERENCE_CONCURRENCY)
        srv.flush_cache()
        serial_forced = fidelity.run_forced(srv.url, reference, concurrency=fidelity.CALIBRATION_CONCURRENCY)
        info = {"commit": srv.commit, "server_info": srv.server_info(), "weights": srv.weight_checksum()}
    calib = fidelity.compare(reference, serial)
    calib_forced = fidelity.compare_forced(reference, ref_forced, serial_forced)
    self_forced = fidelity.compare_forced(reference, ref_forced, ref_forced)
    thresholds = fidelity.thresholds_from_calibration(calib, calib_forced)
    thresholds.update({"ref": args.ref, "commit": info["commit"], "prompts_digest": fidelity.prompts_digest(),
                       "same_composition_repeat": fidelity.compare(reference, repeat),
                       "forced_vs_decode_self_agreement": {k: self_forced[k] for k in ("min_top1_agreement",
                                                                                       "mean_top1_agreement")}})
    fidelity.save_json(fidelity.REFERENCE_PATH, reference)
    fidelity.save_json(fidelity.REFERENCE_FORCED_PATH, ref_forced)
    fidelity.save_json(fidelity.THRESHOLDS_PATH, thresholds)
    fidelity.save_json(os.path.join(out_dir, "calibration.json"),
                       {"serial_vs_batched": calib, "serial_vs_batched_forced": calib_forced,
                        "forced_self": self_forced, "thresholds": thresholds, "server": info})
    print(json.dumps({k: v for k, v in thresholds.items() if k != "same_composition_repeat"}, indent=1))


def _quality(args) -> None:
    out_dir = _out_dir(args.label or f"quality-{args.ref}")
    tok = runner.load_tokenizer()
    gsm8k_n = None if args.gsm8k_n == "all" else int(args.gsm8k_n)
    with hostwatch.host_lock(), _server(args.ref, out_dir) as srv:
        res = quality.run(srv.url, tok, gsm8k_n)
        res["commit"] = srv.commit
    res["ref"] = args.ref
    res["harness"] = server.harness_commit()
    if args.set_baseline:
        fidelity.save_json(QUALITY_BASELINE, res)
    elif os.path.exists(QUALITY_BASELINE):
        res["verdict"] = quality.verdict(res, fidelity.load_json(QUALITY_BASELINE))
    fidelity.save_json(os.path.join(out_dir, "quality.json"), res)
    print(json.dumps({k: {kk: vv for kk, vv in res[k].items() if kk != "correct"} if k != "verdict" else res[k]
                      for k in ("gsm8k", "tool_json", "verdict") if k in res}, indent=1))


def _rp_quality(args) -> None:
    out_dir = _out_dir(args.label or f"rp-quality-{args.ref}")
    if args.set_baseline:
        items, reference = rpquality.build_items(schema.read(args.sessions), runner.load_tokenizer()), None
    else:
        base = fidelity.load_json(RP_BASELINE)
        items, reference = base["items"], base["results"]
    with hostwatch.host_lock(), _server(args.ref, out_dir) as srv:
        results = rpquality.run(srv.url, items, reference)
        commit = srv.commit
    res = {"ref": args.ref, "commit": commit, "harness": server.harness_commit(), "results": results}
    if args.set_baseline:
        fidelity.save_json(RP_BASELINE, {**res, "items": items})
    else:
        res["verdict"] = rpquality.consistency_verdict(reference, results)
        print(json.dumps(res["verdict"], indent=1))
    fidelity.save_json(os.path.join(out_dir, "rp.json"), res)
    print(f"run dir: {out_dir}", file=sys.stderr)


def _rp_judge(args) -> None:
    base = fidelity.load_json(RP_BASELINE)
    cand = fidelity.load_json(args.candidate)
    if args.judge_ref:
        out_dir = _out_dir(f"rp-judge-{args.judge_ref}")
        with hostwatch.host_lock(), _server(args.judge_ref, out_dir) as srv:
            res = rpquality.judge(srv.url, "default", base["items"], base["results"], cand["results"])
        res["judge"]["ref"] = args.judge_ref
    else:
        res = rpquality.judge(args.judge_url, args.judge_model, base["items"], base["results"], cand["results"])
    fidelity.save_json(os.path.join(os.path.dirname(os.path.abspath(args.candidate)), "rp_judge.json"), res)
    print(json.dumps(res["verdict"], indent=1))


def _log_since(srv: server.Server, offset: int) -> str:
    srv.log_offset()  # flushes the log file
    with open(srv.log_path, errors="replace") as f:
        f.seek(offset)
        return f.read()


def _pd_measure(args) -> None:
    out_dir = _out_dir(f"pd-{args.ref}")
    tok, sessions = runner.load_tokenizer(), schema.read(args.sessions)
    res = {"ref": args.ref, "prefill": [], "decode": []}
    with hostwatch.host_lock(), _server(args.ref, out_dir) as srv:
        runner.warm_up(srv, sessions, tok, "pd")
        for length in (int(x) for x in args.prefill_lens.split(",")):
            for in_flight in (int(x) for x in args.prefill_in_flight.split(",")):
                srv.flush_cache()
                prompts = pd.prompts_of_length(sessions, tok, length, in_flight * args.prefill_rounds,
                                               f"pd-prefill-{length}-{in_flight}")
                res["prefill"].append(pd.prefill_point(srv.url, prompts, in_flight))
                fidelity.save_json(os.path.join(out_dir, "pd.json"), res)
        for batch in (int(x) for x in args.batches.split(",")):
            srv.flush_cache()
            prompts = pd.prompts_of_length(sessions, tok, args.decode_len, batch, f"pd-decode-{batch}")
            offset = srv.log_offset()
            res["decode"].append(pd.decode_point(srv.url, prompts, log_since=lambda: _log_since(srv, offset)))
            fidelity.save_json(os.path.join(out_dir, "pd.json"), res)
        res["server_info"] = srv.server_info()
    res["host"] = srv.host_summary
    res["best_prefill"] = pd.best_prefill(res["prefill"])
    res["best_decode"] = pd.best_decode(res["decode"], args.max_tpot)
    res["max_tpot_s"] = args.max_tpot
    fidelity.save_json(os.path.join(out_dir, "pd.json"), res)
    print(json.dumps({k: res[k] for k in ("best_prefill", "best_decode")}, indent=1))
    print(f"run dir: {out_dir}", file=sys.stderr)


def _pd_model(args) -> None:
    pdm = fidelity.load_json(args.pd)
    sw = fidelity.load_json(args.sweep)
    best = sw["capacity"]["capacity"].get(f"{args.slo:g}")
    if best is None or pdm["best_decode"] is None:
        sys.exit(f"no sweep point meets E2E p90 <= {args.slo:g} s, or no decode point under the TPOT bound")
    point = next(p for p in sw["points"] if p["load"] == best["load"])["summary"]
    wl = pd.Workload(mean_prompt_tokens=point["mean_prompt_tokens"], hit_rate=point["prefix_cache_hit_rate"],
                     mean_output_tokens=point["mean_output_tokens"])
    rp, rd = pdm["best_prefill"]["prefill_tok_s"], pdm["best_decode"]["decode_tok_s"]
    gc = best["goodput_output_tok_s_per_gpu"]
    links = [float(x) for x in args.link_gbps.split(",")] if args.link_gbps else [None]
    res = {"models": [pd.fleet_model(wl, rp, rd, gc, link) for link in links],
           "fleet": pd.fleet_size(wl, args.sessions, args.think_s, point["e2e_mean_s"], rp, rd, gc),
           "inputs": {"pd": args.pd, "sweep": args.sweep, "sweep_point": best["load"], "slo_e2e_p90_s": args.slo}}
    print(json.dumps(res, indent=1))


def _prebuild(args) -> None:
    """First launch of a ref: JIT and autotune under the host cap, then one warm-up replay."""
    out_dir = _out_dir(f"prebuild-{args.ref}")
    tok, sessions = runner.load_tokenizer(), schema.read(args.sessions)
    srv = _server(args.ref, out_dir)
    with hostwatch.host_lock():
        try:
            srv.start(timeout_s=3600)
            srv._watchdog.set_phase("first_requests")
            runner.warm_up(srv, sessions, tok, "prebuild")
        finally:
            srv.stop()
    res = {"ref": args.ref, "commit": srv.commit, "preflight": srv.preflight, "host": srv.host_summary}
    fidelity.save_json(os.path.join(out_dir, "prebuild.json"), res)
    print(json.dumps(res, indent=1))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gate")
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_sessions(sp):
        sp.add_argument("--sessions", default=config.SESSIONS_PATH)
        return sp

    r = with_sessions(sub.add_parser("run"))
    r.add_argument("--control", required=True)
    r.add_argument("--candidate", required=True)
    r.add_argument("--pairs", type=int, default=config.MIN_PAIRS)
    r.add_argument("--load", default=config.GATED_LOAD.name, choices=sorted(config.LOADS))
    r.add_argument("--label", default=None)
    r.add_argument("--notes", default="")
    r.add_argument("--decide-on", default=stats.DEFAULT_DECIDING_METRIC, choices=list(stats.DECIDING_METRICS))
    r.add_argument("--nonce-at", default="start", choices=["start", "after_system"])
    sn = sub.add_parser("set-noise")
    sn.add_argument("--report", required=True)
    for name in ("smoke", "sweep"):
        sp = with_sessions(sub.add_parser(name))
        sp.add_argument("--ref", default="base")
        sp.add_argument("--load", default="L8-smoke" if name == "smoke" else config.GATED_LOAD.name,
                        choices=sorted(config.LOADS))
        sp.add_argument("--nonce-at", default="start", choices=["start", "after_system"])
        if name == "sweep":
            sp.add_argument("--concurrency", default="8,16,24,32,48,64")
    c = sub.add_parser("calibrate")
    c.add_argument("--ref", default="base")
    c.add_argument("--force", action="store_true")
    q = sub.add_parser("quality")
    q.add_argument("--ref", required=True)
    q.add_argument("--gsm8k-n", default=str(config.GSM8K_N))
    q.add_argument("--label", default=None)
    q.add_argument("--set-baseline", action="store_true")
    qc = sub.add_parser("quality-compare")
    qc.add_argument("--control", required=True)
    qc.add_argument("--candidate", required=True)
    qc.add_argument("--gsm8k-ci-low-min", type=float, default=-1.0)
    rq = with_sessions(sub.add_parser("rp-quality"))
    rq.add_argument("--ref", required=True)
    rq.add_argument("--label", default=None)
    rq.add_argument("--set-baseline", action="store_true")
    rj = sub.add_parser("rp-judge")
    rj.add_argument("--candidate", required=True, help="the candidate's rp.json")
    rj.add_argument("--judge-ref", default=None)
    rj.add_argument("--judge-url", default=None)
    rj.add_argument("--judge-model", default="default")
    pm = with_sessions(sub.add_parser("pd-measure"))
    pm.add_argument("--ref", default="base")
    pm.add_argument("--prefill-lens", default="5120,10240")
    pm.add_argument("--prefill-in-flight", default="1,2,4,8")
    pm.add_argument("--prefill-rounds", type=int, default=4)
    pm.add_argument("--decode-len", type=int, default=5120)
    pm.add_argument("--batches", default="8,16,32,48,64")
    pm.add_argument("--max-tpot", type=float, default=0.030, help="p90 s per output token (300 tokens in 9 s)")
    pmo = sub.add_parser("pd-model")
    pmo.add_argument("--pd", required=True)
    pmo.add_argument("--sweep", required=True)
    pmo.add_argument("--slo", type=float, default=config.DEFAULT_SLO_E2E_P90_S)
    pmo.add_argument("--link-gbps", default="", help="comma list; empty until the bs2<->bs3 link is measured")
    pmo.add_argument("--sessions", type=int, default=2200)
    pmo.add_argument("--think-s", type=float, default=17.9, help="mean think time of the session file")
    pc = sub.add_parser("pin-checkpoint")
    pc.add_argument("--source", required=True)
    pb = with_sessions(sub.add_parser("prebuild"))
    pb.add_argument("--ref", default="base")
    return p


def main() -> None:
    args = _parser().parse_args()
    if args.cmd == "run":
        if args.pairs < config.MIN_PAIRS:
            sys.exit(f"--pairs must be >= {config.MIN_PAIRS}")
        label = args.label or ("AA" if args.control == args.candidate else args.candidate)
        report = runner.run_gate(args.control, args.candidate, args.pairs, label, args.notes, args.decide_on,
                                 config.LOADS[args.load], args.sessions, args.nonce_at)
        print(json.dumps({"exp_id": report["exp_id"], "overall": report["summary"]["overall"],
                          "hit_rate": report["summary"]["hit_rate"], "retractions": report["summary"]["retractions"],
                          "per_pair_log_sigma": report["summary"]["per_pair_log_sigma"],
                          "bars": report["timing"]["bars"], "verdict": report["verdict"],
                          "integrity": {k: v for k, v in report["integrity"].items() if k != "per_leg"}}, indent=1))
    elif args.cmd == "set-noise":
        print(json.dumps(runner.save_noise_from(args.report), indent=1))
    elif args.cmd == "smoke":
        _smoke(args)
    elif args.cmd == "sweep":
        _sweep(args)
    elif args.cmd == "calibrate":
        _calibrate(args)
    elif args.cmd == "quality":
        _quality(args)
    elif args.cmd == "quality-compare":
        res = quality.compare(fidelity.load_json(args.control), fidelity.load_json(args.candidate),
                              args.gsm8k_ci_low_min)
        print(json.dumps(res, indent=1))
    elif args.cmd == "rp-quality":
        _rp_quality(args)
    elif args.cmd == "rp-judge":
        if not (args.judge_ref or args.judge_url):
            sys.exit("give --judge-ref or --judge-url")
        _rp_judge(args)
    elif args.cmd == "pd-measure":
        _pd_measure(args)
    elif args.cmd == "pd-model":
        _pd_model(args)
    elif args.cmd == "prebuild":
        _prebuild(args)
    elif args.cmd == "pin-checkpoint":
        rec = checkpoint.pin(args.source)
        print(json.dumps({k: rec[k] for k in ("model_dir", "source")} | {"n_files": len(rec["files"])}, indent=1))


if __name__ == "__main__":
    main()
