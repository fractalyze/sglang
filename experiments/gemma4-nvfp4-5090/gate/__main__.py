"""gemma4nv verify gate.

  gate run --control base --candidate <ref> [--pairs 4] [--label L] [--notes N]
  gate calibrate --ref base            # fidelity reference + KL thresholds (once per baseline)
  gate set-noise --report <A/A report.json>
  gate quality --ref <ref> [--set-baseline]
  gate prebuild --ref base            # JIT + autotune once, capped and watched
  gate vllm-ref                       # ungated vLLM 0.20 reference on W8/W1/W32
  gate peaks                           # measured DRAM BW, GEMM peaks, launch floor
  gate sol --ref base                  # router recording + SOL tables
  gate sol-report --report <report.json>
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

from gate import config, fidelity, hostwatch, prompts, quality, runner, server, solrun, vllm_ref

PEAKS_PATH = os.path.join(config.REFERENCE_DIR, "peaks.json")
SOL_DIR = os.path.join(config.REFERENCE_DIR, "sol")
QUALITY_BASELINE = os.path.join(config.REFERENCE_DIR, "quality_baseline.json")


def _calibrate(args) -> None:
    if os.path.exists(fidelity.REFERENCE_PATH) and not args.force:
        sys.exit(f"{fidelity.REFERENCE_PATH} exists; the reference is pinned (use --force to replace it)")
    ref = server.load_ref(args.ref)
    out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id("calibrate"))
    os.makedirs(out_dir)
    with hostwatch.host_lock(), server.Server(ref, os.path.join(out_dir, "server.log")) as srv:
        reference = fidelity.run(srv.url, concurrency=fidelity.REFERENCE_CONCURRENCY)
        serial = fidelity.run(srv.url, concurrency=fidelity.CALIBRATION_CONCURRENCY)
        repeat = fidelity.run(srv.url, concurrency=fidelity.REFERENCE_CONCURRENCY)
        ref_forced = fidelity.run_forced(srv.url, reference, concurrency=fidelity.REFERENCE_CONCURRENCY)
        serial_forced = fidelity.run_forced(srv.url, reference, concurrency=fidelity.CALIBRATION_CONCURRENCY)
        info = {"commit": srv.commit, "server_info": srv.server_info(), "weights": srv.weight_checksum()}
    calib = fidelity.compare(reference, serial)
    calib_forced = fidelity.compare_forced(reference, ref_forced, serial_forced)
    # Teacher-forced prefill argmax vs the decode-path greedy token of the same run: checks the
    # row alignment and measures how often prefill and decode numerics pick different tokens.
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
    ref = server.load_ref(args.ref)
    out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"quality-{args.ref}"))
    os.makedirs(out_dir)
    tok = prompts.load_tokenizer()
    with hostwatch.host_lock(), server.Server(ref, os.path.join(out_dir, "server.log")) as srv:
        res = quality.run(srv.url, tok)
        res["commit"] = srv.commit
    res["ref"] = args.ref
    if args.set_baseline:
        fidelity.save_json(QUALITY_BASELINE, res)
    elif os.path.exists(QUALITY_BASELINE):
        res["verdict"] = quality.verdict(res, fidelity.load_json(QUALITY_BASELINE))
    fidelity.save_json(os.path.join(out_dir, "quality.json"), res)
    print(json.dumps({k: res[k] for k in ("gsm8k", "tool_json", "verdict") if k in res}, indent=1))


def _peaks(args) -> None:
    out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id("peaks"))
    os.makedirs(out_dir)
    tmp = os.path.join(out_dir, "peaks.json")
    with hostwatch.host_lock():
        res = hostwatch.run_wrapped([config.VENV_PYTHON, "-m", "gate.peaks", tmp],
                                    os.path.join(out_dir, "hostmem.csv"), os.path.join(out_dir, "peaks.log"))
    if res["returncode"] != 0:
        sys.exit(f"peaks failed: {res}")
    peaks = json.load(open(tmp))
    peaks["measured_at"] = time.strftime("%Y-%m-%dT%H:%M")
    peaks["run_dir"] = out_dir
    fidelity.save_json(PEAKS_PATH, peaks)
    print(json.dumps(peaks, indent=1))


def _prebuild(args) -> None:
    """JIT/autotune step before any timed launch: the ref's exact server, every timed shape once, then stop."""
    ref = server.load_ref(args.ref)
    out_dir = os.path.join(config.RUNS_DIR, runner.new_exp_id(f"prebuild-{args.ref}"))
    os.makedirs(out_dir)
    tok = prompts.load_tokenizer()
    corpus = prompts.load_corpus(tok)
    with hostwatch.host_lock():
        env = dict(os.environ, MAX_JOBS=str(config.JIT_MAX_JOBS), PYTHONPATH=os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        jit = hostwatch.run_wrapped([ref["python"], "-m", "gate.jit_prebuild"], os.path.join(out_dir, "jit-hostmem.csv"),
                                    os.path.join(out_dir, "jit.log"), env=env)
        if jit["returncode"] != 0:
            fidelity.save_json(os.path.join(out_dir, "prebuild.json"), {"jit": jit})
            sys.exit(f"JIT prebuild failed: {jit['returncode']} (watchdog: {jit['tripped']})")
        srv = server.Server(ref, os.path.join(out_dir, "server.log"))
        try:
            srv.start(timeout_s=3600)
            srv._watchdog.set_phase("first_requests")
            runner.warm_up(srv, corpus, tok.bos_token_id, "prebuild")
        finally:
            srv.stop()
    res = {"ref": args.ref, "commit": srv.commit, "jit": jit, "preflight": srv.preflight, "host": srv.host_summary,
           "env": {k: os.environ.get(k) for k in ("MAX_JOBS", "FLASHINFER_NVCC_THREADS", "NVCC_THREADS",
                                                  "TORCH_CUDA_ARCH_LIST", "FLASHINFER_WORKSPACE_BASE")}}
    fidelity.save_json(os.path.join(out_dir, "prebuild.json"), res)
    print(json.dumps(res, indent=1))


def _sol(args) -> None:
    with hostwatch.host_lock():
        res = solrun.run(args.ref, SOL_DIR, PEAKS_PATH)
    print(json.dumps({wl: {"decode_step_ms": w["decode"]["sol_step_ms"], "prefill": w["prefill"],
                           "distinct_experts_mean": w["distinct_experts_mean"]}
                      for wl, w in res["workloads"].items()}, indent=1))


def sol_fractions(report: dict, legs_dir: str, sol_res: dict) -> dict:
    """Achieved times of the control legs vs the SOL bounds (sol_fraction = t_sol / t_achieved)."""
    legs = []
    for name in sorted(os.listdir(legs_dir)):
        path = os.path.join(legs_dir, name, "leg.json")
        if name.endswith("-control") and os.path.exists(path):
            legs.append(json.load(open(path)))
    out = {}
    for wl in ("W8", "W1", "W32"):
        w = sol_res["workloads"][wl]
        reps = [r for leg in legs for r in leg["workloads"][wl]]
        decode_steps = w["decode"] - 1
        step_s = [sum(s["e2e_s"] - s["ttft_s"] for s in r["streams"]) / len(r["streams"]) / decode_steps for r in reps]
        prefill_s = [max(s["ttft_s"] for s in r["streams"]) for r in reps]
        achieved_step = sum(step_s) / len(step_s)
        achieved_prefill = sum(prefill_s) / len(prefill_s)
        sol_step = w["decode"]["sol_step_ms"] / 1e3
        out[wl] = {
            "achieved_decode_step_ms": 1e3 * achieved_step,
            "sol_decode_step_ms": 1e3 * sol_step,
            "decode_sol_fraction": sol_step / achieved_step,
            "achieved_batch_prefill_ms": 1e3 * achieved_prefill,
            "sol_prefill_ms_nvfp4_fp8": 1e3 * w["prefill"]["sol_s_nvfp4_fp8"],
            "sol_prefill_ms_as_served": 1e3 * w["prefill"]["sol_s_as_served"],
            "prefill_sol_fraction_nvfp4_fp8": w["prefill"]["sol_s_nvfp4_fp8"] / achieved_prefill,
            "prefill_sol_fraction_as_served": w["prefill"]["sol_s_as_served"] / achieved_prefill,
            "audit_below_sol": achieved_step < sol_step or achieved_prefill < w["prefill"]["sol_s_nvfp4_fp8"],
        }
        if wl == "W32":
            toks = sum(s["output_tokens"] for r in reps for s in r["streams"])
            out[wl]["achieved_tok_s"] = toks / sum(r["wall_s"] for r in reps)
    return out


def _sol_report(args) -> None:
    sol_res = json.load(open(os.path.join(SOL_DIR, "sol.json")))
    report = json.load(open(args.report))
    res = sol_fractions(report, os.path.dirname(args.report), sol_res)
    fidelity.save_json(os.path.join(os.path.dirname(args.report), "sol_fractions.json"), res)
    print(json.dumps(res, indent=1))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="gate")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--control", required=True)
    r.add_argument("--candidate", required=True)
    r.add_argument("--pairs", type=int, default=config.MIN_PAIRS)
    r.add_argument("--label", default=None)
    r.add_argument("--notes", default="")
    c = sub.add_parser("calibrate")
    c.add_argument("--ref", default="base")
    c.add_argument("--force", action="store_true")
    n = sub.add_parser("set-noise")
    n.add_argument("--report", required=True)
    q = sub.add_parser("quality")
    q.add_argument("--ref", required=True)
    q.add_argument("--set-baseline", action="store_true")
    sub.add_parser("peaks")
    sub.add_parser("vllm-ref")
    rv = sub.add_parser("reevaluate")
    rv.add_argument("--run", required=True)
    pb = sub.add_parser("prebuild")
    pb.add_argument("--ref", default="base")
    s = sub.add_parser("sol")
    s.add_argument("--ref", default="base")
    sr = sub.add_parser("sol-report")
    sr.add_argument("--report", required=True)
    args = p.parse_args()

    if args.cmd == "run":
        if args.pairs < config.MIN_PAIRS:
            sys.exit(f"--pairs must be >= {config.MIN_PAIRS}")
        label = args.label or ("AA" if args.control == args.candidate else args.candidate)
        report = runner.run_gate(args.control, args.candidate, args.pairs, label, args.notes)
        print(json.dumps({"exp_id": report["exp_id"], "overall": report["summary"]["overall"],
                          "per_pair_log_sigma": report["summary"]["per_pair_log_sigma"],
                          "bars": report["timing"]["bars"], "verdict": report["verdict"],
                          "integrity": {k: v for k, v in report["integrity"].items() if k != "per_leg"}}, indent=1))
    elif args.cmd == "calibrate":
        _calibrate(args)
    elif args.cmd == "set-noise":
        print(json.dumps(runner.save_noise_from(args.report), indent=1))
    elif args.cmd == "quality":
        _quality(args)
    elif args.cmd == "reevaluate":
        report = runner.reevaluate(args.run)
        print(json.dumps({"verdict": report["verdict"],
                          "integrity": {k: v for k, v in report["integrity"].items() if k != "per_leg"}}, indent=1))
    elif args.cmd == "vllm-ref":
        res = vllm_ref.run(os.path.join(config.RUNS_DIR, runner.new_exp_id("vllm-ref")))
        print(json.dumps({"summary": res["summary"], "host": res["host"]}, indent=1))
    elif args.cmd == "prebuild":
        _prebuild(args)
    elif args.cmd == "peaks":
        _peaks(args)
    elif args.cmd == "sol":
        _sol(args)
    elif args.cmd == "sol-report":
        _sol_report(args)


if __name__ == "__main__":
    main()
