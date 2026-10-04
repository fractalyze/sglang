"""W13 T-MOE1 step 0: FlashInfer's SM120 NVFP4 CUTLASS MoE tactics at verify widths.

Gemma-4-26B-A4B routed experts (128 x top-8, hidden 2816, intermediate 704, gated GELU) run
through flashinfer.fused_moe.cutlass_fused_moe exactly as SGLang's flashinfer_cutlass runner
calls it (bf16 input, NVFP4 weights, use_fused_finalize from SGLANG_FLASHINFER_MOE_FUSED_FINALIZE,
default False). Every tactic the runner enumerates is forced through ``profile_ids`` and timed;
no kernel is written. Weight values are random NVFP4 (timing does not depend on them); routing
is W8's recorded routing (``routing_timing.npz``, [tokens, 30 layers, top-8] per prompt),
windowed into verify groups of (1 + k) consecutive tokens; B streams' windows are stacked.

Parts:
  tactics   per (M, finalize): autotuned choice, then every GEMM1 tactic with GEMM2 fixed at the
            autotuned one and vice versa; kernel names from one profiled call per tactic
  padding   the autotuned tactic at 24 active experts with r rows each (r in 1, 2, 4, 8, 128)

Timing: CUDA graph of GRAPH_CALLS calls cycling over N_ROUTINGS routings and 2 weight copies,
best of 7 replays.

  python moe_tactic_bench.py --routing .../routing_timing.npz --out w13-moe-tactics.json
"""

import argparse
import json
import math

import numpy as np
import torch
from flashinfer import fp4_quantize
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

E, TOPK, HIDDEN, INTER = 128, 8, 2816, 704
K_DRAFT = 5
GRAPH_CALLS = 32
N_ROUTINGS = 16
N_WEIGHT_COPIES = 2


def make_weights(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w1 = torch.randn(E, 2 * INTER, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w2 = torch.randn(E, HIDDEN, INTER, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    gs1 = (448.0 * 6.0 / w1.abs().amax(dim=(1, 2))).float()
    gs2 = (448.0 * 6.0 / w2.abs().amax(dim=(1, 2))).float()
    q1, s1, q2, s2 = [], [], [], []
    for e in range(E):
        a, b = fp4_quantize(w1[e], gs1[e])
        q1.append(a)
        s1.append(b)
        a, b = fp4_quantize(w2[e], gs2[e])
        q2.append(a)
        s2.append(b)
    del w1, w2
    ones = torch.ones(E, device="cuda", dtype=torch.float32)
    return {
        "w13": torch.stack(q1).view(torch.long),
        "w2": torch.stack(q2).view(torch.long),
        "scales": [ones, torch.stack(s1).view(torch.int32), 1.0 / gs1,
                   ones, torch.stack(s2).view(torch.int32), 1.0 / gs2],
    }


def verify_routings(routing, b, n, rng):
    """n top-k id tensors [b * (1 + k), TOPK]: b prompts' windows of 1 + k tokens at one layer."""
    width = 1 + K_DRAFT
    out = []
    for _ in range(n):
        layer = int(rng.integers(0, routing[0].shape[1]))
        rows = []
        for p in rng.choice(len(routing), size=b, replace=b > len(routing)):
            t0 = int(rng.integers(0, routing[p].shape[0] - width))
            rows.append(routing[p][t0:t0 + width, layer])
        out.append(torch.from_numpy(np.concatenate(rows)).to("cuda", torch.int32))
    return out


def padding_routings(rows_per_expert, n, rng):
    """Every token picks one of 3 disjoint 8-expert groups: 24 active experts, r rows each."""
    out = []
    for _ in range(n):
        experts = torch.from_numpy(rng.permutation(E)[:24].astype(np.int32)).view(3, TOPK)
        ids = experts.repeat(rows_per_expert, 1)
        out.append(ids.to("cuda"))
    return out


class Moe:
    def __init__(self, weights, finalize):
        self.weights = weights
        self.finalize = finalize

    def call(self, x, ids, out, w, profile_ids=None):
        return cutlass_fused_moe(
            input=x, token_selected_experts=ids,
            token_final_scales=torch.full(ids.shape, 1.0 / TOPK, device="cuda"),
            fc1_expert_weights=w["w13"], fc2_expert_weights=w["w2"],
            output_dtype=torch.bfloat16, quant_scales=w["scales"], output=out,
            tune_max_num_tokens=1 << max(0, (x.shape[0] - 1).bit_length()),
            activation_type=ActivationType.Geglu, use_fused_finalize=self.finalize,
            profile_ids=profile_ids)[0]


def time_calls(fns):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(GRAPH_CALLS):
            fns[i % len(fns)]()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = math.inf
    for _ in range(7):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / GRAPH_CALLS)
    return best


def kernel_names(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name[:160] for e in prof.events() if e.device_type.name == "CUDA"})


def time_case(moe, weights, routings, profile_ids):
    m = routings[0].shape[0]
    xs = [torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in routings]
    outs = [torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in routings]
    fns = []
    for i, ids in enumerate(routings):
        w = weights[i % len(weights)]
        fns.append(lambda x=xs[i], ids=ids, o=outs[i], w=w: moe.call(x, ids, o, w, profile_ids))
    return time_calls(fns), fns[0]


def tactic_sweep(weights, routings, finalize, results, label):
    moe = Moe(weights, finalize)
    m = routings[0].shape[0]
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    # Autotune at this M (as SGLang's warmup does), then time the tuned call without profile_ids.
    with autotune(True):
        moe.call(x, routings[0], out, weights[0])
    torch.cuda.synchronize()
    ref = moe.call(x, routings[0], out.clone(), weights[0]).clone()
    tuned_us, tuned_fn = time_case(moe, weights, routings, None)
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    MoERunner = get_cutlass_fused_moe_module("120").MoERunner
    runner = next(iter(MoERunner.runner_dict.values()))
    for key, r in MoERunner.runner_dict.items():
        if key[7] == finalize:
            runner = r
    n1, n2 = runner.get_gemm1_tactic_count(), runner.get_gemm2_tactic_count()
    rec = {"case": label, "M": m, "finalize": finalize, "n_gemm1": n1, "n_gemm2": n2,
           "tuned_us": tuned_us, "tuned_kernels": kernel_names(tuned_fn), "rows": []}
    print(json.dumps({k: rec[k] for k in ("case", "M", "finalize", "n_gemm1", "n_gemm2", "tuned_us")}),
          flush=True)
    # GEMM1 and GEMM2 tactics time independently: sweep GEMM1 with the first GEMM2 tactic,
    # then GEMM2 with the fastest GEMM1.
    best_t1 = None
    for stage, pairs in (("gemm1", lambda: [(t, n1) for t in range(n1)]),
                         ("gemm2", lambda: [(best_t1, t) for t in range(n1, n1 + n2)])):
        for t1, t2 in pairs():
            row = {"stage": stage, "t1": t1, "t2": t2}
            try:
                y = moe.call(x, routings[0], out.clone(), weights[0], [t1, t2])
                torch.cuda.synchronize()
                row["max_abs_vs_tuned"] = (y.float() - ref.float()).abs().max().item()
                row["us"], fn = time_case(moe, weights, routings, [t1, t2])
                row["kernels"] = [k for k in kernel_names(fn) if "emm" in k or "inalize" in k]
            except Exception as e:  # noqa: BLE001 - an unbuildable tactic is a recorded result
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            torch.cuda.synchronize()
            rec["rows"].append(row)
            print(label, finalize, stage, t1, t2, row.get("us"), row.get("error", ""), flush=True)
        if stage == "gemm1":
            ok = [r for r in rec["rows"] if "us" in r]
            best_t1 = min(ok, key=lambda r: r["us"])["t1"]
    results.append(rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--routing", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parts", default="tactics,padding")
    a = p.parse_args()
    d = np.load(a.routing)
    routing = [d[k] for k in d.files]
    rng = np.random.default_rng(0)
    weights = [make_weights(s) for s in range(N_WEIGHT_COPIES)]
    torch.cuda.synchronize()
    results = []
    parts = set(a.parts.split(","))
    if "tactics" in parts:
        for b in (1, 8):
            routings = verify_routings(routing, b, N_ROUTINGS, rng)
            for finalize in (False, True):
                tactic_sweep(weights, routings, finalize, results, f"verify_B{b}")
    if "padding" in parts:
        for finalize in (False, True):
            moe = Moe(weights, finalize)
            for r in (1, 2, 4, 8, 128):
                routings = padding_routings(r, N_ROUTINGS, rng)
                m = routings[0].shape[0]
                x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
                with autotune(True):
                    moe.call(x, routings[0], torch.empty_like(x), weights[0])
                us, _ = time_case(moe, weights, routings, None)
                rec = {"case": "padding", "rows_per_expert": r, "M": m, "finalize": finalize,
                       "tuned_us": us}
                results.append(rec)
                print(json.dumps(rec), flush=True)
    with open(a.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
