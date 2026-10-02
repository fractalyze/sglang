"""Per-component GPU time from a SGLang torch-profiler trace (one stage).

Decode runs inside a CUDA graph, so kernels carry no CPU-op parent. Instead,
each step's kernel stream is split at the lm_head GEMM and each layer is split
at the attention kernel; kernels are then labelled by name pattern plus their
position relative to the layer's attention kernel (BF16 GEMMs before attention
are qkv_proj, the first after is o_proj, and so on).

Usage: python classify_trace.py <trace.json.gz> [--dump-seq N] [--json-out f]
"""

import argparse
import collections
import gzip
import json
import re

ATTN = re.compile(r"_fwd_(grouped_)?kernel_stage1|_fwd_kernel_stage1|extend_attention|_fwd_kernel\b|flash|fmha|attention", re.I)
ATTN_STAGE2 = re.compile(r"stage2", re.I)
MOE = re.compile(r"moe|expert|cutlass.*grouped|Grouped|fp4.*gemm.*group|computeStrides|expandInputRows|finalizeMoe|doActivation|buildExpertMaps|blockExpertPrefix|globalExpertPrefix|mergeExpertPrefix|fusedBuildExpertMaps|topk", re.I)
ROUTING = re.compile(r"gemma4_fused_routing|routing", re.I)
NORM = re.compile(r"norm", re.I)
ACT = re.compile(r"gelu|act_and_mul|silu", re.I)
ROPE = re.compile(r"rope|rotary", re.I)
KVSTORE = re.compile(r"store_kv|set_kv|kv_buffer|copy_kernel|index_put|scatter", re.I)
GEMM = re.compile(r"gemm|nvjet|xmma|cutlass|sm\d+_|cublas|matmul|ampere|hopper|blackwell", re.I)
SOFTCAP_SAMPLE = re.compile(r"softcap|argmax|sampl|reduce|max_kernel", re.I)


def load_kernels(path):
    op = gzip.open if path.endswith(".gz") else open
    trace = json.load(op(path, "rt"))
    ev = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ev.sort(key=lambda e: e["ts"])
    return ev


def label_layer(kernels):
    """Label kernels within one decoder layer by name and position."""
    labels = []
    attn_seen = False
    gemm_after_attn = 0
    for k in kernels:
        n = k["name"]
        if ROUTING.search(n):
            lab = "router_topk"
        elif MOE.search(n):
            lab = "moe_experts"
        elif ATTN.search(n) or ATTN_STAGE2.search(n):
            lab = "attention"
            attn_seen = True
        elif NORM.search(n):
            lab = "norms"
        elif ACT.search(n):
            lab = "dense_mlp"
        elif ROPE.search(n):
            lab = "rope_kvstore"
        elif KVSTORE.search(n):
            lab = "rope_kvstore"
        elif GEMM.search(n):
            if not attn_seen:
                lab = "qkv_proj"
            else:
                gemm_after_attn += 1
                # o_proj, dense gate_up, dense down, router
                lab = {1: "o_proj", 2: "dense_mlp", 3: "dense_mlp", 4: "router_topk"}.get(gemm_after_attn, "gemm_other")
        else:
            lab = "other"
        labels.append(lab)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--dump-seq", type=int, default=0, help="print first N kernel names")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ev = load_kernels(args.trace)
    if args.dump_seq:
        for e in ev[: args.dump_seq]:
            print(f"{e['dur']:8.1f} us  {e['name'][:150]}")
        return

    total_busy = sum(e["dur"] for e in ev)
    span = ev[-1]["ts"] + ev[-1]["dur"] - ev[0]["ts"]
    by_name = collections.Counter()
    for e in ev:
        by_name[e["name"][:120]] += e["dur"]
    print(f"kernels={len(ev)} busy={total_busy / 1e3:.2f} ms span={span / 1e3:.2f} ms gpu_busy={total_busy / span:.1%}")
    for n, d in by_name.most_common(40):
        print(f"{d / 1e3:9.3f} ms {d / total_busy:6.1%}  {n}")
    if args.json_out:
        json.dump(dict(total_busy_us=total_busy, span_us=span, by_name=dict(by_name)), open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
