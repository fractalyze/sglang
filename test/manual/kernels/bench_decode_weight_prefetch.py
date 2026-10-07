"""Decode weight prefetch on one H100: what staging the next layer's
pre-attention weights in L2 during a collective saves the GEMMs that read them.

Each row flushes L2, holds HBM idle for `window_us` (a sleep standing in for
the reduce-scatter, which moves activations over NVLink) while a side stream
stages the weights, then runs the chain of reads the next DeepSeek V3.2 layer
issues before its attention core at M = 64 tokens per rank (c512 over DP8):
the W4A16 projections and the bf16 w_kc absorb. Stages: the L2 bulk prefetch
at a byte budget, and `touch`, a full-GPU read of every byte. Every sequence
is a CUDA graph, timed as the median replay, so host launch gaps stay out;
the chain's time is a replay minus the same replay without it. `staged_us`
also carries any staging that outlasts the window. `hot_us` is the chain with
nothing flushed, the floor a stage could reach.

    python test/manual/kernels/bench_decode_weight_prefetch.py --out results.jsonl
"""

import argparse
import json
import statistics
import sys

import torch

from sglang.kernels.ops.gemm.w4a16_sm90 import GROUP_SIZE, w4a16_sm90_gemm
from sglang.kernels.ops.memory.l2_prefetch import l2_prefetch, plan_l2_prefetch

M = 64
HIDDEN = 7168
Q_LORA = 1536
NUM_HEADS = 128
QK_NOPE = 128
KV_LORA = 512
# (name, K, N) in decode read order: fused q_a + kv_a, q_b, indexer wq_b, wk.
PROJECTIONS = [
    ("qkv_a", HIDDEN, 2112),
    ("q_b", Q_LORA, NUM_HEADS * 192),
    ("idx_wq_b", Q_LORA, 64 * 128),
    ("idx_wk", HIDDEN, 128),
]
# Several times the L2, so writing it evicts everything else.
_FLUSH_BYTES = 256 << 20


def _awq_weights(k: int, n: int):
    # Marlin-layout shapes; values do not change the GEMM's time.
    qweight = torch.randint(
        -(2**31), 2**31 - 1, (k // 16, n * 2), dtype=torch.int32, device="cuda"
    )
    scales = torch.rand(k // GROUP_SIZE, n, device="cuda").to(torch.bfloat16)
    qzeros = torch.randint(
        -(2**31), 2**31 - 1, (k // GROUP_SIZE, n // 8), dtype=torch.int32, device="cuda"
    )
    return qweight, scales, qzeros


class NextLayer:
    def __init__(self):
        self.projections = [(n, _awq_weights(k, n)) for _, k, n in PROJECTIONS]
        self.w_kc = torch.randn(
            NUM_HEADS, QK_NOPE, KV_LORA, device="cuda", dtype=torch.bfloat16
        )
        self.x_hidden = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
        self.x_lora = torch.randn(M, Q_LORA, device="cuda", dtype=torch.bfloat16)
        self.q_nope = torch.randn(
            NUM_HEADS, M, QK_NOPE, device="cuda", dtype=torch.bfloat16
        )

    def weights(self):
        return [t for _, ws in self.projections for t in ws] + [self.w_kc]

    def run(self):
        for (_, k, _), (n, (qweight, scales, qzeros)) in zip(
            PROJECTIONS, self.projections
        ):
            a = self.x_hidden if k == HIDDEN else self.x_lora
            w4a16_sm90_gemm(a, qweight, scales, qzeros, size_n=n)
        torch.bmm(self.q_nope, self.w_kc)


def _sleep_cycles_per_us() -> float:
    cycles = 10_000_000
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    torch.cuda.synchronize()
    return cycles / (start.elapsed_time(end) * 1e3)


def _touch(weights):
    # A full-GPU read of every byte: the load-based way to put weights in L2.
    for w in weights:
        w.view(-1).view(torch.uint8).max()


def _graph(fn) -> torch.cuda.CUDAGraph:
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _median_us(graph, repeats) -> float:
    times = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times)


def _step(layer, flush, side, cycles, stage, chain):
    """flush L2 (or not), open the window with `stage` on the side stream,
    hold HBM idle for `cycles`, join, then run the chain if asked."""
    main = torch.cuda.current_stream()
    if flush is not None:
        flush.zero_()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        stage()
    if cycles:
        torch.cuda._sleep(cycles)
    main.wait_stream(side)
    if chain:
        layer.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--windows-us", type=float, nargs="+", default=[0, 17, 25, 45, 100, 200]
    )
    parser.add_argument("--budgets-mb", type=int, nargs="+", default=[24, 48])
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--out")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        sys.exit("needs an SM90 GPU")

    layer = NextLayer()
    flush = torch.empty(_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    side = torch.cuda.Stream()
    cycles_per_us = _sleep_cycles_per_us()
    weights = layer.weights()
    total_mb = sum(t.nbytes for t in weights) / 2**20
    print(f"pre-core weights {total_mb:.1f} MiB, M = {M}")
    layer.run()
    _touch(weights)

    stages = {"none": lambda: None}
    for budget_mb in args.budgets_mb:
        ranges = plan_l2_prefetch(weights, budget_bytes=budget_mb << 20)
        stages[f"bulk{budget_mb}"] = lambda r=ranges: l2_prefetch(r)
        stages[f"bulk{budget_mb}_evict_last"] = lambda r=ranges: l2_prefetch(
            r, evict_last=True
        )
    stages["touch"] = lambda: _touch(weights)

    def chain_us(flushed, cycles, stage):
        f = flush if flushed else None
        with_chain = _graph(lambda: _step(layer, f, side, cycles, stage, True))
        without = _graph(lambda: _step(layer, f, side, cycles, stage, False))
        return _median_us(with_chain, args.repeats) - _median_us(without, args.repeats)

    hot_us = chain_us(False, 0, stages["none"])
    print(f"hot-L2 chain {hot_us:.2f} us")
    rows = []
    for window_us in args.windows_us:
        cycles = int(window_us * cycles_per_us)
        cold_us = chain_us(True, cycles, stages["none"])
        for name, stage in stages.items():
            if name == "none":
                continue
            # The window plus the chain, against the window plus the cold chain:
            # a stage that outlasts the window delays the chain by the excess.
            f = lambda: _step(layer, flush, side, cycles, stage, True)
            base = lambda: _step(layer, flush, side, cycles, stages["none"], False)
            staged_us = _median_us(_graph(f), args.repeats) - _median_us(
                _graph(base), args.repeats
            )
            row = dict(
                window_us=window_us,
                stage=name,
                hot_us=round(hot_us, 2),
                cold_us=round(cold_us, 2),
                staged_us=round(staged_us, 2),
                saved_us=round(cold_us - staged_us, 2),
            )
            rows.append(row)
            print(json.dumps(row))
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
