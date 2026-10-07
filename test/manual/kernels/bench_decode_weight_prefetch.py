"""Decode weight prefetch on one H100: what staging the next layer's
pre-attention weights in L2 during a collective saves the GEMMs that read them.

Each row flushes L2, holds HBM idle for `window_us` (a sleep standing in for
the reduce-scatter, which moves activations over NVLink), then times the chain
of reads the next DeepSeek V3.2 layer issues before its attention core at
M = 64 tokens per rank (c512 over DP8): the W4A16 projections and the bf16
w_kc absorb. With prefetch, a side stream issues the L2 prefetch as the window
opens. `saved_us` is the chain's cold time minus its prefetched time; the
slice-2 model caps it at one L2 of weights per window. Each time is the
median of `--repeats` runs.

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


def _time_chain(layer, flush, ranges, side, window_cycles, evict_last) -> float:
    main = torch.cuda.current_stream()
    chain_start, chain_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    flush.zero_()
    if ranges is not None:
        side.wait_stream(main)
        with torch.cuda.stream(side):
            l2_prefetch(ranges, evict_last=evict_last)
    if window_cycles:
        torch.cuda._sleep(window_cycles)
    chain_start.record()
    layer.run()
    chain_end.record()
    main.wait_stream(side)
    torch.cuda.synchronize()
    return chain_start.elapsed_time(chain_end) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--windows-us", type=float, nargs="+", default=[0, 10, 17, 25, 45, 100]
    )
    parser.add_argument("--budgets-mb", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--out")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        sys.exit("needs an SM90 GPU")

    layer = NextLayer()
    flush = torch.empty(_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    side = torch.cuda.Stream()
    cycles_per_us = _sleep_cycles_per_us()
    total_mb = sum(t.nbytes for t in layer.weights()) / 2**20
    print(f"pre-core weights {total_mb:.1f} MiB, M = {M}")
    for _ in range(3):
        layer.run()

    rows = []
    for window_us in args.windows_us:
        cycles = int(window_us * cycles_per_us)

        def median(ranges, evict_last=False):
            return statistics.median(
                _time_chain(layer, flush, ranges, side, cycles, evict_last)
                for _ in range(args.repeats)
            )

        cold_us = median(None)
        for budget_mb in args.budgets_mb:
            ranges = plan_l2_prefetch(layer.weights(), budget_bytes=budget_mb << 20)
            for evict_last in (False, True):
                warm_us = median(ranges, evict_last)
                row = dict(
                    window_us=window_us,
                    budget_mb=budget_mb,
                    evict_last=evict_last,
                    cold_us=round(cold_us, 2),
                    prefetched_us=round(warm_us, 2),
                    saved_us=round(cold_us - warm_us, 2),
                )
                rows.append(row)
                print(json.dumps(row))
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
