"""Per-rank decode-step floor for DeepSeek V3.2 AWQ under DP attention on H100.

The serving recipe is `--enable-dp-attention --dp 8` with tensor-parallel MoE
(expert parallelism and DeepEP do not run this AWQ checkpoint). Per rank and per
decode step, that recipe reads:

  * every attention and indexer weight in full (attn_tp_size == 1),
  * 1/8 of every dense-MLP, shared-expert and touched routed-expert weight
    (TP 8 over the intermediate dimension), and the router replicated,
  * 1/8 of the LM head (vocab parallel),
  * its own requests' MLA KV (at most `index_topk` tokens each, the DSA
    selection) and their full indexer-K cache,

and runs one all-gather and one reduce-scatter of the gathered hidden states per
layer, plus one more all-gather for the logits.

The floor is those bytes over a measured HBM bandwidth (`--hbm-gbps`, the best
any kernel reaches on the machine), with the collectives at the latency
`benchmark/kernels/all_gather/benchmark_dp_attn_ag_rs.py --out` recorded
(`--comm-jsonl`). Three floors bracket what a persistent per-layer kernel can
reach:

  serial    every op at its floor, run back to back as separate kernels;
  prefetch  each collective overlapped with streaming the next GEMM's weights,
            limited by how much fits on chip (L2) while the collective runs;
  overlap   collectives fully hidden behind weight streaming (an upper bound
            the per-layer dependency chain does not allow).

The defaults describe `python -m sglang.bench_serving --dataset-name random
--random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0` at
c128, c256 and c512: a request's mean KV length while it decodes is its input
plus half its output. `--json` prints the rows as JSON lines.
"""

import argparse
import bisect
import json
from typing import Dict, List, Optional, Sequence

import msgspec

# H100 SXM L2 capacity, the most weight data a prefetch can stage on chip
# while a collective runs.
H100_L2_BYTES = 50 * 1024 * 1024

AWQ_GROUP_SIZE = 128
BF16_BYTES = 2
# The indexer-K cache stores 128 fp8 values and one fp32 scale per token.
INDEXER_K_BYTES_PER_TOKEN = 128 + 4


class ModelShape(msgspec.Struct, frozen=True, kw_only=True):
    """The DeepSeek V3.2 config fields the floor depends on."""

    hidden_size: int = 7168
    num_layers: int = 61
    first_k_dense: int = 3
    dense_intermediate: int = 18432
    moe_intermediate: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    experts_per_token: int = 8
    num_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    vocab_size: int = 129280

    @classmethod
    def from_hf_config(cls, config: dict) -> "ModelShape":
        return cls(
            hidden_size=config["hidden_size"],
            num_layers=config["num_hidden_layers"],
            first_k_dense=config["first_k_dense_replace"],
            dense_intermediate=config["intermediate_size"],
            moe_intermediate=config["moe_intermediate_size"],
            n_routed_experts=config["n_routed_experts"],
            n_shared_experts=config["n_shared_experts"],
            experts_per_token=config["num_experts_per_tok"],
            num_heads=config["num_attention_heads"],
            q_lora_rank=config["q_lora_rank"],
            kv_lora_rank=config["kv_lora_rank"],
            qk_nope_head_dim=config["qk_nope_head_dim"],
            qk_rope_head_dim=config["qk_rope_head_dim"],
            v_head_dim=config["v_head_dim"],
            index_n_heads=config["index_n_heads"],
            index_head_dim=config["index_head_dim"],
            index_topk=config["index_topk"],
            vocab_size=config["vocab_size"],
        )

    @property
    def num_moe_layers(self) -> int:
        return self.num_layers - self.first_k_dense

    @property
    def kv_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


def awq_bytes(k: int, n: int) -> int:
    """Bytes of one AWQ W4 (group 128, zero point) weight of k inputs, n outputs.

    int4 weights, one bf16 scale and one int4 zero per group of 128 inputs.
    """
    groups = k // AWQ_GROUP_SIZE
    return k * n // 2 + groups * n * BF16_BYTES + groups * n // 2


def bf16_bytes(k: int, n: int) -> int:
    return k * n * BF16_BYTES


def swiglu_mlp_awq_bytes(hidden: int, intermediate: int) -> int:
    """gate, up and down projections of one SwiGLU MLP, all AWQ."""
    return 2 * awq_bytes(hidden, intermediate) + awq_bytes(intermediate, hidden)


def attention_weight_bytes(m: ModelShape) -> int:
    """MLA plus indexer weights of one layer, as one rank reads them.

    kv_b_proj is dequantized at load into the bf16 w_kc / w_vc the absorbed
    decode path multiplies with; everything else stays AWQ except the indexer's
    bf16 weights_proj.
    """
    qk_head_dim = m.qk_nope_head_dim + m.qk_rope_head_dim
    h = m.hidden_size
    mla = (
        awq_bytes(h, m.q_lora_rank + m.kv_cache_dim)  # fused q_a + kv_a
        + awq_bytes(m.q_lora_rank, m.num_heads * qk_head_dim)  # q_b
        + bf16_bytes(m.kv_lora_rank, m.num_heads * m.qk_nope_head_dim)  # w_kc
        + bf16_bytes(m.kv_lora_rank, m.num_heads * m.v_head_dim)  # w_vc
        + awq_bytes(m.num_heads * m.v_head_dim, h)  # o_proj
    )
    indexer = (
        awq_bytes(m.q_lora_rank, m.index_n_heads * m.index_head_dim)  # wq_b
        + awq_bytes(h, m.index_head_dim)  # wk
        + bf16_bytes(h, m.index_n_heads)  # weights_proj
    )
    return mla + indexer


def expected_experts_touched(m: ModelShape, global_tokens: int) -> float:
    """Distinct routed experts a step of `global_tokens` touches, uniform routing.

    Each token picks `experts_per_token` distinct experts. Skewed routing
    touches fewer; a measured count replaces this through `experts_touched`.
    """
    miss = 1.0 - m.experts_per_token / m.n_routed_experts
    return m.n_routed_experts * (1.0 - miss**global_tokens)


class DecodePoint(msgspec.Struct, frozen=True, kw_only=True):
    """One decode operating point of the DP-attention recipe."""

    concurrency: int  # requests across all DP ranks
    dp_size: int = 8
    moe_tp_size: int = 8
    context_tokens: int = 1536  # mean KV length per request during decode
    tokens_per_request: int = 1  # > 1 for speculative verify steps
    experts_touched: Optional[float] = None  # measured; None = uniform routing

    @property
    def requests_per_rank(self) -> float:
        return self.concurrency / self.dp_size

    @property
    def global_tokens(self) -> int:
        return self.concurrency * self.tokens_per_request

    def experts_touched_or_uniform(self, m: "ModelShape") -> float:
        if self.experts_touched is not None:
            return self.experts_touched
        return expected_experts_touched(m, self.global_tokens)


class StepBytes(msgspec.Struct, frozen=True, kw_only=True):
    """HBM bytes one rank reads in one decode step, by op class."""

    attention_weights: int
    dense_mlp: int
    shared_experts: int
    routed_experts: int
    router: int
    lm_head: int
    mla_kv: int
    indexer_k: int

    @property
    def dense_gemm(self) -> int:
        """Weights the dense (non-grouped) GEMMs stream."""
        return (
            self.attention_weights
            + self.dense_mlp
            + self.shared_experts
            + self.router
            + self.lm_head
        )

    @property
    def kv(self) -> int:
        return self.mla_kv + self.indexer_k

    @property
    def total(self) -> int:
        return self.dense_gemm + self.routed_experts + self.kv


def step_bytes(m: ModelShape, p: DecodePoint) -> StepBytes:
    touched = p.experts_touched_or_uniform(m)
    expert = swiglu_mlp_awq_bytes(m.hidden_size, m.moe_intermediate)
    attended = min(p.context_tokens, m.index_topk)
    return StepBytes(
        attention_weights=m.num_layers * attention_weight_bytes(m),
        dense_mlp=m.first_k_dense
        * swiglu_mlp_awq_bytes(m.hidden_size, m.dense_intermediate)
        // p.moe_tp_size,
        shared_experts=m.num_moe_layers * m.n_shared_experts * expert // p.moe_tp_size,
        routed_experts=round(m.num_moe_layers * touched * expert / p.moe_tp_size),
        router=m.num_moe_layers * bf16_bytes(m.hidden_size, m.n_routed_experts),
        lm_head=bf16_bytes(m.hidden_size, m.vocab_size) // p.moe_tp_size,
        mla_kv=round(
            m.num_layers * p.requests_per_rank * attended * m.kv_cache_dim * BF16_BYTES
        ),
        indexer_k=round(
            m.num_layers
            * p.requests_per_rank
            * p.context_tokens
            * INDEXER_K_BYTES_PER_TOKEN
        ),
    )


def interpolate_us(table: Dict[int, float], tokens: int) -> float:
    """Latency at `tokens`, linear between measured sizes, clamped outside."""
    sizes = sorted(table)
    if tokens <= sizes[0]:
        return table[sizes[0]]
    if tokens >= sizes[-1]:
        return table[sizes[-1]]
    hi = bisect.bisect_left(sizes, tokens)
    lo_t, hi_t = sizes[hi - 1], sizes[hi]
    frac = (tokens - lo_t) / (hi_t - lo_t)
    return table[lo_t] + frac * (table[hi_t] - table[lo_t])


class CommTable(msgspec.Struct, frozen=True, kw_only=True):
    """Measured all-gather and reduce-scatter latency (us) by global tokens."""

    all_gather_us: Dict[int, float]
    reduce_scatter_us: Dict[int, float]


def load_comm_table(
    path: str, *, config: str = "symm", nccl_proto: str = "auto"
) -> CommTable:
    """NCCL rows of one engine setup from benchmark_dp_attn_ag_rs.py's JSONL."""
    tables: Dict[str, Dict[int, float]] = {"all_gather": {}, "reduce_scatter": {}}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if (
                row["config"] == config
                and row["nccl_proto"] == nccl_proto
                and row["impl"] == "nccl"
            ):
                tables[row["op"]][row["global_tokens"]] = row["us"]
    if not tables["all_gather"] or not tables["reduce_scatter"]:
        raise ValueError(f"{path} has no {config}/{nccl_proto} NCCL rows")
    return CommTable(
        all_gather_us=tables["all_gather"], reduce_scatter_us=tables["reduce_scatter"]
    )


def comm_windows_us(m: ModelShape, p: DecodePoint, comm: CommTable) -> List[float]:
    """Every collective of one decode step.

    One all-gather and one reduce-scatter per layer, plus the logits gather.
    """
    ag = interpolate_us(comm.all_gather_us, p.global_tokens)
    rs = interpolate_us(comm.reduce_scatter_us, p.global_tokens)
    return [ag, rs] * m.num_layers + [ag]


def hidden_behind_prefetch_us(
    windows_us: Sequence[float], *, prefetch_bytes: int, hbm_gbps: float
) -> float:
    """Time of `windows_us` the next GEMM's weight prefetch can fill.

    While the GEMM waits on its input, it can only stage what fits on chip.
    """
    window_cap_us = prefetch_bytes / (hbm_gbps * 1e3)
    return sum(min(w, window_cap_us) for w in windows_us)


class Floor(msgspec.Struct, frozen=True, kw_only=True):
    concurrency: int
    hbm_bytes: int
    hbm_ms: float
    comm_ms: float
    serial_ms: float
    prefetch_ms: float
    overlap_ms: float
    experts_touched: float


def decode_floor(
    m: ModelShape,
    p: DecodePoint,
    *,
    hbm_gbps: float,
    comm: CommTable,
    prefetch_bytes: int = H100_L2_BYTES,
) -> Floor:
    nbytes = step_bytes(m, p)
    hbm_ms = nbytes.total / (hbm_gbps * 1e6)
    windows = comm_windows_us(m, p, comm)
    comm_ms = sum(windows) / 1e3
    hidden_ms = (
        hidden_behind_prefetch_us(
            windows, prefetch_bytes=prefetch_bytes, hbm_gbps=hbm_gbps
        )
        / 1e3
    )
    return Floor(
        concurrency=p.concurrency,
        hbm_bytes=nbytes.total,
        hbm_ms=hbm_ms,
        comm_ms=comm_ms,
        serial_ms=hbm_ms + comm_ms,
        prefetch_ms=hbm_ms + comm_ms - hidden_ms,
        overlap_ms=max(hbm_ms, comm_ms),
        experts_touched=p.experts_touched_or_uniform(m),
    )


def _per_concurrency(items: Sequence[str]) -> Dict[int, float]:
    """Parse CONC:VALUE arguments."""
    out = {}
    for item in items:
        conc, value = item.split(":")
        out[int(conc)] = float(value)
    return out


def add_floor_args(parser: argparse.ArgumentParser) -> None:
    """The inputs both scripts take, so one run's floors agree between them."""
    parser.add_argument(
        "--hbm-gbps",
        type=float,
        required=True,
        help="Best HBM read bandwidth measured on the machine, in GB/s.",
    )
    parser.add_argument(
        "--comm-jsonl",
        required=True,
        help="JSONL from benchmark/kernels/all_gather/benchmark_dp_attn_ag_rs.py "
        "--out; its --config symm rows are used.",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=1536,
        help="Mean KV length per request during decode.",
    )
    parser.add_argument(
        "--tokens-per-request",
        type=int,
        default=1,
        help="Tokens each request feeds the MoE per step (draft tokens + 1 for "
        "an EAGLE verify step).",
    )
    parser.add_argument(
        "--experts-touched",
        nargs="*",
        default=[],
        metavar="CONC:N",
        help="Measured distinct routed experts per MoE layer per step; "
        "uniform routing where absent.",
    )
    parser.add_argument(
        "--hf-config",
        help="config.json of the checkpoint; the built-in DeepSeek V3.2 shape "
        "otherwise.",
    )


def shape_from_args(args: argparse.Namespace) -> ModelShape:
    if args.hf_config is None:
        return ModelShape()
    with open(args.hf_config) as f:
        return ModelShape.from_hf_config(json.load(f))


def point_from_args(args: argparse.Namespace, concurrency: int) -> DecodePoint:
    return DecodePoint(
        concurrency=concurrency,
        context_tokens=args.context_tokens,
        tokens_per_request=args.tokens_per_request,
        experts_touched=_per_concurrency(args.experts_touched).get(concurrency),
    )


def _report(floors: List[Floor], measured: Dict[int, float]) -> None:
    print(
        f"{'conc':>5} {'experts':>8} {'GB/rank':>8} {'hbm ms':>7} {'comm ms':>8} "
        f"{'serial':>7} {'prefetch':>8} {'overlap':>8} {'measured':>9}"
    )
    for f in floors:
        meas = measured.get(f.concurrency)
        meas_s = f"{meas:9.1f}" if meas is not None else f"{'-':>9}"
        print(
            f"{f.concurrency:5d} {f.experts_touched:8.1f} {f.hbm_bytes / 1e9:8.2f} "
            f"{f.hbm_ms:7.2f} {f.comm_ms:8.2f} {f.serial_ms:7.2f} "
            f"{f.prefetch_ms:8.2f} {f.overlap_ms:8.2f} {meas_s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument(
        "--measured-ms",
        nargs="*",
        default=[],
        metavar="CONC:MS",
        help="Measured decode step (median ITL) to print beside the floor.",
    )
    parser.add_argument("--json", action="store_true")
    add_floor_args(parser)
    args = parser.parse_args()

    shape = shape_from_args(args)
    comm = load_comm_table(args.comm_jsonl)
    floors = [
        decode_floor(shape, point_from_args(args, c), hbm_gbps=args.hbm_gbps, comm=comm)
        for c in args.concurrency
    ]
    if args.json:
        for f in floors:
            print(msgspec.json.encode(f).decode())
    else:
        _report(floors, _per_concurrency(args.measured_ms))


if __name__ == "__main__":
    main()
