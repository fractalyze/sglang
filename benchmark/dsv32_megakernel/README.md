# DeepSeek V3.2 decode floor

How much a persistent per-layer decode megakernel could win on DeepSeek V3.2
AWQ, served with DP attention on one node of eight H100s, sized from bytes and
profiles alone. Two scripts answer it:

- `decode_floor.py` is the bytes model. Per rank and per decode step it adds
  up the HBM bytes the recipe must read, divides them by a measured
  bandwidth, and adds the measured `--enable-symm-mem` collectives. It
  prints three floors:
  - serial: every op at its floor, run one after another;
  - prefetch: each collective overlapped with an L2's worth of the next
    GEMM's weights;
  - overlap: every collective hidden.
- `trace_split.py` books every microsecond of measured decode steps to an op
  class or to idle (no kernel running), then sets each class beside its floor.
- `grid_barrier.py` times one grid barrier of a persistent kernel on the local
  GPU, the cost slice 1 pays per launch it removes.

## The recipe the model assumes

`--enable-dp-attention --dp 8`, with the MoE tensor-parallel over 8 ranks.
Expert parallelism and DeepEP do not run this AWQ checkpoint. Per rank and per
step that means:

| Weights | Read per rank |
|---|---|
| MLA and indexer projections | all (attention TP size 1) |
| `w_kc` / `w_vc` | all, bf16 (the loader dequantizes `kv_b_proj`) |
| dense MLP (first 3 layers), shared expert, each touched routed expert | 1/8 (TP over the intermediate dim) |
| router | all, bf16 (AWQ skips `mlp.gate.`) |
| LM head | 1/8 (vocab parallel) |

KV traffic comes from the rank's own requests: the MLA cache for at most
`index_topk` selected tokens, and the indexer-K cache (fp8 plus a scale) for
the whole context. The collectives are one all-gather and one reduce-scatter
per layer, plus the logits gather, timed at the global token count by
`benchmark/kernels/all_gather/benchmark_dp_attn_ag_rs.py`.

Experts touched per step default to uniform routing over the gathered batch.
Pass a measured count per concurrency to both scripts with
`--experts-touched CONC:N`. The SGLang expert-distribution recorder produces
one on an 8-GPU run.

## Running

Both scripts take the same measured inputs:

- `--hbm-gbps`: the best HBM read bandwidth any kernel reaches on the machine,
  in GB/s. The W4A16 MoE kernel benches report it per row (`gbps`).
- `--comm-jsonl`: the JSONL that
  `benchmark/kernels/all_gather/benchmark_dp_attn_ag_rs.py --config symm --out`
  writes on the same node.

```sh
# Floors at c128 / c256 / c512, beside the measured decode step (median ITL).
python benchmark/dsv32_megakernel/decode_floor.py \
  --hbm-gbps <GB/s> --comm-jsonl dp_attn_ag_rs.jsonl \
  --measured-ms 128:<ms> 256:<ms> 512:<ms>

# One grid barrier at one CTA per SM, on one GPU of the node.
python benchmark/dsv32_megakernel/grid_barrier.py

# Split a profiled run's decode steps (one *-DECODE.trace.json.gz per rank).
python benchmark/dsv32_megakernel/trace_split.py <trace dir> --concurrency 512 \
  --hbm-gbps <GB/s> --comm-jsonl dp_attn_ag_rs.jsonl --grid-barrier-us <us>
```

`--grid-barrier-us` takes the cheaper of `grid_barrier.py`'s two barriers at
the thread count the megakernel would run.

The default `--context-tokens 1536` is the mean KV length of a request while
it decodes in `python -m sglang.bench_serving --dataset-name random
--random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0`:
its input plus half its output. Profile the run with SGLang's `--profile` flow,
so each rank writes its `*-DECODE.trace.json.gz`. For an EAGLE baseline, pass
`--tokens-per-request <draft tokens + 1>` to both scripts.

## Reading the split

`trace_split.py` divides the measured step into four parts:

- **Floor:** HBM bytes per class at the best measured bandwidth, plus the
  symm-mem collectives.
- **Kernel inefficiency:** class time over that floor. Per-kernel work (the
  MoE and dense W4A16 kernels) removes it, not a megakernel.
- **Collective excess:** NCCL time over the symm-mem latency. The stock
  `--enable-symm-mem` flag removes it, so the baseline subtracts it first.
- **Serialization:** idle gaps between kernels, plus small ops and routing.
  These ops move only activations, so their floor is about zero.

It then prints a ceiling for each megakernel slice:

- **Slice 1, fused non-GEMM path:** the idle gaps, small ops and routing,
  all removed.
- **Slice 2, in-layer overlap:** each collective, and each layer's attention
  time above its floor, hidden behind weight prefetch. Prefetch is capped at
  one L2 of weights per window, since the next GEMM can only stage what fits
  on chip while it waits on its input.

Both ceilings are upper bounds. A persistent kernel pays a grid barrier
wherever the unfused path launches a kernel, so the idle bucket turns into
barrier waits instead of disappearing. The split therefore also reports slice 1
net of one `--grid-barrier-us` per kernel the step launches. Slice 2 stays
gross, because its overlap waits on flags, not grid barriers.
