# decode-mk megakernels for Qwen3.8-27B

decode-mk's hand-written CUDA megakernels for batch-1 decoding of Qwen3.8-27B
on one RTX 5090 (`sm_120a`). They run on the int4 checkpoint
`cyankiwi/Qwen3.8-27B-AWQ-INT4` (revision `6e134ba`):

- **Decode:** all 64 layers, the embedding and the LM head in one launch a
  token.
- **Prefill:** the prompt, 64 tokens a launch.
- **Speculative decoding:** a verify step that checks up to four tokens in one
  launch, and the checkpoint's MTP head drafting tokens for it.

SGLang serves the model on them behind three switches. All three are off by
default, and with every switch off SGLang builds and serves the stock
Qwen3.8 model unchanged.

## The switches

| Switch | What it does |
|---|---|
| `SGLANG_DECODE_MK_DECODE=1` | Builds `Qwen3_5DecodeMkForConditionalGeneration` (`sglang/srt/models/qwen3_5_decode_mk.py`) in place of the stock model. The kernels read the checkpoint themselves and hold one request's states. Each decode step is one launch, and the prompt runs one token a launch. |
| `SGLANG_DECODE_MK_PREFILL=1` | With DECODE: the prompt runs 64 tokens a launch on the prefill kernel. |
| `SGLANG_DECODE_MK_MTP=1` | With DECODE: greedy requests decode speculatively under `--speculative-algorithm DECODE_MK_MTP` (`sglang/srt/speculative/decode_mk_mtp_worker.py`). The MTP head drafts `--speculative-num-steps` tokens a cycle, 1 to 3 (default 1), and one verify launch checks them. Every emitted token is plain greedy decoding's. Requests that sample, use a grammar or ask for logprobs decode one token a step instead. |

## Batch 1 only

The kernels hold one request's KV caches and linear-attention states, and
the stock model is not built beside them: a second copy of the weights would
not fit on the one RTX 5090 they target. **A server that could batch therefore refuses to
start; it does not fall back to the stock path.** With DECODE on,
`sglang/srt/arg_groups/decode_mk_hook.py` requires:

- `--max-running-requests 1`: a second running request would need its own
  states.
- `--disable-radix-cache`: a cached prefix would skip prompt tokens the states
  are built from.
- `--disable-cuda-graph`: each step is already one launch.
- `--context-length N`: the KV caches are allocated for N positions at load.
- One GPU: tensor, pipeline and data parallel sizes of 1, and no
  disaggregation.
- With MTP: `--speculative-algorithm DECODE_MK_MTP` and
  `--disable-overlap-schedule`. Without MTP, no other speculative algorithm.

The model serves text only; media requests get a 400.

```bash
SGLANG_DECODE_MK_DECODE=1 SGLANG_DECODE_MK_PREFILL=1 SGLANG_DECODE_MK_MTP=1 \
python -m sglang.launch_server --model-path cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --revision 6e134bae811fb5adac50ee042ae5f029ac6779aa \
  --max-running-requests 1 --disable-radix-cache --disable-cuda-graph \
  --context-length 16384 \
  --speculative-algorithm DECODE_MK_MTP --disable-overlap-schedule
```

The first start JIT-compiles the kernels into `$SGLANG_CACHE_DIR/decode_mk/`.
`_ext.py` compiles them with the CUDA toolkit torch was built against: the one
at `CUDA_HOME`, else the `nvidia-cuda-nvcc` wheel beside torch.

## Updating the kernels

Everything here but `_ext.py` and this README is a copy of fractalyze/decode-mk
at the commit `VENDORED` names. Change it in decode-mk, then copy the new
commit in:

```bash
scripts/sync_decode_mk.py <decode-mk checkout> <commit>
```

The script rewrites the copy from that commit and the manifest beside it:

- The CUDA sources are copied unchanged.
- The Python modules are copied with their `s2mk` imports pointed here.
- `ops.cpp` is cut down to the bindings of the kernels copied here.

`test/registered/unit/models/test_qwen3_5_decode_mk.py` fails when a file
differs from the manifest, so a fix made here instead of in decode-mk does not
survive unnoticed.

A kernel change that keeps decode-mk's Python API is one sync and one PR. One
that changes the API also changes `qwen3_5_decode_mk.py`, the API's one
consumer here. If a new decode-mk binding wraps a kernel the copy leaves out,
the extension fails to load with an undefined symbol; list the binding in the
script's `_DROPPED_*` sets.

## Tests

- `test/registered/unit/models/test_qwen3_5_decode_mk.py` (CPU): the switches,
  the servers they refuse, the model they build, and the vendored copy.
- `test/registered/kernels/ops/decode_mk/` (RTX 5090): the kernels against
  PyTorch references, and the runner on random weights. Speculative replies
  must equal plain greedy decoding's token for token.
- `test/registered/e2e/models/test_qwen3_8_decode_mk.py` (RTX 5090 and the
  checkpoint at `SGLANG_DECODE_MK_QWEN38_CHECKPOINT`): each switch set's
  served replies against decode-mk's own API on the same weights, token for
  token.
