"""Unit tests for the DeepSeek V3.2 decode floor and trace split, CPU only."""

import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase

BENCHMARK_DIR = Path(__file__).parents[4] / "benchmark" / "dsv32_megakernel"
sys.path.insert(0, str(BENCHMARK_DIR))
import decode_floor  # noqa: E402
import trace_split  # noqa: E402

# `metadata.total_size` of QuantTrio/DeepSeek-V3.2-AWQ's
# model.safetensors.index.json, revision 340023cb.
_CHECKPOINT_TOTAL_BYTES = 361_909_457_408


def _checkpoint_bytes(m: decode_floor.ModelShape) -> int:
    """Every weight the checkpoint stores, built from the floor's byte helpers.

    It differs from what a rank reads: kv_b_proj is stored AWQ, and the MTP
    layer carries its own embedding, head and eh_proj.
    """
    h = m.hidden_size
    qk_head_dim = m.qk_nope_head_dim + m.qk_rope_head_dim
    kv_b_out = m.num_heads * (m.qk_nope_head_dim + m.v_head_dim)
    attention = (
        decode_floor.awq_bytes(h, m.q_lora_rank)
        + decode_floor.awq_bytes(h, m.kv_cache_dim)
        + decode_floor.awq_bytes(m.q_lora_rank, m.num_heads * qk_head_dim)
        + decode_floor.awq_bytes(m.kv_lora_rank, kv_b_out)
        + decode_floor.awq_bytes(m.num_heads * m.v_head_dim, h)
        + decode_floor.awq_bytes(m.q_lora_rank, m.index_n_heads * m.index_head_dim)
        + decode_floor.awq_bytes(h, m.index_head_dim)
        + decode_floor.bf16_bytes(h, m.index_n_heads)
    )
    expert = decode_floor.swiglu_mlp_awq_bytes(h, m.moe_intermediate)
    moe_layer = (
        attention
        + (m.n_routed_experts + m.n_shared_experts) * expert
        + decode_floor.bf16_bytes(h, m.n_routed_experts)
    )
    dense_layer = attention + decode_floor.swiglu_mlp_awq_bytes(h, m.dense_intermediate)
    embed_and_head = 2 * decode_floor.bf16_bytes(h, m.vocab_size)
    mtp_layer = moe_layer + embed_and_head + decode_floor.bf16_bytes(2 * h, h)
    return (
        m.first_k_dense * dense_layer
        + m.num_moe_layers * moe_layer
        + embed_and_head
        + mtp_layer
    )


class TestStepBytes(CustomTestCase):
    def test_shape_reproduces_checkpoint_size(self):
        # Norms and biases are the only tensors the helpers leave out.
        total = _checkpoint_bytes(decode_floor.ModelShape())
        self.assertAlmostEqual(total / _CHECKPOINT_TOTAL_BYTES, 1.0, delta=1e-4)

    def test_from_hf_config_matches_builtin_shape(self):
        config = {
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "first_k_dense_replace": 3,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "num_attention_heads": 128,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 2048,
            "vocab_size": 129280,
        }
        self.assertEqual(
            decode_floor.ModelShape.from_hf_config(config), decode_floor.ModelShape()
        )

    def test_experts_touched_bounds(self):
        m = decode_floor.ModelShape()
        self.assertAlmostEqual(decode_floor.expected_experts_touched(m, 1), 8.0)
        self.assertAlmostEqual(decode_floor.expected_experts_touched(m, 4096), 256.0)

    def test_measured_experts_touched_scales_routed_bytes(self):
        m = decode_floor.ModelShape()
        full = decode_floor.step_bytes(
            m, decode_floor.DecodePoint(concurrency=512, experts_touched=256)
        )
        half = decode_floor.step_bytes(
            m, decode_floor.DecodePoint(concurrency=512, experts_touched=128)
        )
        self.assertEqual(full.routed_experts, 2 * half.routed_experts)
        self.assertEqual(full.dense_gemm, half.dense_gemm)

    def test_mla_reads_capped_at_index_topk_but_indexer_reads_all(self):
        m = decode_floor.ModelShape()
        short = decode_floor.step_bytes(
            m, decode_floor.DecodePoint(concurrency=8, context_tokens=2048)
        )
        long = decode_floor.step_bytes(
            m, decode_floor.DecodePoint(concurrency=8, context_tokens=4096)
        )
        self.assertEqual(short.mla_kv, long.mla_kv)
        self.assertEqual(2 * short.indexer_k, long.indexer_k)


# Synthetic collective latencies and bandwidth; the tests check structure, not H100.
_COMM = decode_floor.CommTable(
    all_gather_us={128: 10.0, 512: 20.0}, reduce_scatter_us={128: 12.0, 512: 24.0}
)
_HBM_GBPS = 2000.0


class TestFloor(CustomTestCase):
    def test_interpolate_is_exact_at_measured_sizes_and_clamped_outside(self):
        table = {128: 10.0, 512: 30.0}
        self.assertEqual(decode_floor.interpolate_us(table, 128), 10.0)
        self.assertEqual(decode_floor.interpolate_us(table, 320), 20.0)
        self.assertEqual(decode_floor.interpolate_us(table, 64), 10.0)
        self.assertEqual(decode_floor.interpolate_us(table, 1024), 30.0)

    def test_load_comm_table_keeps_only_the_requested_nccl_setup(self):
        def row(config, proto, op, impl, tokens, us):
            return {
                "config": config,
                "nccl_proto": proto,
                "op": op,
                "impl": impl,
                "global_tokens": tokens,
                "us": us,
            }

        rows = [
            row("symm", "auto", "all_gather", "nccl", 512, 25.0),
            row("symm", "auto", "reduce_scatter", "nccl", 512, 26.0),
            row("symm", "auto", "all_gather", "multimem", 512, 1.0),
            row("symm", "LL", "all_gather", "nccl", 512, 2.0),
            row("default", "auto", "all_gather", "nccl", 512, 3.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dp_attn_ag_rs.jsonl")
            with open(path, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            table = decode_floor.load_comm_table(path)
            with self.assertRaises(ValueError):
                decode_floor.load_comm_table(path, config="nvls")
        self.assertEqual(table.all_gather_us, {512: 25.0})
        self.assertEqual(table.reduce_scatter_us, {512: 26.0})

    def test_one_all_gather_and_reduce_scatter_per_layer_plus_logits(self):
        m = decode_floor.ModelShape()
        windows = decode_floor.comm_windows_us(
            m, decode_floor.DecodePoint(concurrency=512), _COMM
        )
        self.assertEqual(len(windows), 2 * m.num_layers + 1)
        self.assertEqual(windows.count(20.0), m.num_layers + 1)
        self.assertEqual(windows.count(24.0), m.num_layers)

    def test_floors_are_ordered(self):
        m = decode_floor.ModelShape()
        for concurrency in (128, 256, 512):
            f = decode_floor.decode_floor(
                m,
                decode_floor.DecodePoint(concurrency=concurrency),
                hbm_gbps=_HBM_GBPS,
                comm=_COMM,
            )
            self.assertAlmostEqual(f.serial_ms, f.hbm_ms + f.comm_ms)
            self.assertEqual(f.overlap_ms, max(f.hbm_ms, f.comm_ms))
            # Equal when every collective fits the prefetch window.
            self.assertLessEqual(f.overlap_ms, f.prefetch_ms + 1e-9)
            self.assertLessEqual(f.prefetch_ms, f.serial_ms + 1e-9)

    def test_no_prefetch_capacity_leaves_the_serial_floor(self):
        m = decode_floor.ModelShape()
        f = decode_floor.decode_floor(
            m,
            decode_floor.DecodePoint(concurrency=512),
            hbm_gbps=_HBM_GBPS,
            comm=_COMM,
            prefetch_bytes=0,
        )
        self.assertAlmostEqual(f.prefetch_ms, f.serial_ms)


def _kernel(start, end, op_class):
    return trace_split.Kernel(start_us=start, end_us=end, op_class=op_class)


class TestTraceSplit(CustomTestCase):
    def test_classify_decode_kernels(self):
        cases = {
            "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)": "comm",
            "void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 1>": "moe",
            "void sglang::device::marlin::Marlin<__nv_bfloat16, 1, 256>": "dense",
            "nvjet_sm90_tss_64x32_64x16_1x2_h_bz_splitK_TNT": "dense",
            "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::"
            "FlashAttnFwdSm90<>>>": "attention",
            "void deep_gemm::sm90_fp8_paged_mqa_logits<1u, 64u>": "attention",
            "kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnorm": "small",
            "void tensorrt_llm::kernels::deepseek_v3_topk_kernel<float>": "routing",
            "void sglang::topk_main_kernel<true, 3>(sglang::TopKPagedParams)": (
                "attention"
            ),
            "void sglang::act_and_mul_kernel<__nv_bfloat16>": "small",
            "void w4a16_moe_sm90_kernel<Trait<32>>(Params)": "moe",
            "void w4a16_sm90_kernel<Trait<2112, 7168>, true>(CUtensorMap, Params)": (
                "dense"
            ),
        }
        for name, expected in cases.items():
            self.assertEqual(trace_split.classify(name), expected, name)

    def test_split_books_overlap_evenly_and_gaps_to_idle(self):
        buckets = trace_split.split_step(
            [
                _kernel(0, 10, "moe"),
                _kernel(5, 10, "dense"),  # overlaps the second half of moe
                _kernel(12, 20, "comm"),  # an idle gap before it
            ]
        )
        self.assertAlmostEqual(buckets["moe"], 7.5)
        self.assertAlmostEqual(buckets["dense"], 2.5)
        self.assertAlmostEqual(buckets["comm"], 8.0)
        self.assertAlmostEqual(buckets["idle"], 2.0)
        self.assertAlmostEqual(sum(buckets.values()), 20.0)

    def test_load_steps_uses_outer_step_ranges(self):
        events = [
            # The profiler repeats a step's range per stream, nested inside it.
            {
                "cat": "gpu_user_annotation",
                "name": "step[DECODE bs=64]",
                "ts": 0,
                "dur": 100,
            },
            {
                "cat": "gpu_user_annotation",
                "name": "step[DECODE bs=64]",
                "ts": 10,
                "dur": 20,
            },
            {
                "cat": "gpu_user_annotation",
                "name": "step[DECODE bs=64]",
                "ts": 200,
                "dur": 50,
            },
            {"cat": "kernel", "name": "nccl_AllGather", "ts": 0, "dur": 40},
            {"cat": "kernel", "name": "marlin_moe", "ts": 50, "dur": 50},
            {"cat": "kernel", "name": "rmsnorm", "ts": 200, "dur": 50},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "0-TP-0-DP-0-DECODE.trace.json.gz")
            with gzip.open(path, "wt") as f:
                json.dump({"traceEvents": events}, f)
            steps = trace_split.load_steps(path)
        self.assertEqual(len(steps), 2)
        self.assertAlmostEqual(steps[0].class_us["comm"], 40)
        self.assertAlmostEqual(steps[0].class_us["idle"], 10)
        self.assertAlmostEqual(steps[0].class_us["moe"], 50)
        self.assertAlmostEqual(steps[1].class_us["small"], 50)
        self.assertEqual([s.launches for s in steps], [2, 1])
        self.assertAlmostEqual(trace_split.mean_launches(steps), 1.5)

    _MEASURED = {
        "comm": 6.0,
        "small": 3.0,
        "moe": 32.0,
        "routing": 0.5,
        "attention": 5.0,
        "dense": 13.0,
        "idle": 1.5,
    }

    def _split(self, measured, prefetch_bytes=decode_floor.H100_L2_BYTES, **kwargs):
        return trace_split.split_gap(
            measured,
            shape=decode_floor.ModelShape(),
            point=decode_floor.DecodePoint(concurrency=512),
            hbm_gbps=_HBM_GBPS,
            comm=_COMM,
            prefetch_bytes=prefetch_bytes,
            **kwargs,
        )

    def test_split_gap_slices(self):
        m = decode_floor.ModelShape()
        point = decode_floor.DecodePoint(concurrency=512)
        gap = self._split(self._MEASURED)
        comm_floor = sum(decode_floor.comm_windows_us(m, point, _COMM)) / 1e3
        self.assertAlmostEqual(gap.measured_ms, 61.0)
        self.assertAlmostEqual(gap.comm_symm_saving_ms, 6.0 - comm_floor)
        self.assertAlmostEqual(gap.baseline_ms, 61.0 - (6.0 - comm_floor))
        self.assertAlmostEqual(gap.slice1_ms, 1.5 + 3.0 + 0.5)
        self.assertAlmostEqual(gap.slice1_net_ms, gap.slice1_ms)
        self.assertGreater(gap.slice2_ms, 0.0)
        self.assertLessEqual(
            gap.slice2_ms, comm_floor + gap.inefficiency_ms["attention"]
        )

    def test_slice1_net_pays_one_barrier_per_launch(self):
        gap = self._split(self._MEASURED, launches=1000, grid_barrier_us=2.0)
        self.assertAlmostEqual(gap.barrier_ms, 2.0)
        self.assertAlmostEqual(gap.slice1_net_ms, 5.0 - 2.0)
        # Slice 2 hides work behind prefetch with flags, not grid barriers.
        self.assertAlmostEqual(gap.slice2_ms, self._split(self._MEASURED).slice2_ms)

    def test_slice1_net_is_zero_when_barriers_cost_more(self):
        gap = self._split(self._MEASURED, launches=1000, grid_barrier_us=10.0)
        self.assertAlmostEqual(gap.slice1_net_ms, 0.0)

    def test_attention_under_its_floor_hides_nothing(self):
        fast_attention = dict(self._MEASURED, attention=0.0)
        gap = self._split(fast_attention)
        self.assertLess(gap.inefficiency_ms["attention"], 0.0)
        comm_only = self._split(
            dict(fast_attention, attention=gap.floor_ms["attention"])
        )
        self.assertAlmostEqual(gap.slice2_ms, comm_only.slice2_ms)


if __name__ == "__main__":
    unittest.main()
