import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_decode_steps as pds  # noqa: E402

_DIV = ("void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, "
        "c10::BFloat16, at::native::binary_internal::DivFunctor<float> >, std::array<char*, 3ul> >(int, ...)")


def _k(name, ts, dur, corr, grid=(1, 1, 1)):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur,
            "args": {"correlation": corr, "grid": list(grid)}}


def _launch(corr, ts, name="cudaGraphLaunch"):
    return {"ph": "X", "cat": "cuda_runtime", "name": name, "ts": ts, "dur": 1, "args": {"correlation": corr}}


def _decode_step(corr, t0, bs, n_glue):
    """One graph replay: n_glue 2-us glue kernels, then the stage-1/stage-2 decode attention."""
    ks = [_k(_DIV if i % 2 else "_gemma_qkv_rmsnorm_kernel", t0 + 3 * i, 2, corr) for i in range(n_glue)]
    t = t0 + 3 * n_glue
    ks += [_k("_fwd_grouped_kernel_stage1", t, 5, corr, grid=(bs, 2, 8)), _k("_fwd_kernel_stage2", t + 5, 1, corr)]
    return [_launch(corr, t0 - 1)] + ks


class ShortNameTest(absltest.TestCase):
    def test_strips_templates_and_keeps_the_aten_op(self):
        self.assertEqual(pds.short_name("_gemma_qkv_rmsnorm_kernel"), "_gemma_qkv_rmsnorm_kernel")
        self.assertEqual(pds.short_name("void flashinfer::norm::RMSNormKernel<8u, float>(float*, ...)"),
                         "flashinfer::norm::RMSNormKernel")
        self.assertEqual(pds.short_name(_DIV), "at::native::vectorized_elementwise_kernel[DivFunctor]")


class SummarizeTest(absltest.TestCase):
    def test_one_graph_launch_is_one_step(self):
        ev = _decode_step(1, 10, bs=12, n_glue=4) + _decode_step(2, 100, bs=12, n_glue=4)
        # An extend forward outside any graph: eager launches with their own correlation ids.
        ev += [_launch(7, 199, "cudaLaunchKernel"), _k("_fwd_kernel", 200, 50, 7)]
        res = pds.summarize(ev)
        self.assertEqual(res["n_decode_steps"], 2)
        self.assertEqual(res["kernels_per_step"]["median"], 6)
        self.assertEqual(res["by_batch_size"]["12"]["steps"], 2)
        self.assertEqual(res["by_batch_size"]["12"]["span_us_median"], 3 * 4 + 6)
        self.assertEqual(res["eager_kernels"], 1)
        self.assertEqual(res["eager_busy_us"], 50)
        self.assertEqual(res["window_span_us"], 250 - 10)
        self.assertAlmostEqual(res["decode_share_of_span"], 2 * 18 / 240)
        self.assertEqual(res["names_per_step"]["names"]["_gemma_qkv_rmsnorm_kernel"], 2)
        self.assertEqual(res["names_per_step"]["names"]["at::native::vectorized_elementwise_kernel[DivFunctor]"], 2)

    def test_steps_group_by_captured_batch_size(self):
        ev = _decode_step(1, 0, bs=8, n_glue=2) + _decode_step(2, 50, bs=24, n_glue=6) + _decode_step(3, 120, bs=24,
                                                                                                    n_glue=6)
        res = pds.summarize(ev)
        self.assertEqual(sorted(res["by_batch_size"]), ["24", "8"])
        self.assertEqual(res["names_per_step"]["bs"], 24)
        self.assertAlmostEqual(res["mean_bs"], (8 + 24 + 24) / 3)


class ParsePointsTest(absltest.TestCase):
    def test_parses_load_and_concurrency(self):
        self.assertEqual(pds.parse_points("inflight:12,pthink30:72"), [("inflight", 12), ("pthink30", 72)])


if __name__ == "__main__":
    absltest.main()
