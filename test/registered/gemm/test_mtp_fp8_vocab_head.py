"""
Tests the Gemma-4 MTP assistant's FP8 vocab head (SGLANG_OPT_MTP_FP8_LM_HEAD):
the FP8 logits stay within E4M3's rounding bound of the BF16 head's, keep a
clearly separated top-k, and reach LogitsProcessor through its quant hook.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

_VOCAB, _HIDDEN = 262144, 1024
# E4M3 keeps 3 explicit mantissa bits, so per-row rounding errs by at most 2^-4 relative.
_E4M3_REL = 2.0**-4


def _bf16_ulp(t: torch.Tensor) -> torch.Tensor:
    mag = t.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


@unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA GPU")
class TestMtpFp8VocabHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.w = torch.randn(_VOCAB, _HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02

    def _head(self):
        from sglang.srt.models.gemma4_mtp import _Fp8VocabHead

        return _Fp8VocabHead(self.w)

    def test_quantization_matches_one_shot_per_row(self):
        from sglang.kernels.ops.gemm.triton_small_m_bf16_gemm import (
            quantize_fp8_weight_per_channel,
        )

        head = self._head()
        w8, scale = quantize_fp8_weight_per_channel(self.w[:20000])
        self.assertTrue(
            torch.equal(head.weight[:20000].view(torch.uint8), w8.view(torch.uint8))
        )
        self.assertTrue(torch.equal(head.weight_scale[:20000], scale))
        self.assertEqual(head.weight.dtype, torch.float8_e4m3fn)

    def test_logits_within_e4m3_bound_of_bf16_head(self):
        head = self._head()
        for m in (1, 8, 32, 48, 100):
            with self.subTest(m=m):
                x = torch.randn(m, _HIDDEN, dtype=torch.bfloat16, device="cuda")
                fp8 = head.quant_method.apply(head, x).float()
                bf16 = F.linear(x, self.w).float()
                self.assertEqual(fp8.shape, (m, _VOCAB))
                bound = _E4M3_REL * (x.float().abs() @ self.w.float().abs().t())
                excess = (fp8 - bf16).abs() - (bound + _bf16_ulp(bf16) + _bf16_ulp(fp8))
                self.assertLessEqual(excess.max().item(), 0.0)

    def test_separated_top_k_is_preserved(self):
        head = self._head()
        torch.manual_seed(1)
        k, rows = 3, 16
        ids = torch.randint(0, _VOCAB, (rows, k), device="cuda")
        # Row r scores ids[r] at about 12, 8 and 4 (w / |w|^2 has unit dot with w);
        # every other logit is a dot with a random row, about N(0, 0.46^2).
        target_logits = torch.tensor([12.0, 8.0, 4.0], device="cuda")
        rows_w = self.w[ids].float()
        x = (
            target_logits[None, :, None] * rows_w / rows_w.pow(2).sum(-1, keepdim=True)
        ).sum(1)
        x = x.to(torch.bfloat16)
        bf16 = F.linear(x, self.w).float()
        fp8 = head.quant_method.apply(head, x).float()
        top_bf16 = bf16.topk(k + 1, dim=-1)
        # Only rows whose k-th / (k+1)-th BF16 gap beats twice the error bound must agree.
        bound = (
            _E4M3_REL * (x.float().abs() @ self.w.float().abs().t()).max(dim=-1).values
        )
        separated = (top_bf16.values[:, k - 1] - top_bf16.values[:, k]) > 2 * bound
        self.assertGreater(separated.sum().item(), rows // 2)
        top_fp8 = fp8.topk(k, dim=-1).indices
        for r in separated.nonzero().flatten().tolist():
            self.assertEqual(
                set(top_fp8[r].tolist()), set(top_bf16.indices[r, :k].tolist())
            )
            self.assertEqual(top_fp8[r, 0].item(), top_bf16.indices[r, 0].item())

    def test_model_loader_postprocess_accepts_the_head(self):
        """The loader's post-load pass reaches every quant_method; the head must survive it unchanged."""
        from sglang.srt.model_loader.loader import DefaultModelLoader

        head = self._head()
        weight = head.weight.clone()
        DefaultModelLoader.postprocess_weights(
            torch.nn.ModuleList([head]), torch.device("cuda")
        )
        self.assertTrue(
            torch.equal(head.weight.view(torch.uint8), weight.view(torch.uint8))
        )

    def test_logits_processor_takes_the_quant_hook(self):
        from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

        head = self._head()
        self.assertTrue(should_apply_lm_head_quant_method(head, head.quant_method))


if __name__ == "__main__":
    unittest.main()
