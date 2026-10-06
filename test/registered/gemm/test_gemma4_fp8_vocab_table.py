"""
Tests the Gemma-4 FP8 vocab table (SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE): one FP8
E4M3 table with per-row scales replaces the tied BF16 embedding and LM head;
the lookup and the head (row-chunked past the tile's 48 rows) both stay within
E4M3's rounding bound of the BF16 table (with an absolute floor for subnormals), the swap drops the BF16 table, it
refuses an untied head or an untuned shape, and LogitsProcessor takes the hook.
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_VOCAB, _HIDDEN = 262144, 2816
_FP8_MAX_M = 48
# E4M3 keeps 3 explicit mantissa bits, so per-row rounding errs by at most 2^-4 relative.
_E4M3_REL = 2.0**-4


def _bf16_ulp(t: torch.Tensor) -> torch.Tensor:
    mag = t.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


@unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA GPU")
class TestGemma4Fp8VocabTable(unittest.TestCase):
    _EMBED_SCALE = _HIDDEN**0.5

    @classmethod
    def setUpClass(cls):
        from sglang.srt.models.gemma4_causal import _Fp8VocabTable

        torch.manual_seed(0)
        cls.embed = torch.nn.Embedding(
            _VOCAB, _HIDDEN, device="cuda", dtype=torch.bfloat16
        )
        cls.embed.weight.data.normal_(0.0, 0.02)
        cls.w = cls.embed.weight.data
        cls.table = _Fp8VocabTable(cls.embed, cls._EMBED_SCALE)

    def test_chunked_quantization_matches_one_shot(self):
        from sglang.kernels.ops.gemm.triton_small_m_bf16_gemm import (
            quantize_fp8_weight_per_channel,
            quantize_fp8_weight_per_channel_chunked,
        )

        w = self.w[:20000]
        w8, scale = quantize_fp8_weight_per_channel(w)
        c8, cscale = quantize_fp8_weight_per_channel_chunked(w, chunk_rows=3000)
        self.assertTrue(torch.equal(c8.view(torch.uint8), w8.view(torch.uint8)))
        self.assertTrue(torch.equal(cscale, scale))

    def test_head_is_within_e4m3_bound_at_every_batch_width(self):
        # Widths past the tile's max M run in row chunks; the bound must hold across chunk edges.
        for m in (1, _FP8_MAX_M, _FP8_MAX_M + 1, 2 * _FP8_MAX_M + 5):
            with self.subTest(m=m):
                x = torch.randn(m, _HIDDEN, dtype=torch.bfloat16, device="cuda")
                fp8 = self.table.quant_method.apply(self.table, x).float()
                bf16 = torch.matmul(x, self.w.T).float()
                self.assertEqual(fp8.shape, (m, _VOCAB))
                bound = _E4M3_REL * (x.float().abs() @ self.w.float().abs().t())
                excess = (fp8 - bf16).abs() - (bound + _bf16_ulp(bf16) + _bf16_ulp(fp8))
                self.assertLessEqual(excess.max().item(), 0.0)

    def test_lookup_is_within_e4m3_bound_of_the_scaled_bf16_lookup(self):
        ids = torch.tensor([0, 1, 7, 4096, _VOCAB - 1, 7], device="cuda")
        out = self.table(ids)
        self.assertEqual(out.dtype, torch.bfloat16)
        got = out.float()
        ref = (self.embed(ids) * self._EMBED_SCALE).float()
        # E4M3 subnormals (entries far below the row max) err by up to half their 2^-9
        # step times the row scale: an absolute floor under the relative bound.
        row_scale = self.table.weight_scale[ids].unsqueeze(-1) * self._EMBED_SCALE
        bound = _E4M3_REL * (self.w[ids].float().abs() * self._EMBED_SCALE)
        bound = bound + 2.0**-10 * row_scale
        excess = (got - ref).abs() - (bound + _bf16_ulp(ref) + _bf16_ulp(got))
        self.assertLessEqual(excess.max().item(), 0.0)
        self.assertTrue(torch.equal(got[2], got[5]))

    def test_model_loader_postprocess_leaves_the_table_unchanged(self):
        """The loader's post-load pass reaches every quant_method; the table must survive it."""
        from sglang.srt.model_loader.loader import DefaultModelLoader

        weight = self.table.weight.clone()
        DefaultModelLoader.postprocess_weights(
            torch.nn.ModuleList([self.table]), torch.device("cuda")
        )
        self.assertTrue(
            torch.equal(self.table.weight.view(torch.uint8), weight.view(torch.uint8))
        )

    def test_swap_drops_the_bf16_table_and_refuses_an_untied_head(self):
        from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
        from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM

        embed = torch.nn.Embedding(16384, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        embed.embed_scale = self._EMBED_SCALE
        stub = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=embed), lm_head=embed
        )
        with self.assertRaises(ValueError):
            # 16384 x 2816 has no tuned FP8 vocab-head tile.
            Gemma4ForCausalLM._use_fp8_vocab_table(stub)

        full = torch.nn.Embedding(_VOCAB, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        full.embed_scale = self._EMBED_SCALE
        untied = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=full), lm_head=None
        )
        with self.assertRaises(ValueError):
            Gemma4ForCausalLM._use_fp8_vocab_table(untied)

        tied = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=full), lm_head=full
        )
        Gemma4ForCausalLM._use_fp8_vocab_table(tied)
        self.assertIs(tied.lm_head, tied.model.embed_tokens)
        self.assertEqual(tied.lm_head.weight.dtype, torch.float8_e4m3fn)
        tensors = [
            v for v in vars(tied.lm_head).values() if isinstance(v, torch.Tensor)
        ]
        self.assertEqual(
            sorted(str(t.dtype) for t in tensors),
            ["torch.float32", "torch.float8_e4m3fn"],
        )
        self.assertTrue(
            should_apply_lm_head_quant_method(tied.lm_head, tied.lm_head.quant_method)
        )


if __name__ == "__main__":
    unittest.main()
