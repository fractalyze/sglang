"""
Tests the Gemma-4 target's FP8 LM head copy (SGLANG_OPT_GEMMA4_FP8_LM_HEAD):
batches up to the tile's 48 rows get FP8 logits within E4M3's rounding bound
of the BF16 head, wider batches get the BF16 head's logits bit for bit, the
copy leaves the tied embedding intact, and LogitsProcessor takes the hook.
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
class TestGemma4Fp8LmHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.models.gemma4_mm import _Fp8TiedHead

        torch.manual_seed(0)
        cls.w = torch.randn(_VOCAB, _HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
        cls.head = _Fp8TiedHead(cls.w)

    def _default_logits(self, x):
        # LogitsProcessor's path for a tied BF16 head.
        return torch.matmul(x.to(self.w.dtype), self.w.T)

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
        self.assertEqual(self.head.weight.dtype, torch.float8_e4m3fn)
        self.assertIs(self.head.bf16_weight, self.w)

    def test_narrow_batches_get_fp8_logits_within_e4m3_bound(self):
        for m in (1, 6, 8, 32, _FP8_MAX_M):
            with self.subTest(m=m):
                x = torch.randn(m, _HIDDEN, dtype=torch.bfloat16, device="cuda")
                fp8 = self.head.quant_method.apply(self.head, x).float()
                bf16 = self._default_logits(x).float()
                self.assertEqual(fp8.shape, (m, _VOCAB))
                self.assertFalse(torch.equal(fp8, bf16))
                bound = _E4M3_REL * (x.float().abs() @ self.w.float().abs().t())
                excess = (fp8 - bf16).abs() - (bound + _bf16_ulp(bf16) + _bf16_ulp(fp8))
                self.assertLessEqual(excess.max().item(), 0.0)

    def test_wide_batches_keep_the_bf16_head_exactly(self):
        for m in (_FP8_MAX_M + 1, 192):
            with self.subTest(m=m):
                x = torch.randn(m, _HIDDEN, dtype=torch.bfloat16, device="cuda")
                out = self.head.quant_method.apply(self.head, x)
                self.assertTrue(torch.equal(out, self._default_logits(x)))

    def test_hook_adds_copy_and_keeps_tied_embedding(self):
        from sglang.srt.environ import envs
        from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration

        embed = torch.nn.Embedding(_VOCAB, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        embed.weight.data.copy_(self.w)
        model = types.SimpleNamespace(
            language_model=types.SimpleNamespace(embed_tokens=embed),
            lm_head_is_tied=True,
            fp8_lm_head=None,
        )
        Gemma4ForConditionalGeneration._add_fp8_lm_head(model)
        self.assertIsNotNone(model.fp8_lm_head)
        # The copy reads the embedding's own storage: no second BF16 table.
        self.assertEqual(
            model.fp8_lm_head.bf16_weight.data_ptr(), embed.weight.data_ptr()
        )
        self.assertEqual(embed.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(embed.weight.data, self.w))
        # An untied head has no single table to copy, so the switch leaves it alone.
        untied = types.SimpleNamespace(
            language_model=model.language_model, lm_head_is_tied=False, fp8_lm_head=None
        )
        Gemma4ForConditionalGeneration._add_fp8_lm_head(untied)
        self.assertIsNone(untied.fp8_lm_head)
        self.assertFalse(envs.SGLANG_OPT_GEMMA4_FP8_LM_HEAD.get())

    def test_model_loader_postprocess_accepts_the_head(self):
        """The loader's post-load pass reaches every quant_method; the copy must survive it unchanged."""
        from sglang.srt.model_loader.loader import DefaultModelLoader

        weight = self.head.weight.clone()
        DefaultModelLoader.postprocess_weights(
            torch.nn.ModuleList([self.head]), torch.device("cuda")
        )
        self.assertTrue(
            torch.equal(self.head.weight.view(torch.uint8), weight.view(torch.uint8))
        )

    def test_logits_processor_takes_the_quant_hook(self):
        from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

        self.assertTrue(
            should_apply_lm_head_quant_method(self.head, self.head.quant_method)
        )


@unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA GPU")
class TestGemma4Fp8VocabTable(unittest.TestCase):
    """SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE: one FP8 table replaces the tied BF16
    embedding and head; both sides stay within E4M3's rounding bound."""

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
        bound = _E4M3_REL * (self.w[ids].float().abs() * self._EMBED_SCALE)
        excess = (got - ref).abs() - (bound + _bf16_ulp(ref) + _bf16_ulp(got))
        self.assertLessEqual(excess.max().item(), 0.0)
        self.assertTrue(torch.equal(got[2], got[5]))

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
