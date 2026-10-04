"""CPU tests for make_fp8_ckpt (run: python experiments/g4poc/baseline/test_make_fp8_ckpt.py)."""

import os
import sys

import torch
from absl.testing import absltest

sys.path.insert(0, os.path.dirname(__file__))
import make_fp8_ckpt as m  # noqa: E402


class QuantizePerChannelTest(absltest.TestCase):
    def test_roundtrip_error_and_layout(self):
        torch.manual_seed(0)
        w = (torch.randn(64, 256) * torch.logspace(-3, 1, 64)[:, None]).to(torch.bfloat16)
        q, s = m.quantize_per_channel(w)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(s.shape), (64, 1))
        self.assertEqual(s.dtype, torch.bfloat16)
        deq = q.float() * s.float()
        rel = (deq - w.float()).abs().amax(dim=1) / w.float().abs().amax(dim=1)
        # Bound: half an E4M3 step in the top binade (2^-4) plus the BF16 scale rounding.
        self.assertLess(rel.max().item(), 0.07)
        self.assertGreater(q.float().abs().amax(dim=1).min().item(), 400)

    def test_zero_row_does_not_nan(self):
        q, s = m.quantize_per_channel(torch.zeros(2, 8, dtype=torch.bfloat16))
        self.assertFalse(torch.isnan(q.float()).any())
        self.assertFalse(torch.isnan(s.float()).any())


class ConvertTensorTest(absltest.TestCase):
    def test_gate_up_split_order(self):
        # HF Gemma4: gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)
        e, inter, hidden = 2, 4, 8
        t = torch.zeros(e, 2 * inter, hidden, dtype=torch.bfloat16)
        t[:, :inter] = 1.0
        t[:, inter:] = 2.0
        out = dict(m.convert_tensor("model.language_model.layers.3.experts.gate_up_proj", t))
        base = "model.language_model.layers.3.experts.1."
        gate = out[base + "gate_proj.weight"].float() * out[base + "gate_proj.weight_scale"].float()
        up = out[base + "up_proj.weight"].float() * out[base + "up_proj.weight_scale"].float()
        self.assertTrue(torch.allclose(gate, torch.ones(inter, hidden), rtol=1e-2))
        self.assertTrue(torch.allclose(up, torch.full((inter, hidden), 2.0), rtol=1e-2))
        self.assertLen(out, e * 4)

    def test_down_proj_per_expert(self):
        t = torch.randn(3, 8, 4).to(torch.bfloat16)
        out = dict(m.convert_tensor("model.language_model.layers.0.experts.down_proj", t))
        self.assertEqual(tuple(out["model.language_model.layers.0.experts.2.down_proj.weight"].shape), (8, 4))

    def test_linear_quantized_others_passthrough(self):
        w = torch.randn(16, 8).to(torch.bfloat16)
        out = dict(m.convert_tensor("model.language_model.layers.5.self_attn.k_proj.weight", w))
        self.assertEqual(set(out), {
            "model.language_model.layers.5.self_attn.k_proj.weight",
            "model.language_model.layers.5.self_attn.k_proj.weight_scale",
        })
        for keep in (
            "model.language_model.layers.5.router.proj.weight",
            "model.language_model.embed_tokens.weight",
            "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
        ):
            out = list(m.convert_tensor(keep, w))
            self.assertEqual(out[0][0], keep)
            self.assertEqual(out[0][1].dtype, torch.bfloat16)


if __name__ == "__main__":
    absltest.main()
