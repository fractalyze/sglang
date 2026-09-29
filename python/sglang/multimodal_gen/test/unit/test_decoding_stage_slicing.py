import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage


class RecordingVAE(nn.Module):
    """Decodes by doubling, recording the batch size of every call."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def decode(self, latents):
        self.batch_sizes.append(latents.shape[0])
        return latents * 2


def _server_args(vae_slicing):
    return SimpleNamespace(pipeline_config=SimpleNamespace(vae_slicing=vae_slicing))


class TestDecodingStageSlicing(unittest.TestCase):
    def _decode(self, vae, vae_slicing, latents):
        stage = DecodingStage(vae)
        with patch.object(DecodingStage, "_get_vae_decode_fn", lambda self, v, a: v.decode):
            return stage._decode_batch(vae, _server_args(vae_slicing), latents)

    def test_slicing_decodes_one_sample_at_a_time_in_order(self):
        vae = RecordingVAE()
        latents = torch.arange(4.0).view(4, 1, 1, 1, 1)
        out = self._decode(vae, True, latents)
        self.assertEqual(vae.batch_sizes, [1, 1, 1, 1])
        torch.testing.assert_close(out, latents * 2)

    def test_without_slicing_the_batch_is_one_call(self):
        vae = RecordingVAE()
        out = self._decode(vae, False, torch.ones(4, 1, 1, 1, 1))
        self.assertEqual(vae.batch_sizes, [4])
        self.assertEqual(out.shape[0], 4)

    def test_single_sample_is_not_split(self):
        vae = RecordingVAE()
        self._decode(vae, True, torch.ones(1, 1, 1, 1, 1))
        self.assertEqual(vae.batch_sizes, [1])


if __name__ == "__main__":
    unittest.main()
