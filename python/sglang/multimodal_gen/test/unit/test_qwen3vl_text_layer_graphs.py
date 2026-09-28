# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL text-encoder layer graphs (SGLANG_ENABLE_QWEN3VL_TEXT_CUDA_GRAPH) against eager; no checkpoint needed."""

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLTextModel
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.srt.environ import envs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def models():
    set_global_server_args(
        ServerArgs(model_path="Qwen/Qwen-Image-2.1", num_gpus=1, attention_backend="torch_sdpa")
    )
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    config = Qwen3VLTextConfig(
        vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        rope_scaling={"rope_type": "default", "mrope_section": [8, 12, 12], "mrope_interleaved": True},
    )
    torch.manual_seed(0)
    eager = Qwen3VLTextModel(config).cuda().bfloat16().eval()
    for param in eager.parameters():
        torch.nn.init.normal_(param, std=0.05)
    with envs.SGLANG_ENABLE_QWEN3VL_TEXT_CUDA_GRAPH.override(True):
        graphed = Qwen3VLTextModel(config).cuda().bfloat16().eval()
    graphed.load_state_dict(eager.state_dict())
    assert eager.layer_graphs is None and graphed.layer_graphs is not None
    return eager, graphed


def hidden_states(model, input_ids):
    with torch.no_grad(), set_forward_context(None, None):
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states, out.last_hidden_state


def assert_same(eager, graphed, input_ids):
    (eager_layers, eager_last), (graph_layers, graph_last) = (
        hidden_states(eager, input_ids),
        hidden_states(graphed, input_ids),
    )
    assert len(graph_layers) == len(eager_layers)
    for e, g in zip(eager_layers, graph_layers):
        torch.testing.assert_close(g, e, atol=0, rtol=0)
    torch.testing.assert_close(graph_last, eager_last, atol=0, rtol=0)


def test_graphed_layers_are_bitwise_per_length(models):
    eager, graphed = models
    for tokens in (34, 57, 34):  # a repeated length replays its captured graph
        ids = torch.randint(0, 1000, (1, tokens), device="cuda")
        assert_same(eager, graphed, ids)


def test_graphed_layers_follow_moved_weights(models):
    eager, graphed = models
    ids = torch.randint(0, 1000, (1, 34), device="cuda")
    assert_same(eager, graphed, ids)
    # the text encoder's CPU offload brings the weights back in new memory
    graphed.cpu()
    graphed.cuda()
    assert_same(eager, graphed, ids)
