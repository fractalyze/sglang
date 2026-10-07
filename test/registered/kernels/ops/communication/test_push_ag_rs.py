from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import sp_collective
from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    graph_capture,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=120, stage="nightly", runner_config="8-gpu-h200")

_HIDDEN_SIZE = 7168
_PUSH = sp_collective.Dispatch(
    "push", sp_collective.Tuning(num_blocks=64, block_size=512)
)


@cache_once
def _tp_group():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=local_rank
    )
    publish_build_topology(world_rank=rank, tp_size=world_size)
    with envs.SGLANG_OPT_USE_PUSH_AG_RS.override(True):
        initialize_model_parallel()
    group = get_parallel().tp_group
    if group.push_ag_rs is None or group.push_ag_rs.disabled:
        pytest.skip("push all-gather / reduce-scatter needs multicast NVLink")
    return group


@pytest.fixture
def forced_push(monkeypatch):
    """Route every eligible call to push, whatever the device's table says."""
    push = _tp_group().push_ag_rs
    monkeypatch.setattr(sp_collective, "get_dispatch", lambda *args: _PUSH)
    push._dispatches.clear()
    yield push
    push._dispatches.clear()


def _rows(seed: int, rows: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        rows, _HIDDEN_SIZE, generator=gen, device="cuda", dtype=torch.bfloat16
    )


def _fixed_order_sum(input: torch.Tensor, group) -> torch.Tensor:
    """This rank's rows of the rank-0-first fp32 sum the push kernel computes."""
    inputs = torch.empty(
        (group.world_size, *input.shape), dtype=input.dtype, device=input.device
    )
    dist.all_gather_into_tensor(inputs, input, group=group.device_group)
    total = inputs[0].float()
    for peer in inputs[1:]:
        total = total + peer.float()
    return total.to(input.dtype).tensor_split(group.world_size)[group.rank_in_group]


@pytest.mark.parametrize("global_tokens", [8, 128, 512, 1024])
@torch.inference_mode()
def test_all_gather_matches_nccl(forced_push, global_tokens):
    group = _tp_group()
    local = _rows(group.rank_in_group, global_tokens // group.world_size)
    expected = torch.empty(global_tokens, _HIDDEN_SIZE, **_like(local))
    dist.all_gather_into_tensor(expected, local, group=group.device_group)

    output = torch.empty_like(expected)
    assert forced_push.all_gather(output, local)
    group.all_gather_into_tensor(output, local)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("global_tokens", [8, 128, 512, 1024])
@torch.inference_mode()
def test_reduce_scatter_is_fixed_order_and_repeatable(forced_push, global_tokens):
    group = _tp_group()
    input = _rows(100 + group.rank_in_group, global_tokens)
    expected = _fixed_order_sum(input, group)

    first = torch.empty_like(expected)
    assert forced_push.reduce_scatter(first, input)
    second = torch.empty_like(expected)
    group.reduce_scatter_tensor(second, input)
    nccl = torch.empty_like(expected)
    dist.reduce_scatter_tensor(nccl, input, group=group.device_group)
    torch.cuda.synchronize()

    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    # NCCL sums in bf16 along its ring, so it agrees only to bf16 rounding.
    torch.testing.assert_close(nccl, expected, rtol=2e-2, atol=6e-2)


@torch.inference_mode()
def test_graph_replay_with_new_inputs(forced_push):
    group = _tp_group()
    rank, world = group.rank_in_group, group.world_size
    local = _rows(200 + rank, 512 // world)
    gathered = torch.empty(512, _HIDDEN_SIZE, **_like(local))
    scattered = torch.empty_like(local)
    # A JIT build inside stream capture fails; run the kernels once eagerly.
    group.all_gather_into_tensor(gathered, local)
    group.reduce_scatter_tensor(scattered, gathered)

    with graph_capture() as ctx:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=ctx.stream):
            for _ in range(3):
                group.all_gather_into_tensor(gathered, local)
                group.reduce_scatter_tensor(scattered, gathered)

    for step in range(2):
        local.copy_(_rows(300 + 10 * step + rank, local.shape[0]))
        graph.replay()
        torch.cuda.synchronize()
        expected_gather = torch.empty_like(gathered)
        dist.all_gather_into_tensor(expected_gather, local, group=group.device_group)
        torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)
        torch.testing.assert_close(
            scattered, _fixed_order_sum(gathered, group), rtol=0, atol=0
        )
    graph.reset()


@torch.inference_mode()
def test_falls_back_past_the_push_slot(forced_push):
    group = _tp_group()
    rows_per_rank = forced_push.comm.max_push_size // (_HIDDEN_SIZE * 2) + 1
    local = _rows(group.rank_in_group, rows_per_rank)
    output = torch.empty(group.world_size * rows_per_rank, _HIDDEN_SIZE, **_like(local))
    assert not forced_push.all_gather(output, local)
    assert not forced_push.reduce_scatter(local, output)


def test_device_table_routes_decode_sizes():
    group = _tp_group()
    if (
        sp_collective.get_dispatch(
            "all_gather", group.world_size, _HIDDEN_SIZE, 512, group.device
        )
        is None
    ):
        pytest.skip(f"no sp_collective table for {torch.cuda.get_device_name()}")
    push = group.push_ag_rs
    push._dispatches.clear()
    assert push._dispatch("all_gather", _HIDDEN_SIZE, 512) is not None
    assert push._dispatch("reduce_scatter", _HIDDEN_SIZE, 512) is not None
    assert push._dispatch("reduce_scatter", _HIDDEN_SIZE, 2048) is None


def _like(tensor: torch.Tensor) -> dict:
    return {"dtype": tensor.dtype, "device": tensor.device}


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
