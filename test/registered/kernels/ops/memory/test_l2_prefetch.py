import sys

import pytest
import torch

from sglang.kernels.ops.memory.l2_prefetch import (
    _RANGE_BYTES,
    l2_prefetch,
    plan_l2_prefetch,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="l2_prefetch requires SM90 or newer",
)


def _covered(ranges: torch.Tensor) -> list:
    """The table's ranges merged into (start, end) intervals, in table order."""
    spans = []
    for start, nbytes in ranges.tolist():
        if spans and spans[-1][1] == start:
            spans[-1][1] = start + nbytes
        else:
            spans.append([start, start + nbytes])
    return [tuple(s) for s in spans]


def test_plan_covers_tensors_in_order():
    a = torch.empty(3 * _RANGE_BYTES + 48, dtype=torch.uint8, device="cuda")
    b = torch.empty(1000, dtype=torch.bfloat16, device="cuda")
    ranges = plan_l2_prefetch([a, b], budget_bytes=1 << 30)

    assert ranges.dtype == torch.int64 and ranges.shape[1] == 2
    assert (ranges[:, 0] % 16 == 0).all() and (ranges[:, 1] % 16 == 0).all()
    assert (ranges[:, 1] <= _RANGE_BYTES).all()
    assert _covered(ranges) == [
        (a.data_ptr(), a.data_ptr() + a.numel()),
        (b.data_ptr(), b.data_ptr() + 2000),
    ]


def test_plan_stops_at_budget():
    a = torch.empty(4 * _RANGE_BYTES, dtype=torch.uint8, device="cuda")
    b = torch.empty(_RANGE_BYTES, dtype=torch.uint8, device="cuda")
    budget = _RANGE_BYTES * 4 + 100
    ranges = plan_l2_prefetch([a, b], budget_bytes=budget).tolist()

    assert ranges[-1] == [b.data_ptr(), 96]
    assert sum(nbytes for _, nbytes in ranges[:-1]) == a.numel()


def test_plan_spans_strided_and_unaligned_views():
    base = torch.empty(64, 512, dtype=torch.bfloat16, device="cuda")
    transposed = base.t()
    offset = base.view(-1)[3:1000]
    ranges = plan_l2_prefetch([transposed, offset], budget_bytes=1 << 30)

    start = offset.data_ptr() // 16 * 16
    end = -(-(offset.data_ptr() + 997 * 2) // 16) * 16
    assert _covered(ranges)[0] == (base.data_ptr(), base.data_ptr() + base.nbytes)
    assert _covered(ranges)[1:] == [(start, end)]


def test_plan_skips_empty_tensors():
    empty = torch.empty(0, device="cuda")
    assert plan_l2_prefetch([empty], budget_bytes=1 << 20).shape == (0, 2)


@pytest.mark.parametrize("evict_last", [False, True])
def test_prefetch_writes_nothing(evict_last):
    data = torch.randint(0, 256, (8 << 20,), dtype=torch.uint8, device="cuda")
    expected = data.clone()
    l2_prefetch(plan_l2_prefetch([data], budget_bytes=1 << 30), evict_last=evict_last)
    torch.cuda.synchronize()
    assert torch.equal(data, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
