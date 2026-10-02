# Copyright 2026 Fractalyze Inc. All rights reserved.
"""The grid barrier's watchdog record, and the probe kernel that tests the barrier.

A kernel that waits at a barrier longer than its timeout writes where it
stopped into a host-mapped `ErrorRecord` and traps. The trap leaves the CUDA
context unusable, so the process can report the record but must then exit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.kernels.decode_mk import _ext
from sglang.kernels.decode_mk.gemv import num_ctas

DEFAULT_TIMEOUT_NS = 1_000_000_000
_TIMEOUT_STATUS = 1


@dataclass(frozen=True)
class Hang:
    """Where a launch stopped: `cta`'s watchdog fired at launch-wide barrier
    `barrier` of decode step `step`, with `arrived` of `expected` CTAs there."""

    cta: int
    barrier: int
    step: int
    arrived: int
    expected: int


class BarrierTimeout(RuntimeError):
    def __init__(self, hang: Hang) -> None:
        super().__init__(
            f"grid barrier {hang.barrier} of step {hang.step} timed out on CTA "
            f"{hang.cta}: {hang.arrived} of {hang.expected} CTAs arrived; the "
            "CUDA context is lost"
        )
        self.hang = hang


class ErrorRecord:
    """The watchdog's record, in pinned host memory the kernel writes directly."""

    def __init__(self) -> None:
        words = _ext.load().ERROR_RECORD_WORDS
        self.tensor = torch.zeros(words, dtype=torch.int32, pin_memory=True)

    def hang(self) -> Hang | None:
        status, *fields = self.tensor.tolist()
        return Hang(*fields) if status == _TIMEOUT_STATUS else None

    def synchronize(self) -> None:
        """Waits for the GPU; raises BarrierTimeout if a watchdog fired."""
        try:
            torch.cuda.synchronize()
        except RuntimeError as err:
            hang = self.hang()
            if hang is not None:
                raise BarrierTimeout(hang) from err
            raise


def sync_words(device: torch.device | str = "cuda") -> torch.Tensor:
    """The zeroed-per-launch device words a launch's barriers use."""
    return torch.zeros(_ext.load().SYNC_WORDS, dtype=torch.int32, device=device)


def run_barrier_probe(
    rounds: int,
    skip_cta: int = -1,
    skip_barrier: int = -1,
    step: int = 0,
    timeout_ns: int = DEFAULT_TIMEOUT_NS,
) -> int:
    """Runs `rounds` grid barriers on one CTA per SM; returns the visibility violations.

    Each round, every thread of every CTA writes its own word, passes the
    barrier and checks every word. CTA `skip_cta` leaves instead of arriving
    at barrier `skip_barrier`, which must raise BarrierTimeout.
    """
    ctas = num_ctas()
    record = ErrorRecord()
    mismatches = torch.zeros(1, dtype=torch.int32, device="cuda")
    _ext.load().run_barrier_probe(
        rounds=rounds,
        skip_cta=skip_cta,
        skip_barrier=skip_barrier,
        step=step,
        timeout_ns=timeout_ns,
        sync=sync_words(),
        error=record.tensor,
        slots=torch.zeros(2 * ctas * _ext.load().PROBE_WORDS_PER_CTA, dtype=torch.int32,
                          device="cuda"),
        mismatches=mismatches,
        num_ctas=ctas,
    )
    record.synchronize()
    return int(mismatches.item())
