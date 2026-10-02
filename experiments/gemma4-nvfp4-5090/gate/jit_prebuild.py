"""Builds FlashInfer's SM120 CUTLASS fused-MoE JIT module outside any server.

Its 97 CUTLASS translation units are the host-memory hazard: single cicc
processes reached 9.6 GB on 2026-10-02 (`gate prebuild` hostmem.csv), so this
runs alone, capped, at config.JIT_MAX_JOBS, and the server then loads it from
the persisted cache.
"""

from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

if __name__ == "__main__":
    get_cutlass_fused_moe_module("120")
    print("fused_moe_120 built")
