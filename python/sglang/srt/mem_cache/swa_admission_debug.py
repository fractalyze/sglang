"""Debug-only trace of SWA admission accounting (G4POC_DEBUG_SWA_ADMISSION=1).

Not for merge: it exists to locate where prefill admission and the allocator
disagree about free sliding-window slots under HiCache.
"""

import logging
import os

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("G4POC_DEBUG_SWA_ADMISSION", "0") == "1"


def snap(tag, allocator, tree_cache, budget=None, **fields):
    if not ENABLED or not hasattr(allocator, "swa_available_size"):
        return
    parts = [
        f"swa_avail={allocator.swa_available_size()}",
        f"swa_evict={tree_cache.swa_evictable_size()}",
        f"swa_prot={tree_cache.swa_protected_size()}",
        f"full_avail={allocator.full_available_size()}",
        f"full_evict={tree_cache.full_evictable_size()}",
    ]
    if budget is not None:
        parts.append(f"swa_offset={getattr(budget, 'swa_offset', None)}")
        parts.append(f"rem_swa={budget.remaining_swa}")
    parts.extend(f"{k}={v}" for k, v in fields.items())
    logger.info("SWADBG %s %s", tag, " ".join(parts))
