"""Debug-only per-request prefix-match log (G4POC_DEBUG_PREFIX_SPLIT=1). Not for merge.

Logs, per prefill admission, the device hit, the HiCache host hit (full and SWA) and what the
load-back actually restored, and counts loaded host slots whose copy is stale: written from one
computation of a prefix, while the node's device value was later replaced by another request's
computation (SWA tombstone recovery). A backup that rewrites a slot clears its stale mark.
"""

import hashlib
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("G4POC_DEBUG_PREFIX_SPLIT", "0") == "1"

_STALE = {"full": set(), "swa": set()}


def _slots(indices):
    return [] if indices is None else indices.reshape(-1).tolist()


def mark_stale(pool, host_indices):
    if ENABLED and host_indices is not None:
        _STALE[pool].update(_slots(host_indices))


def clear_written(pool, host_indices):
    if ENABLED and host_indices is not None and _STALE[pool]:
        _STALE[pool].difference_update(_slots(host_indices))


def count_stale(pool, host_indices):
    if not ENABLED or host_indices is None or not _STALE[pool]:
        return 0
    stale = _STALE[pool]
    return sum(1 for s in _slots(host_indices) if s in stale)


def ids_hash(ids):
    return hashlib.md5(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:12]


def log_admission(req, *, device_prefix, loaded):
    if not ENABLED:
        return
    lb = getattr(req, "_g4poc_load_back", None) or {}
    logger.info(
        "PFXDBG rid=%s hash=%s fill=%d device_hit=%d host_hit=%d swa_host_hit=%d "
        "loaded=%d lb_full=%s lb_swa=%s stale_full=%s stale_swa=%s",
        req.rid,
        ids_hash(req.origin_input_ids),
        len(req.full_untruncated_fill_ids),
        device_prefix,
        req.host_hit_length,
        req.swa_host_hit_length,
        loaded,
        lb.get("full", 0),
        lb.get("swa", 0),
        lb.get("stale_full", 0),
        lb.get("stale_swa", 0),
    )
