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
# Cumulative counters since server start, printed on every PFXDBG line.
# backup_fail_*: write-through backups that returned no host slots (the node stays
# device-only and its KV dies when the device evicts it); "full" = the KV host pool
# could not be reclaimed, "write" = the controller's KV or SWA host alloc failed.
# dropped: Full-layer tokens evicted from device with no host copy.
_COUNTERS = {"backup_fail_full": 0, "backup_fail_write": 0, "backup_fail_tokens": 0, "dropped": 0}


def note_match(req, full_kv_hit_length):
    if ENABLED and req is not None:
        req._g4poc_full_kv_hit = full_kv_hit_length


def note_backup_fail(reason, tokens):
    if ENABLED:
        _COUNTERS["backup_fail_" + reason] += 1
        _COUNTERS["backup_fail_tokens"] += tokens


def note_dropped(tokens):
    if ENABLED and tokens:
        _COUNTERS["dropped"] += tokens


def _pool_free(tree_cache):
    """(host KV free, host SWA free, device full free, device SWA free) in tokens; -1 if absent."""
    out = [-1, -1, -1, -1]
    try:
        from sglang.srt.mem_cache.hicache_storage import PoolName

        group = tree_cache.cache_controller.mem_pool_host
        out[0] = group.get_pool(PoolName.KV).available_size()
        if PoolName.SWA in group.entry_map:
            out[1] = group.get_pool(PoolName.SWA).available_size()
    except Exception:
        pass
    try:
        alloc = tree_cache.token_to_kv_pool_allocator
        out[2] = alloc.full_available_size()
        out[3] = alloc.swa_available_size()
    except Exception:
        pass
    return out


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


def log_admission(req, *, device_prefix, loaded, tree_cache=None):
    if not ENABLED:
        return
    try:
        _log_admission(req, device_prefix=device_prefix, loaded=loaded, tree_cache=tree_cache)
    except Exception:  # a debug log must never take the scheduler down
        logger.exception("PFXDBG log failed for rid=%s", getattr(req, "rid", None))


def _log_admission(req, *, device_prefix, loaded, tree_cache):
    lb = getattr(req, "_g4poc_load_back", None) or {}
    hfree_kv, hfree_swa, dfree_full, dfree_swa = _pool_free(tree_cache)
    # sess: the first 32 ids carry the per-session nonce ("[session <nonce>]" at the start
    # of the system prompt), so it names one session instance across its turns.
    logger.info(
        "PFXDBG rid=%s hash=%s sess=%s fill=%d device_hit=%d host_hit=%d swa_host_hit=%d "
        "loaded=%d lb_full=%s lb_swa=%s stale_full=%s stale_swa=%s full_kv=%d "
        "hfree_kv=%d hfree_swa=%d dfree_full=%d dfree_swa=%d "
        "bk_fail_full=%d bk_fail_write=%d bk_fail_tok=%d dropped=%d",
        req.rid,
        ids_hash(req.origin_input_ids),
        ids_hash(req.origin_input_ids[:32]),
        len(req.full_untruncated_fill_ids),
        device_prefix,
        req.host_hit_length,
        req.swa_host_hit_length,
        loaded,
        lb.get("full", 0),
        lb.get("swa", 0),
        lb.get("stale_full", 0),
        lb.get("stale_swa", 0),
        getattr(req, "_g4poc_full_kv_hit", -1),
        hfree_kv,
        hfree_swa,
        dfree_full,
        dfree_swa,
        _COUNTERS["backup_fail_full"],
        _COUNTERS["backup_fail_write"],
        _COUNTERS["backup_fail_tokens"],
        _COUNTERS["dropped"],
    )
