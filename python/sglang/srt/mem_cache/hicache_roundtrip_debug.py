"""Debug-only byte-level HiCache round-trip check (G4POC_DEBUG_HICACHE_ROUNDTRIP=1).

Not for merge. At each write-through ack and each load-back ack, compares the
device and host bytes of every node on the path that holds both copies, token by
token and per layer, for the full-attention pool (KV) and the sliding pool
(SWA). After a load-back it also checks that the full->SWA slot mapping of the
loaded SWA tokens points at their SWA values.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("G4POC_DEBUG_HICACHE_ROUNDTRIP", "0") == "1"


def _rows(buffer, indices):
    return buffer[indices].contiguous().view(torch.uint8).reshape(len(indices), -1)


def _compare_pool(entry, device_indices, host_indices):
    device_pool, host_pool = entry.device_pool, entry.host_pool
    host_indices = host_indices.cpu()
    bad = {}
    for side, dev_bufs, host_refs in (
        ("k", device_pool.k_buffer, host_pool.k_data_refs),
        ("v", device_pool.v_buffer, host_pool.v_data_refs),
    ):
        tokens = torch.zeros(len(device_indices), dtype=torch.bool)
        layers = []
        for layer in range(host_pool.layer_num):
            dev = _rows(dev_bufs[layer], device_indices).cpu()
            host = _rows(host_refs[layer], host_indices)
            diff = (dev != host).any(dim=1)
            if bool(diff.any()):
                layers.append(layer)
                tokens |= diff
        if layers:
            bad[side] = (int(tokens.sum()), layers)
    return bad


def check_path(tag, cache, node_id):
    """Compare device vs host bytes for every node from node_id to the root."""
    if not ENABLED or cache.cache_controller is None:
        return
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    group = cache.cache_controller.mem_pool_host
    tree = cache.tree_core
    allocator = cache.token_to_kv_pool_allocator
    node = tree.node_by_id(node_id)
    checked = {"full": 0, "swa": 0}
    while node is not tree.root_node:
        for ct, pool_name, label in (
            (ComponentType.FULL, PoolName.KV, "full"),
            (ComponentType.SWA, PoolName.SWA, "swa"),
        ):
            cd = node.component_data[ct]
            entry = group.entry_map.get(pool_name)
            if entry is None:
                continue
            if cd.value is None or cd.host_value is None:
                continue
            if len(cd.value) != len(cd.host_value):
                logger.warning(
                    "HCRT %s node=%s %s length device=%d host=%d",
                    tag, node.id, label, len(cd.value), len(cd.host_value),
                )
                continue
            checked[label] += len(cd.value)
            bad = _compare_pool(entry, cd.value, cd.host_value)
            if bad:
                logger.warning(
                    "HCRT %s MISMATCH node=%s %s n=%d bad=%s", tag, node.id, label,
                    len(cd.value), bad,
                )
            if label == "swa" and tag == "load":
                full = node.component_data[ComponentType.FULL].value
                if full is not None and len(full) == len(cd.value):
                    mapped = allocator.full_to_swa_index_mapping[full]
                    wrong = int((mapped != cd.value).sum())
                    if wrong:
                        logger.warning(
                            "HCRT load MAPPING node=%s %d/%d full->swa slots differ",
                            node.id, wrong, len(full),
                        )
        node = node.parent
    logger.warning("HCRT %s checked node=%s tokens=%s", tag, node_id, checked)
