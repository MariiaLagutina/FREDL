"""Measure Python-owned BM25 index memory with a documented method."""

from collections.abc import Mapping
import gc
from sys import getsizeof
import tracemalloc
from typing import Any

from src.retrieval.bm25 import BM25Document, BM25Index, BM25Parameters


def measure_index_size(index: BM25Index) -> int:
    """Return recursive size without counting shared Python data twice."""
    seen: set[int] = set()
    return _deep_size(index, seen)


def measure_peak_build_memory(
    documents: tuple[BM25Document, ...],
    parameters: BM25Parameters,
) -> int:
    """Return peak traced Python allocations during a separate index build."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        warm_index = BM25Index(documents, parameters)
        del warm_index
        gc.collect()
        tracemalloc.reset_peak()
        baseline_bytes, _ = tracemalloc.get_traced_memory()
        measured_index = BM25Index(documents, parameters)
        _, peak_bytes = tracemalloc.get_traced_memory()
        del measured_index
        return max(0, peak_bytes - baseline_bytes)
    finally:
        if not was_tracing:
            tracemalloc.stop()


def _deep_size(value: Any, seen: set[int]) -> int:
    """Traverse containers and instance fields using one identity set."""
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = getsizeof(value)

    if isinstance(value, Mapping):
        return size + sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(_deep_size(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        size += _deep_size(vars(value), seen)

    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    for slot in slots:
        if hasattr(value, slot):
            size += _deep_size(getattr(value, slot), seen)
    return size
