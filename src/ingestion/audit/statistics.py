"""Build stable chunk-size distribution summaries."""

from statistics import median

from src.ingestion.audit.models import ChunkSizeSummary


def summarize_sizes(sizes: list[int]) -> ChunkSizeSummary | None:
    """Return ordered nearest-rank distribution statistics."""
    if not sizes:
        return None

    ordered = sorted(sizes)
    p95_index = ((95 * len(ordered) + 99) // 100) - 1
    return ChunkSizeSummary(
        count=len(ordered),
        minimum=ordered[0],
        median=float(median(ordered)),
        p95=ordered[p95_index],
        maximum=ordered[-1],
    )
