"""Pure metric math.

Percentiles use Hyndman-Fan type 7 (linear interpolation), the same definition
as NumPy's default ``percentile``. Quantiles are reported on a 0–100 scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sample")
    return sum(values) / len(values)


def percentile(values: Sequence[float], q: float) -> float:
    """Return the type-7 percentile of ``values`` for ``q`` in ``[0, 100]``."""
    if not values:
        raise ValueError("percentile of an empty sample")
    if q < 0 or q > 100:
        raise ValueError("percentile q must be between 0 and 100")
    ordered = sorted(float(v) for v in values)
    count = len(ordered)
    if count == 1 or q == 0:
        return ordered[0]
    if q == 100:
        return ordered[-1]
    rank = (q / 100.0) * (count - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def histogram(values: Sequence[float]) -> dict[str, list[float] | list[int]]:
    """Equal-width histogram using Sturges' rule.

    Bin edges are rounded to 3 decimals for display. Counts sum to ``len(values)``.
    A constant sample produces a single bin whose edges are both the sample value.
    """
    if not values:
        return {"bin_edges_ms": [], "counts": []}
    ordered = [float(v) for v in values]
    low = min(ordered)
    high = max(ordered)
    if low == high:
        return {"bin_edges_ms": [round(low, 3), round(low, 3)], "counts": [len(ordered)]}
    bins = max(1, math.ceil(math.log2(len(ordered))) + 1)
    width = (high - low) / bins
    counts = [0 for _ in range(bins)]
    for value in ordered:
        index = int((value - low) / width)
        if index >= bins:
            index = bins - 1
        counts[index] += 1
    edges = [round(low + i * width, 3) for i in range(bins + 1)]
    edges[-1] = round(high, 3)
    return {"bin_edges_ms": edges, "counts": counts}


def empirical_cdf(values: Sequence[float], max_points: int = 200) -> list[dict[str, float]]:
    """Empirical CDF as ``fraction = rank / n`` using 1-based ranks.

    Large samples are downsampled across the ordered values. ``fraction`` still
    uses the original rank, so the last point is 1.
    """
    if not values:
        return []
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    ordered = sorted(float(v) for v in values)
    count = len(ordered)
    if count <= max_points:
        indexes = list(range(count))
    else:
        indexes = sorted({round(i * (count - 1) / (max_points - 1)) for i in range(max_points)})
    return [
        {"e2e_ms": round(ordered[index], 3), "fraction": round((index + 1) / count, 6)}
        for index in indexes
    ]
