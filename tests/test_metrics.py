from __future__ import annotations

import pytest
from benchlab.metrics import empirical_cdf, histogram, mean, percentile
from benchlab.records import RequestRecord
from benchlab.summary import _level


def test_percentile_type7_known_values() -> None:
    assert percentile([10, 20, 30, 40, 50], 50) == 30
    assert percentile([10, 20, 30, 40, 50], 0) == 10
    assert percentile([10, 20, 30, 40, 50], 100) == 50
    # rank = 0.95 * 4 = 3.8 -> 40 * 0.2 + 50 * 0.8
    assert percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48)
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([7], 95) == 7


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="between"):
        percentile([1.0], 101)
    with pytest.raises(ValueError, match="empty"):
        mean([])


def test_histogram_counts_cover_the_sample() -> None:
    counts = histogram([1, 2, 3, 4, 5])["counts"]
    assert isinstance(counts, list)
    assert sum(counts) == 5
    single = histogram([4, 4, 4])
    assert single["counts"] == [3]
    assert single["bin_edges_ms"] == [4, 4]
    assert histogram([]) == {"bin_edges_ms": [], "counts": []}


def test_cdf_ends_at_one() -> None:
    points = empirical_cdf([10, 30, 20])
    assert points[-1]["fraction"] == 1
    assert [point["e2e_ms"] for point in points] == [10, 20, 30]
    assert empirical_cdf([]) == []


def _record(
    *,
    phase: str,
    ordinal: int,
    started: int,
    token_times: list[int],
    finished: int | None = None,
    ok: bool = True,
    error_type: str | None = None,
) -> RequestRecord:
    return RequestRecord(
        request_id=f"{phase}-{ordinal}",
        phase=phase,
        ordinal=ordinal,
        concurrency=1,
        batch_size=1,
        prompt_ids=["p"],
        prompt_categories=["short_qa"],
        ok=ok,
        error_type=error_type,
        error_message=None if ok else "failed",
        started_us=started,
        finished_us=token_times[-1] if finished is None and token_times else (finished or started),
        token_times_us=token_times,
        output_text="x" if ok else "",
        backend="scripted",
        model_id="m",
    )


def test_level_math_matches_hand_calculation() -> None:
    records = [
        _record(phase="measure", ordinal=0, started=0, token_times=[100_000, 180_000]),
        _record(phase="measure", ordinal=1, started=0, token_times=[150_000]),
    ]
    level, _warnings = _level(records, concurrency=1, batch_size=1)
    assert level["n_ok"] == 2
    ttft = level["ttft_ms"]
    e2e = level["e2e_ms"]
    itl = level["itl_ms"]
    assert isinstance(ttft, dict) and isinstance(e2e, dict) and isinstance(itl, dict)
    assert ttft["p50"] == 125
    assert e2e["p50"] == 165
    assert itl["mean"] == 80
    assert itl["n"] == 1
    assert level["output_tokens_total"] == 3
    assert level["measurement_window_s"] == 0.18
    assert level["requests_per_sec"] == 11.1111
    assert level["output_tokens_per_sec"] == 16.6667
    assert level["mean_decode_tokens_per_sec"] == 12.5
    assert level["n_decode_samples"] == 1


def test_warmup_and_failures_stay_out_of_latency() -> None:
    records = [
        _record(phase="warmup", ordinal=0, started=0, token_times=[1_000]),
        _record(phase="measure", ordinal=0, started=0, token_times=[50_000]),
        _record(
            phase="measure",
            ordinal=1,
            started=0,
            token_times=[],
            finished=10_000,
            ok=False,
            error_type="timeout",
        ),
    ]
    level, warnings = _level(records, concurrency=1, batch_size=1)
    assert level["n_measured"] == 2
    assert level["n_warmup_excluded"] == 1
    assert level["n_ok"] == 1
    assert level["n_errors"] == 1
    assert level["n_timeouts"] == 1
    assert level["error_rate"] == 0.5
    e2e = level["e2e_ms"]
    assert isinstance(e2e, dict)
    assert e2e["p50"] == 50
    assert e2e["n"] == 1
    assert any("p99" in warning for warning in warnings)
    histogram_counts = level["e2e_histogram"]
    assert isinstance(histogram_counts, dict)
    assert sum(histogram_counts["counts"]) == 1
