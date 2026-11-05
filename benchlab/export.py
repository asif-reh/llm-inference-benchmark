"""Write summary JSON, summary CSV, and per-request JSONL."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

from benchlab.records import RequestRecord

CSV_FIELDS = [
    "concurrency",
    "batch_size",
    "n_measured",
    "n_ok",
    "n_errors",
    "n_timeouts",
    "error_rate",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "itl_mean_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "output_tokens",
    "output_tokens_per_sec",
    "requests_per_sec",
    "mean_decode_tokens_per_sec",
    "measurement_window_s",
]


def write_run(
    directory: Path,
    summary: dict[str, object],
    records: list[RequestRecord],
    *,
    copies: list[Path],
    overwrite: bool,
    include_output_text: bool,
) -> None:
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{directory} is not empty. Set output.overwrite: true or pick a new directory."
        )
    directory.mkdir(parents=True, exist_ok=True)
    document = json.dumps(summary, indent=2) + "\n"
    (directory / "summary.json").write_text(document, encoding="utf-8")
    (directory / "summary.csv").write_text(summary_to_csv(summary), encoding="utf-8")
    lines = [
        json.dumps(record_dict(record, include_text=include_output_text), separators=(",", ":"))
        for record in _sorted_records(records)
    ]
    body = ("\n".join(lines) + "\n") if lines else ""
    (directory / "requests.jsonl").write_text(body, encoding="utf-8")
    for destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document, encoding="utf-8")


def summary_to_csv(summary: dict[str, object]) -> str:
    levels = summary.get("levels")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    if isinstance(levels, list):
        for level in levels:
            if isinstance(level, dict):
                writer.writerow(_csv_row(level))
    return buffer.getvalue()


def record_dict(record: RequestRecord, *, include_text: bool) -> dict[str, object]:
    offsets = [round((stamp - record.started_us) / 1000.0, 3) for stamp in record.token_times_us]
    itl: list[float] = []
    ttft: float | None = None
    e2e: float | None = None
    if record.ok and offsets:
        ttft = offsets[0]
        e2e = round((record.finished_us - record.started_us) / 1000.0, 3)
        itl = [round(offsets[index] - offsets[index - 1], 3) for index in range(1, len(offsets))]
    payload: dict[str, object] = {
        "request_id": record.request_id,
        "phase": record.phase,
        "ordinal": record.ordinal,
        "concurrency": record.concurrency,
        "batch_size": record.batch_size,
        "prompt_ids": list(record.prompt_ids),
        "prompt_categories": list(record.prompt_categories),
        "ok": record.ok,
        "stats_included": record.ok,
        "error_type": record.error_type,
        "error_message": record.error_message,
        "t_start_us": record.started_us,
        "t_finish_us": record.finished_us,
        "ttft_ms": ttft,
        "e2e_ms": e2e,
        "itl_ms": itl,
        "token_offset_ms": offsets,
        "output_tokens": len(record.token_times_us),
        "backend": record.backend,
        "model_id": record.model_id,
    }
    if include_text:
        payload["output_text"] = record.output_text
    return payload


def _sorted_records(records: list[RequestRecord]) -> list[RequestRecord]:
    phase_order = {"warmup": 0, "measure": 1}
    return sorted(
        records,
        key=lambda record: (
            record.concurrency,
            record.batch_size,
            phase_order.get(record.phase, 9),
            record.ordinal,
        ),
    )


def _csv_row(level: dict[str, object]) -> dict[str, object]:
    ttft = _mapping(level.get("ttft_ms"))
    itl = _mapping(level.get("itl_ms"))
    e2e = _mapping(level.get("e2e_ms"))
    return {
        "concurrency": level.get("concurrency"),
        "batch_size": level.get("batch_size"),
        "n_measured": level.get("n_measured"),
        "n_ok": level.get("n_ok"),
        "n_errors": level.get("n_errors"),
        "n_timeouts": level.get("n_timeouts"),
        "error_rate": level.get("error_rate"),
        "ttft_p50_ms": ttft.get("p50"),
        "ttft_p95_ms": ttft.get("p95"),
        "itl_mean_ms": itl.get("mean"),
        "itl_p50_ms": itl.get("p50"),
        "itl_p95_ms": itl.get("p95"),
        "e2e_p50_ms": e2e.get("p50"),
        "e2e_p95_ms": e2e.get("p95"),
        "e2e_p99_ms": e2e.get("p99"),
        "output_tokens": level.get("output_tokens_total"),
        "output_tokens_per_sec": level.get("output_tokens_per_sec"),
        "requests_per_sec": level.get("requests_per_sec"),
        "mean_decode_tokens_per_sec": level.get("mean_decode_tokens_per_sec"),
        "measurement_window_s": level.get("measurement_window_s"),
    }


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}
