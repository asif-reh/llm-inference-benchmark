"""Aggregate measured requests into the summary document the dashboard reads."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from benchlab import __version__
from benchlab.config import LoadedConfig
from benchlab.definitions import DEFINITIONS, PERCENTILE_METHOD, SYNTHETIC_NOTICE
from benchlab.metrics import empirical_cdf, histogram, mean, percentile
from benchlab.prompts import file_sha256
from benchlab.records import RequestRecord
from benchlab.vram import VramReading


def build_summary(
    loaded: LoadedConfig,
    records: list[RequestRecord],
    *,
    vram: VramReading,
    backend_name: str,
    model_id: str,
    revision: str | None,
    synthetic: bool,
    backend_warnings: list[str],
) -> dict[str, object]:
    config = loaded.config
    warnings = list(backend_warnings)
    if synthetic:
        warnings.append("The mock latency model ignores temperature and prompt semantics.")
    elif config.workload.temperature != 0:
        warnings.append("temperature is non-zero, so sampling is part of the latency spread.")
    if config.workload.seed is None:
        warnings.append("workload.seed is null, so the prompt file order was used as-is.")

    levels: list[dict[str, object]] = []
    measured_errors: list[dict[str, object]] = []
    warmup_errors: list[dict[str, object]] = []
    for batch_size in config.workload.batch_sizes:
        for concurrency in config.workload.concurrency:
            group = [
                record
                for record in records
                if record.concurrency == concurrency and record.batch_size == batch_size
            ]
            level, level_warnings = _level(group, concurrency=concurrency, batch_size=batch_size)
            levels.append(level)
            warnings.extend(level_warnings)
            measured_errors.extend(
                _error_rows(record for record in group if record.phase == "measure")
            )
            warmup_errors.extend(
                _error_rows(record for record in group if record.phase == "warmup")
            )

    if warmup_errors:
        warnings.append(
            f"{len(warmup_errors)} warmup request(s) failed. They are excluded from "
            "latency and from error_rate, and listed under warmup_errors."
        )

    notice = SYNTHETIC_NOTICE if synthetic else None
    return {
        "notice": notice,
        "schema_version": 1,
        "meta": {
            "run_id": loaded.run_id,
            "run_name": config.run_name,
            "created_at": loaded.created_at,
            "created_at_is_frozen": loaded.created_at_is_frozen,
            "package_version": __version__,
            "synthetic": synthetic,
            "seed": config.workload.seed,
            "clock": config.clock,
            "clock_origin": "run_start",
            "warmup_excluded_from_stats": True,
            "percentile_method": PERCENTILE_METHOD,
            "closed_loop": True,
            "backend": backend_name,
            "model_id": model_id,
            "revision": revision,
        },
        "config": cast(dict[str, object], config.model_dump(mode="json")),
        "prompts": {
            "path": config.workload.prompts,
            "count": len(loaded.prompts),
            "sha256": file_sha256(loaded.prompts_path),
        },
        "hardware": vram.as_dict(),
        "warnings": warnings,
        "limitations": limitations_for(backend_name, synthetic=synthetic),
        "levels": levels,
        "errors": measured_errors,
        "warmup_errors": warmup_errors,
        "definitions": dict(DEFINITIONS),
    }


def limitations_for(backend_type: str, *, synthetic: bool) -> list[str]:
    items = [
        "Figures are client-observed. Remote backends include network time.",
        "Tail percentiles with small N describe this sample. "
        "They are not a stable service-level objective.",
        "Output tokens are streamed pieces as delivered, not retokenized with a local tokenizer.",
        "Warmup requests are stored in requests.jsonl and excluded from summary statistics.",
        "Load is closed-loop: at most `concurrency` requests are in flight. "
        "This is not an open-loop arrival process.",
    ]
    if synthetic or backend_type == "mock":
        items.insert(
            0,
            SYNTHETIC_NOTICE
            + ". The mock backend simulates token arrival with a discrete-event clock. "
            "It does not run a model and it does not measure a GPU.",
        )
    if backend_type == "huggingface":
        items.append(
            "The Inference API does not expose server VRAM. VRAM is omitted rather than guessed."
        )
        items.append(
            "Serverless routing, cold starts, and shared capacity are inside the numbers. "
            "This is not a dedicated replica."
        )
    elif backend_type == "vllm":
        items.append(
            "This client sends one prompt per request. vLLM batching is the server's configuration."
        )
        items.append("Publish VRAM only from a probe on the vLLM host.")
    elif backend_type == "tgi":
        items.append(
            "This client sends one input per request. TGI batching is the server's configuration."
        )
        items.append("Publish VRAM only from a probe on the TGI host.")
    elif backend_type == "transformers":
        items.append(
            "The reference transformers server serializes generation. Concurrency measures "
            "that queue plus compute, not a production scheduler."
        )
        items.append("VRAM is reported only when the server process can read it.")
    return items


def _level(
    records: list[RequestRecord], *, concurrency: int, batch_size: int
) -> tuple[dict[str, object], list[str]]:
    measured = sorted(
        (record for record in records if record.phase == "measure"),
        key=lambda record: record.ordinal,
    )
    warmup = [record for record in records if record.phase == "warmup"]
    ok = [record for record in measured if record.ok and record.token_times_us]
    warnings: list[str] = []
    n_measured = len(measured)
    n_errors = sum(1 for record in measured if not record.ok)
    n_timeouts = sum(1 for record in measured if record.error_type == "timeout")
    window_us = 0
    if measured:
        window_us = max(record.finished_us for record in measured) - min(
            record.started_us for record in measured
        )
        if window_us <= 0:
            warnings.append(
                f"concurrency={concurrency} batch={batch_size}: measurement window is 0, "
                "so throughput is null."
            )
    if len(ok) < 20:
        warnings.append(
            f"concurrency={concurrency} batch={batch_size}: p95 is a coarse description "
            f"of this sample (N={len(ok)} successful requests)."
        )
    if len(ok) < 100:
        warnings.append(
            f"concurrency={concurrency} batch={batch_size}: p99 is reported for completeness "
            f"and is not stable at N={len(ok)}."
        )

    ttft_us = [record.token_times_us[0] - record.started_us for record in ok]
    e2e_us = [record.finished_us - record.started_us for record in ok]
    itl_us: list[int] = []
    decode: list[float] = []
    for record in ok:
        times = record.token_times_us
        itl_us.extend(times[index] - times[index - 1] for index in range(1, len(times)))
        if len(times) >= 2 and times[-1] > times[0]:
            decode.append((len(times) - 1) / ((times[-1] - times[0]) / 1_000_000))
    e2e_ms = [round(value / 1000.0, 3) for value in e2e_us]
    total_tokens = sum(record.output_tokens for record in ok)
    level: dict[str, object] = {
        "concurrency": concurrency,
        "batch_size": batch_size,
        "n_measured": n_measured,
        "n_warmup_excluded": len(warmup),
        "n_warmup_errors": sum(1 for record in warmup if not record.ok),
        "n_ok": len(ok),
        "n_errors": n_errors,
        "n_timeouts": n_timeouts,
        "error_rate": round(n_errors / n_measured, 6) if n_measured else None,
        "output_tokens_total": total_tokens,
        "output_tokens_mean": round(total_tokens / len(ok), 3) if ok else None,
        "measurement_window_s": round(window_us / 1_000_000, 6),
        "requests_per_sec": _rate(len(ok), window_us),
        "output_tokens_per_sec": _rate(total_tokens, window_us),
        "mean_decode_tokens_per_sec": round(mean(decode), 4) if decode else None,
        "n_decode_samples": len(decode),
        "ttft_ms": _dist(ttft_us, with_p99=False, with_tails=True),
        "itl_ms": _dist(itl_us, with_p99=False, with_tails=False),
        "e2e_ms": _dist(e2e_us, with_p99=True, with_tails=True),
        "e2e_histogram": histogram(e2e_ms),
        "e2e_cdf": empirical_cdf(e2e_ms),
    }
    return level, warnings


def _dist(values_us: list[int], *, with_p99: bool, with_tails: bool) -> dict[str, object] | None:
    if not values_us:
        return None
    samples = [round(value / 1000.0, 3) for value in values_us]
    payload: dict[str, object] = {
        "n": len(samples),
        "mean": round(mean(samples), 3),
        "p50": round(percentile(samples, 50), 3),
        "p95": round(percentile(samples, 95), 3),
    }
    if with_p99:
        payload["p99"] = round(percentile(samples, 99), 3)
    if with_tails:
        payload["min"] = round(min(samples), 3)
        payload["max"] = round(max(samples), 3)
    return payload


def _rate(numerator: float, window_us: int) -> float | None:
    if window_us <= 0:
        return None
    return round(numerator / (window_us / 1_000_000), 4)


def _error_rows(records: Iterable[RequestRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if record.ok:
            continue
        rows.append(
            {
                "request_id": record.request_id,
                "phase": record.phase,
                "ordinal": record.ordinal,
                "concurrency": record.concurrency,
                "batch_size": record.batch_size,
                "prompt_ids": list(record.prompt_ids),
                "error_type": record.error_type,
                "message": record.error_message,
            }
        )
    return rows
