from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from benchlab.backends.mock import plan_batch, plan_prompt
from benchlab.clock import VirtualClock, run_with_clock
from benchlab.export import write_run
from benchlab.prompts import Prompt
from benchlab.runner import run_benchmark
from tests.support import PROMPTS, ScriptedBackend, loaded


def test_plan_is_deterministic_and_respects_max_tokens() -> None:
    prompt = Prompt(id="qa", category="short_qa", text="hello")
    kwargs = {"concurrency": 2, "seed": 7, "ordinal": 1, "phase": "measure", "max_tokens": 64}
    assert plan_prompt(prompt, **kwargs) == plan_prompt(prompt, **kwargs)
    low = plan_prompt(prompt, concurrency=1, seed=7, ordinal=1, phase="measure", max_tokens=64)
    high = plan_prompt(prompt, concurrency=8, seed=7, ordinal=1, phase="measure", max_tokens=64)
    assert high[0] > low[0]
    capped_ttft, capped_gaps = plan_prompt(
        prompt, concurrency=1, seed=7, ordinal=1, phase="measure", max_tokens=8
    )
    assert len(capped_gaps) + 1 == 8
    assert capped_ttft > 0
    batch_ttft, batch_gaps = plan_batch(
        [prompt, prompt],
        concurrency=1,
        batch_size=2,
        seed=7,
        ordinal=1,
        phase="measure",
        max_tokens=8,
    )
    assert len(batch_gaps) + 1 == 16
    assert batch_ttft > capped_ttft


def test_closed_loop_overlaps_on_the_virtual_clock(tmp_path: Path) -> None:
    config = loaded(tmp_path, concurrency=[2], samples=3, warmup=0, timeout_s=30)
    backend = ScriptedBackend([[1_000], [5_000], [1_000]])

    async def run(clock: VirtualClock) -> tuple[dict[str, object], list[object]]:
        summary, records = await run_benchmark(config, backend, clock)
        return summary, records

    summary, records = asyncio.run(run_with_clock(VirtualClock(), run))
    measured = [record for record in records if record.phase == "measure"]
    assert sorted(record.started_us for record in measured) == [0, 0, 1_000]
    assert sorted(record.finished_us for record in measured) == [1_000, 2_000, 5_000]
    levels = summary["levels"]
    assert isinstance(levels, list) and len(levels) == 1
    level = levels[0]
    assert isinstance(level, dict)
    assert level["measurement_window_s"] == 0.005
    assert level["n_ok"] == 3
    assert level["requests_per_sec"] == 600


def test_runner_timeout_excludes_the_request_from_latency(tmp_path: Path) -> None:
    config = loaded(tmp_path, samples=1, warmup=1, timeout_s=0.01)
    backend = ScriptedBackend([[1_000], [5_000_000]])

    async def run(clock: VirtualClock) -> tuple[dict[str, object], list[object]]:
        return await run_benchmark(config, backend, clock)

    summary, records = asyncio.run(run_with_clock(VirtualClock(), run))
    warmup = [record for record in records if record.phase == "warmup"]
    measured = [record for record in records if record.phase == "measure"]
    assert warmup[0].ok is True
    assert measured[0].ok is False
    assert measured[0].error_type == "timeout"
    assert measured[0].token_times_us == []
    levels = summary["levels"]
    assert isinstance(levels, list)
    level = levels[0]
    assert isinstance(level, dict)
    assert level["n_measured"] == 1
    assert level["n_warmup_excluded"] == 1
    assert level["n_timeouts"] == 1
    assert level["ttft_ms"] is None
    assert level["error_rate"] == 1


def test_mock_run_exports_and_is_deterministic(tmp_path: Path) -> None:
    from benchlab.backends.registry import create_backend

    config = loaded(
        tmp_path,
        concurrency=[1, 2],
        batch_sizes=[1],
        warmup=1,
        samples=4,
        timeout_s=2,
    )
    first_summary, first_records = asyncio.run(
        run_with_clock(
            VirtualClock(), lambda clock: run_benchmark(config, create_backend(config), clock)
        )
    )
    second_summary, second_records = asyncio.run(
        run_with_clock(
            VirtualClock(), lambda clock: run_benchmark(config, create_backend(config), clock)
        )
    )
    assert first_summary == second_summary
    assert [record.token_times_us for record in first_records] == [
        record.token_times_us for record in second_records
    ]
    assert first_summary["notice"] == "SYNTHETIC EXAMPLE — NOT A REAL BENCHMARK"
    levels = first_summary["levels"]
    assert isinstance(levels, list)
    for level in levels:
        assert isinstance(level, dict)
        assert level["n_measured"] == 4
        assert level["n_warmup_excluded"] == 1
    out = tmp_path / "export"
    write_run(
        out,
        first_summary,
        first_records,
        copies=[tmp_path / "copy" / "summary.json"],
        overwrite=False,
        include_output_text=True,
    )
    lines = (out / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    phases = [json.loads(line)["phase"] for line in lines]
    assert phases.count("warmup") == 2
    assert phases.count("measure") == 8
    assert json.loads((tmp_path / "copy" / "summary.json").read_text())["notice"].startswith(
        "SYNTHETIC"
    )
    with pytest.raises(FileExistsError):
        write_run(
            out,
            first_summary,
            first_records,
            copies=[],
            overwrite=False,
            include_output_text=True,
        )


def test_prompt_file_exists() -> None:
    assert PROMPTS.is_file()
