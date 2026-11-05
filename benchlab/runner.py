"""Closed-loop load generator.

Latency starts when a concurrency slot is acquired and the request is issued.
Warmup runs at the same concurrency as the level that follows it, and those
requests are kept in the JSONL only.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Callable

from benchlab.backends.base import Backend
from benchlab.clock import Clock
from benchlab.config import LoadedConfig
from benchlab.prompts import Prompt
from benchlab.records import (
    ErrorType,
    GenerationResult,
    RequestContext,
    RequestRecord,
    TokenSink,
    coerce_result,
)
from benchlab.summary import build_summary
from benchlab.vram import VramReading


async def run_benchmark(
    loaded: LoadedConfig,
    backend: Backend,
    clock: Clock,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], list[RequestRecord]]:
    config = loaded.config
    order = _prompt_order(loaded.prompts, config.workload.seed)
    cursor = 0

    def take(count: int) -> list[Prompt]:
        nonlocal cursor
        chosen: list[Prompt] = []
        for _ in range(count):
            chosen.append(order[cursor % len(order)])
            cursor += 1
        return chosen

    records: list[RequestRecord] = []
    vram: VramReading | None = None
    try:
        for batch_size in config.workload.batch_sizes:
            for concurrency in config.workload.concurrency:
                if config.workload.warmup:
                    warmup = await _phase(
                        backend,
                        clock,
                        take=take,
                        phase="warmup",
                        count=config.workload.warmup,
                        concurrency=concurrency,
                        batch_size=batch_size,
                        max_tokens=config.workload.max_tokens,
                        temperature=config.workload.temperature,
                        timeout_s=config.workload.timeout_s,
                    )
                    records.extend(warmup)
                    _emit(
                        log,
                        f"[warmup] concurrency={concurrency} batch={batch_size} "
                        f"n={len(warmup)} excluded_from_stats=yes "
                        f"errors={sum(1 for record in warmup if not record.ok)}",
                    )
                measured = await _phase(
                    backend,
                    clock,
                    take=take,
                    phase="measure",
                    count=config.workload.samples,
                    concurrency=concurrency,
                    batch_size=batch_size,
                    max_tokens=config.workload.max_tokens,
                    temperature=config.workload.temperature,
                    timeout_s=config.workload.timeout_s,
                )
                records.extend(measured)
                ok = sum(1 for record in measured if record.ok)
                _emit(
                    log,
                    f"[measure] concurrency={concurrency} batch={batch_size} "
                    f"ok={ok} errors={len(measured) - ok}",
                )
        vram = await backend.vram()
    finally:
        await backend.aclose()
    if vram is None:
        raise RuntimeError("benchmark ended without a VRAM reading")

    summary = build_summary(
        loaded,
        records,
        vram=vram,
        backend_name=backend.name,
        model_id=backend.model_id,
        revision=config.backend.revision,
        synthetic=backend.synthetic,
        backend_warnings=list(backend.warnings),
    )
    return summary, records


async def _phase(
    backend: Backend,
    clock: Clock,
    *,
    take: Callable[[int], list[Prompt]],
    phase: str,
    count: int,
    concurrency: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> list[RequestRecord]:
    batches = [take(batch_size) for _ in range(count)]
    semaphore = asyncio.Semaphore(concurrency)
    slots: list[RequestRecord | None] = [None] * count

    async def one(ordinal: int) -> None:
        prompts = batches[ordinal]
        ctx = RequestContext(
            request_id=_request_id(concurrency, batch_size, phase, ordinal),
            phase=phase,
            ordinal=ordinal,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        async with semaphore:
            sink = TokenSink()
            result = await _issue(
                backend,
                clock,
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
                ctx=ctx,
                sink=sink,
            )
            slots[ordinal] = RequestRecord(
                request_id=ctx.request_id,
                phase=phase,
                ordinal=ordinal,
                concurrency=concurrency,
                batch_size=batch_size,
                prompt_ids=[prompt.id for prompt in prompts],
                prompt_categories=[prompt.category for prompt in prompts],
                ok=result.ok,
                error_type=result.error_type,
                error_message=result.error_message,
                started_us=result.started_us,
                finished_us=result.finished_us,
                token_times_us=list(result.token_times_us),
                output_text=result.output_text,
                backend=backend.name,
                model_id=backend.model_id,
            )

    await asyncio.gather(*(one(index) for index in range(count)))
    finished = [slot for slot in slots if slot is not None]
    if len(finished) != count:
        raise RuntimeError("a request finished without a record")
    return finished


async def _issue(
    backend: Backend,
    clock: Clock,
    *,
    prompts: list[Prompt],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    ctx: RequestContext,
    sink: TokenSink,
) -> GenerationResult:
    started = clock.now_us()
    timeout_us = int(round(timeout_s * 1_000_000))
    generation = asyncio.create_task(
        backend.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            clock=clock,
            ctx=ctx,
            sink=sink,
        )
    )
    timer = asyncio.create_task(clock.sleep_us(timeout_us))
    try:
        done, _pending = await asyncio.wait(
            {generation, timer}, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        generation.cancel()
        timer.cancel()
        raise

    if generation in done and not generation.cancelled():
        timer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await timer
        exc = generation.exception()
        if exc is not None:
            return GenerationResult(
                ok=False,
                error_type=ErrorType.BACKEND,
                error_message=f"{type(exc).__name__}: {exc}",
                output_text="".join(sink.texts),
                token_times_us=list(sink.times_us),
                started_us=started,
                finished_us=clock.now_us(),
            )
        result = generation.result()
        if not result.token_times_us and sink.times_us:
            result.token_times_us = list(sink.times_us)
            if not result.output_text:
                result.output_text = "".join(sink.texts)
        if result.started_us == 0:
            result.started_us = started
        return coerce_result(result, started)

    generation.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await generation
    return GenerationResult(
        ok=False,
        error_type=ErrorType.TIMEOUT,
        error_message=f"client timeout after {timeout_s}s",
        output_text="".join(sink.texts),
        token_times_us=list(sink.times_us),
        started_us=started,
        finished_us=clock.now_us(),
    )


def _prompt_order(prompts: list[Prompt], seed: int | None) -> list[Prompt]:
    order = list(prompts)
    if seed is None:
        return order
    random.Random(seed).shuffle(order)
    return order


def _request_id(concurrency: int, batch_size: int, phase: str, ordinal: int) -> str:
    code = "w" if phase == "warmup" else "m"
    return f"c{concurrency}-b{batch_size}-{code}{ordinal:03d}"


def _emit(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)
