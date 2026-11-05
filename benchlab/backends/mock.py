"""Deterministic latency model for offline runs.

The arrival times are a function of the seed, the prompt, and the concurrency
level. They are not measurements of a model server. Callers must label the
summary with ``SYNTHETIC_NOTICE``.
"""

from __future__ import annotations

import random

from benchlab.backends.base import Backend
from benchlab.clock import Clock
from benchlab.config import InjectedError, MockBackendConfig
from benchlab.definitions import SYNTHETIC_NOTICE
from benchlab.prompts import Prompt
from benchlab.records import ErrorType, GenerationResult, RequestContext, TokenSink
from benchlab.vram import VramReading

# Target streamed-piece counts before ``max_tokens`` is applied, per prompt.
_BASE_TOKENS = {
    "short_qa": 20,
    "long_context": 48,
    "code": 36,
    "reasoning": 56,
}


def plan_prompt(
    prompt: Prompt,
    *,
    concurrency: int,
    seed: int,
    ordinal: int,
    phase: str,
    max_tokens: int,
) -> tuple[int, list[int]]:
    """Return ``(ttft_us, inter_token_gaps_us)`` for one prompt.

    Token count is ``len(gaps) + 1``. The same arguments always return the
    same plan. Higher concurrency increases TTFT and inter-token gaps so a
    sweep is not a flat line. That slope is part of the model, not a result.
    """
    rng = random.Random(f"{seed}|{phase}|{concurrency}|{ordinal}|{prompt.id}")
    base = _BASE_TOKENS.get(prompt.category, 32)
    count = max(8, base + rng.randint(-4, 4))
    count = min(count, max_tokens)
    ttft = 30_000 + concurrency * 3_500 + min(len(prompt.text), 4_000) * 8 + rng.randint(0, 4_000)
    gaps = [9_000 + concurrency * 250 + rng.randint(0, 1_500) for _ in range(max(count - 1, 0))]
    return ttft, gaps


def plan_batch(
    prompts: list[Prompt],
    *,
    concurrency: int,
    batch_size: int,
    seed: int,
    ordinal: int,
    phase: str,
    max_tokens: int,
) -> tuple[int, list[int]]:
    """Fuse a client batch into one stream.

    The first prompt pays a small extra TTFT per extra prompt in the batch.
    Later prompts are appended; the gap before each later prompt is that
    prompt's own TTFT. This is a simulation of serialized client batching.
    """
    if not prompts:
        raise ValueError("batch is empty")
    ttft, gaps = plan_prompt(
        prompts[0],
        concurrency=concurrency,
        seed=seed,
        ordinal=ordinal,
        phase=phase,
        max_tokens=max_tokens,
    )
    ttft += 400 * (batch_size - 1)
    for index, prompt in enumerate(prompts[1:], start=1):
        next_ttft, next_gaps = plan_prompt(
            prompt,
            concurrency=concurrency,
            seed=seed,
            ordinal=ordinal,
            phase=f"{phase}:batch{index}",
            max_tokens=max_tokens,
        )
        gaps.append(next_ttft)
        gaps.extend(next_gaps)
    return ttft, gaps


class MockBackend(Backend):
    name = "mock"
    synthetic = True

    def __init__(self, config: MockBackendConfig, *, seed: int) -> None:
        self.model_id = config.model_id
        self._revision = config.revision
        self._seed = seed
        self._errors = list(config.injected_errors)
        self.warnings = [
            SYNTHETIC_NOTICE
            + ". Token arrival times come from a discrete-event model, not from a server."
        ]

    async def generate(
        self,
        prompts: list[Prompt],
        *,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        clock: Clock,
        ctx: RequestContext,
        sink: TokenSink,
    ) -> GenerationResult:
        del temperature, timeout_s  # recorded on the run; the latency model does not use them
        started = clock.now_us()
        injected = self._match(ctx)
        if injected is not None and injected.error_type == ErrorType.BACKEND:
            await clock.sleep_us(5_000)
            return GenerationResult(
                ok=False,
                error_type=ErrorType.BACKEND,
                error_message=injected.message,
                started_us=started,
                finished_us=clock.now_us(),
            )
        if injected is not None and injected.error_type == ErrorType.EMPTY_STREAM:
            await clock.sleep_us(5_000)
            return GenerationResult(
                ok=True,
                error_message=injected.message,
                started_us=started,
                finished_us=clock.now_us(),
            )
        if injected is not None and injected.error_type == ErrorType.TIMEOUT:
            # Return promptly with the configured message. A separate test covers the
            # runner's wall/virtual timeout, which cancels a backend that never returns.
            await clock.sleep_us(8_000)
            return GenerationResult(
                ok=False,
                error_type=ErrorType.TIMEOUT,
                error_message=injected.message,
                started_us=started,
                finished_us=clock.now_us(),
            )

        ttft, gaps = plan_batch(
            prompts,
            concurrency=ctx.concurrency,
            batch_size=ctx.batch_size,
            seed=self._seed,
            ordinal=ctx.ordinal,
            phase=ctx.phase,
            max_tokens=max_tokens,
        )
        await clock.sleep_us(ttft)
        pieces: list[str] = []
        token_count = len(gaps) + 1
        for index in range(token_count):
            piece = f"tok{index:02d}"
            sink.add(clock.now_us(), piece)
            pieces.append(piece)
            if index < len(gaps):
                await clock.sleep_us(gaps[index])
        return GenerationResult(
            ok=True,
            output_text="".join(pieces),
            token_times_us=list(sink.times_us),
            started_us=started,
            finished_us=clock.now_us(),
        )

    async def vram(self) -> VramReading:
        return VramReading.not_measurable(
            "Mock backend does not execute a model. VRAM is not applicable."
        )

    async def aclose(self) -> None:
        return None

    def _match(self, ctx: RequestContext) -> InjectedError | None:
        for item in self._errors:
            if (
                item.phase == ctx.phase
                and item.concurrency == ctx.concurrency
                and item.batch_size == ctx.batch_size
                and item.ordinal == ctx.ordinal
            ):
                return item
        return None
