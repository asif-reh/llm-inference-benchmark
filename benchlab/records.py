"""Per-request records and the in-flight token sink."""

from __future__ import annotations

from dataclasses import dataclass, field


class ErrorType:
    TIMEOUT = "timeout"
    BACKEND = "backend"
    EMPTY_STREAM = "empty_stream"


@dataclass
class TokenSink:
    """Tokens observed so far. The timeout path reads this after cancellation."""

    times_us: list[int] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)

    def add(self, timestamp_us: int, text: str) -> None:
        self.times_us.append(timestamp_us)
        self.texts.append(text)


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    phase: str
    ordinal: int
    concurrency: int
    batch_size: int


@dataclass
class GenerationResult:
    ok: bool
    error_type: str | None = None
    error_message: str | None = None
    output_text: str = ""
    token_times_us: list[int] = field(default_factory=list)
    started_us: int = 0
    finished_us: int = 0


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    phase: str
    ordinal: int
    concurrency: int
    batch_size: int
    prompt_ids: list[str]
    prompt_categories: list[str]
    ok: bool
    error_type: str | None
    error_message: str | None
    started_us: int
    finished_us: int
    token_times_us: list[int]
    output_text: str
    backend: str
    model_id: str

    @property
    def output_tokens(self) -> int:
        return len(self.token_times_us) if self.ok else 0


def coerce_result(result: GenerationResult, started_us: int) -> GenerationResult:
    """Fill clocks and turn a token-less success into ``empty_stream``."""
    started = result.started_us or started_us
    times = list(result.token_times_us)
    finished = result.finished_us or started
    if times:
        finished = max(finished, times[-1])
    if finished < started:
        finished = started
    ok = result.ok
    error_type = result.error_type
    error_message = result.error_message
    if ok and not times:
        ok = False
        error_type = ErrorType.EMPTY_STREAM
        error_message = error_message or "stream completed without tokens"
    if ok and any(times[index] < times[index - 1] for index in range(1, len(times))):
        ok = False
        error_type = ErrorType.BACKEND
        error_message = "token timestamps went backwards"
    return GenerationResult(
        ok=ok,
        error_type=error_type,
        error_message=error_message,
        output_text=result.output_text,
        token_times_us=times,
        started_us=started,
        finished_us=finished,
    )
