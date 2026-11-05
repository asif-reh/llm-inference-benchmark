"""In-test backends. Not part of the installed package."""

from __future__ import annotations

from pathlib import Path

from benchlab.backends.base import Backend
from benchlab.clock import Clock
from benchlab.config import LoadedConfig, load_config
from benchlab.prompts import Prompt
from benchlab.records import GenerationResult, RequestContext, TokenSink
from benchlab.vram import VramReading

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data" / "prompts" / "example.jsonl"


class ScriptedBackend(Backend):
    """Each call pops a list of sleeps, then emits one token per sleep."""

    name = "scripted"
    synthetic = False

    def __init__(self, delays_us: list[list[int]]) -> None:
        self._delays = [list(item) for item in delays_us]
        self.model_id = "scripted"
        self.warnings = []

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
        del prompts, max_tokens, temperature, timeout_s, ctx
        started = clock.now_us()
        delays = self._delays.pop(0)
        pieces: list[str] = []
        for index, delay in enumerate(delays):
            await clock.sleep_us(delay)
            piece = f"t{index}"
            sink.add(clock.now_us(), piece)
            pieces.append(piece)
        return GenerationResult(
            ok=True,
            output_text="".join(pieces),
            token_times_us=list(sink.times_us),
            started_us=started,
            finished_us=clock.now_us(),
        )

    async def vram(self) -> VramReading:
        return VramReading.not_measurable("scripted backend")


def write_config(
    directory: Path,
    *,
    clock: str = "virtual",
    backend: str = "mock",
    concurrency: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    warmup: int = 0,
    samples: int = 2,
    timeout_s: float = 30,
    extra_backend: str = "",
) -> Path:
    concurrency = concurrency if concurrency is not None else [1]
    batch_sizes = batch_sizes if batch_sizes is not None else [1]
    path = directory / "bench.yaml"
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "run_name: unit",
                f"clock: {clock}",
                "backend:",
                f"  type: {backend}",
                "  model_id: synthetic/mock-7b",
                extra_backend.rstrip(),
                "workload:",
                f"  prompts: {PROMPTS}",
                f"  concurrency: [{', '.join(str(item) for item in concurrency)}]",
                f"  batch_sizes: [{', '.join(str(item) for item in batch_sizes)}]",
                f"  warmup: {warmup}",
                f"  samples: {samples}",
                "  max_tokens: 32",
                "  temperature: 0.0",
                "  seed: 1",
                f"  timeout_s: {timeout_s}",
                "output:",
                f"  directory: {directory / 'out'}",
                "  overwrite: true",
                "  run_id: unit",
                '  freeze_created_at: "2026-11-05T00:00:00Z"',
                "  include_output_text: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def loaded(directory: Path, **kwargs: object) -> LoadedConfig:
    path = write_config(directory, **kwargs)  # type: ignore[arg-type]
    return load_config(path)
