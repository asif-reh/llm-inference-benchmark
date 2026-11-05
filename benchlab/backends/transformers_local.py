"""HTTP client for ``scripts/serve_transformers.py``.

The reference server is not a production scheduler. See the script docstring.
Client batches are sent as one request; the server generates them in order.
"""

from __future__ import annotations

import asyncio

import httpx

from benchlab.backends.base import Backend, BackendProtocolError
from benchlab.backends.http_sse import (
    iter_sse_json,
    make_async_client,
    raise_for_status,
    strip_base,
)
from benchlab.clock import Clock
from benchlab.prompts import Prompt
from benchlab.records import ErrorType, GenerationResult, RequestContext, TokenSink
from benchlab.vram import VramReading


class TransformersBackend(Backend):
    name = "transformers"
    synthetic = False

    def __init__(
        self, *, model_id: str, revision: str | None, base_url: str, timeout_s: float
    ) -> None:
        self.model_id = model_id
        self._revision = revision
        self._base = strip_base(base_url)
        self._client = make_async_client(timeout_s)
        self.warnings = [
            "The reference transformers server serializes model.generate with a lock. "
            "Concurrency includes queueing on that lock.",
        ]
        if revision is None:
            self.warnings.append(
                "No revision is pinned in the client. Pass --revision on the server "
                "and record that value before publishing."
            )
        else:
            self.warnings.append(
                "revision is included in the request body. The server process must "
                "actually have loaded that revision."
            )

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
        del timeout_s, ctx
        started = clock.now_us()
        payload = {
            "model_id": self.model_id,
            "revision": self._revision,
            "prompts": [prompt.text for prompt in prompts],
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }
        texts: list[str] = []
        try:
            async with self._client.stream(
                "POST", f"{self._base}/generate", json=payload
            ) as response:
                await raise_for_status(response)
                async for event in iter_sse_json(response.aiter_lines()):
                    if event.get("error"):
                        raise BackendProtocolError(str(event["error"]))
                    if event.get("done") is True:
                        break
                    text = event.get("text")
                    if isinstance(text, str) and text:
                        texts.append(text)
                        sink.add(clock.now_us(), text)
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, BackendProtocolError, RuntimeError, ValueError) as exc:
            return _backend_failure(exc, texts, sink, started, clock.now_us())
        return GenerationResult(
            ok=True,
            output_text="".join(texts),
            token_times_us=list(sink.times_us),
            started_us=started,
            finished_us=clock.now_us(),
        )

    async def vram(self) -> VramReading:
        try:
            response = await self._client.get(f"{self._base}/vram")
        except httpx.HTTPError as exc:
            return VramReading.not_measurable(
                f"The transformers server /vram endpoint was not reachable ({exc})."
            )
        if response.status_code == 404:
            return VramReading.not_measurable("The transformers server has no /vram endpoint.")
        if response.status_code >= 400:
            return VramReading.not_measurable(
                f"The transformers server /vram endpoint returned HTTP {response.status_code}."
            )
        payload = response.json()
        if not isinstance(payload, dict):
            return VramReading.not_measurable(
                "The transformers server /vram payload was not an object."
            )
        if payload.get("measurable") is not True:
            note = payload.get("note")
            return VramReading.not_measurable(
                str(note) if isinstance(note, str) else "Server reported VRAM as not measurable."
            )
        source = payload.get("source")
        note = payload.get("note")
        return VramReading(
            measurable=True,
            peak_allocated_mb=_optional_float(payload.get("peak_allocated_mb")),
            instantaneous_used_mb=_optional_float(payload.get("instantaneous_used_mb")),
            source=source if isinstance(source, str) else "server",
            note=note if isinstance(note, str) else "Reported by the server process.",
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _backend_failure(
    exc: BaseException,
    texts: list[str],
    sink: TokenSink,
    started_us: int,
    finished_us: int,
) -> GenerationResult:
    return GenerationResult(
        ok=False,
        error_type=ErrorType.BACKEND,
        error_message=f"{type(exc).__name__}: {exc}",
        output_text="".join(texts),
        token_times_us=list(sink.times_us),
        started_us=started_us,
        finished_us=finished_us,
    )
