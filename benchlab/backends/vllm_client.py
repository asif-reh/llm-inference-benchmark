"""Thin client for a vLLM OpenAI-compatible ``/v1/completions`` stream.

One prompt per request. Server-side batching is whatever you started vLLM with
(``--max-num-seqs`` and related flags). This client does not pretend to
configure that.
"""

from __future__ import annotations

import asyncio
import os

import httpx

from benchlab.backends.base import Backend, BackendProtocolError
from benchlab.backends.http_sse import (
    iter_sse_json,
    make_async_client,
    raise_for_status,
    strip_base,
)
from benchlab.backends.parsing import texts_from_completion_chunk
from benchlab.clock import Clock
from benchlab.prompts import Prompt
from benchlab.records import ErrorType, GenerationResult, RequestContext, TokenSink
from benchlab.vram import VramReading


class VllmBackend(Backend):
    name = "vllm"
    synthetic = False

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        base_url: str,
        api_key_env: str,
        timeout_s: float,
        seed: int | None,
    ) -> None:
        self.model_id = model_id
        self._seed = seed
        self._base = strip_base(base_url)
        self._client = make_async_client(timeout_s)
        api_key = os.environ.get(api_key_env) or None
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self.warnings = [
            "vLLM client sends one prompt per request to /v1/completions. "
            "Batching and paged attention are properties of the server process, not this client.",
            "VRAM is not read from the client. Record nvidia-smi or the vLLM metrics "
            "endpoint on the server host if you publish a GPU result.",
        ]
        if revision is None:
            self.warnings.append("No revision is pinned. Record the weights the server loaded.")
        else:
            self.warnings.append(
                "revision is stored in the run metadata only. "
                "This client cannot ask vLLM to reload a revision."
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
        if len(prompts) != 1:
            raise BackendProtocolError("the vLLM client sends one prompt per request")
        started = clock.now_us()
        body: dict[str, object] = {
            "model": self.model_id,
            "prompt": prompts[0].text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if self._seed is not None:
            body["seed"] = self._seed
        texts: list[str] = []
        try:
            async with self._client.stream(
                "POST",
                f"{self._base}/v1/completions",
                json=body,
                headers=self._headers,
            ) as response:
                await raise_for_status(response)
                async for event in iter_sse_json(response.aiter_lines()):
                    for text in texts_from_completion_chunk(event):
                        texts.append(text)
                        sink.add(clock.now_us(), text)
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, BackendProtocolError, RuntimeError, ValueError) as exc:
            return GenerationResult(
                ok=False,
                error_type=ErrorType.BACKEND,
                error_message=f"{type(exc).__name__}: {exc}",
                output_text="".join(texts),
                token_times_us=list(sink.times_us),
                started_us=started,
                finished_us=clock.now_us(),
            )
        return GenerationResult(
            ok=True,
            output_text="".join(texts),
            token_times_us=list(sink.times_us),
            started_us=started,
            finished_us=clock.now_us(),
        )

    async def vram(self) -> VramReading:
        return VramReading.not_measurable(
            "vLLM VRAM has to be measured on the server host. This client does not probe it."
        )

    async def aclose(self) -> None:
        await self._client.aclose()
