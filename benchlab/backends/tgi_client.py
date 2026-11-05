"""Thin client for a Text Generation Inference ``/generate_stream`` server.

One input per request. TGI's batch budget is a server flag
(``--max-batch-total-tokens`` and related settings).
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
from benchlab.backends.parsing import text_from_tgi
from benchlab.clock import Clock
from benchlab.prompts import Prompt
from benchlab.records import ErrorType, GenerationResult, RequestContext, TokenSink
from benchlab.vram import VramReading


class TgiBackend(Backend):
    name = "tgi"
    synthetic = False

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        base_url: str,
        timeout_s: float,
        seed: int | None,
    ) -> None:
        self.model_id = model_id
        self._seed = seed
        self._base = strip_base(base_url)
        self._client = make_async_client(timeout_s)
        self.warnings = [
            "TGI client sends one input per /generate_stream request. "
            "Token budgets and continuous batching are server configuration.",
            "VRAM is not read from the client. "
            "Record it on the TGI host if you publish a GPU result.",
        ]
        if revision is None:
            self.warnings.append(
                "No revision is pinned. Record the weights the TGI process loaded."
            )
        else:
            self.warnings.append(
                "revision is stored in the run metadata only. "
                "This client cannot ask TGI to reload a revision."
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
            raise BackendProtocolError("the TGI client sends one input per request")
        started = clock.now_us()
        parameters: dict[str, object] = {"max_new_tokens": max_tokens}
        if temperature == 0:
            parameters["do_sample"] = False
        else:
            parameters["do_sample"] = True
            parameters["temperature"] = temperature
        if self._seed is not None:
            parameters["seed"] = self._seed
        body = {"inputs": prompts[0].text, "parameters": parameters}
        texts: list[str] = []
        try:
            async with self._client.stream(
                "POST", f"{self._base}/generate_stream", json=body
            ) as response:
                await raise_for_status(response)
                async for event in iter_sse_json(response.aiter_lines()):
                    text = text_from_tgi(event)
                    if text:
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
            "TGI VRAM has to be measured on the server host. This client does not probe it."
        )

    async def aclose(self) -> None:
        await self._client.aclose()
