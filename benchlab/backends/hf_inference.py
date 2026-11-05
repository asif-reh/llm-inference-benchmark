"""Hugging Face Inference API via ``InferenceClient`` streaming.

Timestamps are taken on the asyncio side when a token arrives from the worker
thread, so TTFT includes client scheduling. The sync client runs in a daemon
thread because current ``InferenceClient`` streaming is synchronous.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Iterable, Iterator
from typing import cast

from huggingface_hub import InferenceClient

from benchlab.backends.base import Backend, BackendProtocolError
from benchlab.backends.parsing import token_from_chat_chunk, token_from_text_generation
from benchlab.clock import Clock
from benchlab.config import HuggingFaceBackendConfig
from benchlab.prompts import Prompt
from benchlab.records import ErrorType, GenerationResult, RequestContext, TokenSink
from benchlab.vram import VramReading


class HuggingFaceBackend(Backend):
    name = "huggingface"
    synthetic = False

    def __init__(
        self, config: HuggingFaceBackendConfig, *, seed: int | None, timeout_s: float
    ) -> None:
        self.model_id = config.model_id
        self._mode = config.mode
        self._seed = seed
        self._timeout_s = timeout_s
        token = os.environ.get(config.token_env) or None
        self.warnings = []
        if token is None:
            self.warnings.append(
                f"{config.token_env} is unset. The call will be anonymous and may be rejected."
            )
        if config.revision is not None:
            self.warnings.append(
                "revision is recorded in the run metadata only. InferenceClient in this "
                "environment does not pin a revision on the serverless Inference API."
            )
        else:
            self.warnings.append(
                "No model revision is pinned. The serverless API may move the default revision."
            )
        self.warnings.append(
            "Chat mode counts delta.content only. Reasoning-channel text is ignored "
            "when the provider does not put it in content."
            if config.mode == "chat"
            else "Text mode counts non-special text-generation stream pieces."
        )
        self._client = InferenceClient(
            model=config.model_id,
            token=token,
            timeout=timeout_s,
            provider=config.provider,
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
        del ctx
        if len(prompts) != 1:
            raise BackendProtocolError("the Hugging Face backend sends one prompt per request")
        started = clock.now_us()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        stop = threading.Event()

        def worker() -> None:
            try:
                for text in self._iterate(prompts[0].text, max_tokens, temperature, stop):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", text))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", ""))
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        thread = threading.Thread(target=worker, name="benchlab-hf", daemon=True)
        thread.start()
        texts: list[str] = []
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=timeout_s)
                except TimeoutError:
                    stop.set()
                    return GenerationResult(
                        ok=False,
                        error_type=ErrorType.TIMEOUT,
                        error_message=f"client timeout after {timeout_s}s",
                        output_text="".join(texts),
                        token_times_us=list(sink.times_us),
                        started_us=started,
                        finished_us=clock.now_us(),
                    )
                if kind == "token":
                    text = str(payload)
                    texts.append(text)
                    sink.add(clock.now_us(), text)
                    continue
                if kind == "error":
                    message = _format_error(payload)
                    error_type = (
                        ErrorType.TIMEOUT if "timeout" in message.lower() else ErrorType.BACKEND
                    )
                    return GenerationResult(
                        ok=False,
                        error_type=error_type,
                        error_message=message,
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
        finally:
            stop.set()

    async def vram(self) -> VramReading:
        return VramReading.not_measurable(
            "Hugging Face Inference API does not expose server VRAM to the client. "
            "A probe of this machine would describe the wrong device."
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _iterate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: threading.Event,
    ) -> Iterator[str]:
        if self._mode == "text":
            chunks: Iterable[object] = self._text_stream(prompt, max_tokens, temperature)
            for chunk in chunks:
                if stop.is_set():
                    return
                text = token_from_text_generation(chunk)
                if text:
                    yield text
            return
        for chunk in self._chat_stream(prompt, max_tokens, temperature):
            if stop.is_set():
                return
            text = token_from_chat_chunk(chunk)
            if text:
                yield text

    def _text_stream(self, prompt: str, max_tokens: int, temperature: float) -> Iterable[object]:
        greedy = temperature == 0
        if self._seed is None:
            result = self._client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                stream=True,
                details=True,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
            )
        else:
            result = self._client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                stream=True,
                details=True,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
                seed=self._seed,
            )
        return _as_stream(result, "text_generation")

    def _chat_stream(self, prompt: str, max_tokens: int, temperature: float) -> Iterable[object]:
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        if self._seed is None:
            result = self._client.chat_completion(
                messages,
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
            )
        else:
            result = self._client.chat_completion(
                messages,
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
                seed=self._seed,
            )
        return _as_stream(result, "chat_completion")


def _as_stream(result: object, method: str) -> Iterable[object]:
    if isinstance(result, str) or not hasattr(result, "__iter__"):
        raise BackendProtocolError(f"{method} did not return a stream")
    return cast(Iterable[object], result)


def _format_error(payload: object) -> str:
    if isinstance(payload, BaseException):
        return f"{type(payload).__name__}: {payload}"
    return str(payload)
