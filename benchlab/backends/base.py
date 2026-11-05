"""Adapter interface for a streaming generation backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchlab.clock import Clock
from benchlab.prompts import Prompt
from benchlab.records import GenerationResult, RequestContext, TokenSink
from benchlab.vram import VramReading


class BackendProtocolError(RuntimeError):
    """The server returned a payload this client does not accept."""


class Backend(ABC):
    """One streaming call is one measured request, possibly containing a batch.

    Implementations append every emitted piece to ``sink`` as the client
    observes it, and return the same timestamps on the ``GenerationResult``.
    They must not swallow ``CancelledError``; the runner uses cancellation for
    the client timeout and then reads ``sink`` for partial tokens.
    """

    name: str
    model_id: str
    synthetic: bool = False
    warnings: list[str]

    @abstractmethod
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
        """Generate ``prompts`` as a single client request."""

    async def vram(self) -> VramReading:
        return VramReading.not_measurable(
            "This backend does not expose accelerator memory to the client."
        )

    async def aclose(self) -> None:
        return None
