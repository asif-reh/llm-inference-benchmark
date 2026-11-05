"""Shared async SSE reading for HTTP backends."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from benchlab import __version__
from benchlab.backends.parsing import DONE, StreamDone, payload_from_sse_line


def make_async_client(timeout_s: float) -> httpx.AsyncClient:
    # The read budget sits above the runner's logical timeout so cancellation
    # classifies the request as a timeout instead of a socket error.
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=timeout_s + 5.0, write=30.0, pool=10.0),
        headers={"User-Agent": f"benchlab/{__version__}", "Accept": "text/event-stream"},
        follow_redirects=True,
    )


def strip_base(url: str) -> str:
    return url.rstrip("/")


async def raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    body = (await response.aread()).decode("utf-8", errors="replace")[:500]
    raise RuntimeError(f"HTTP {response.status_code}: {body}")


async def iter_sse_json(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, object]]:
    async for line in lines:
        payload = payload_from_sse_line(line)
        if payload is None:
            continue
        if isinstance(payload, StreamDone) or payload is DONE:
            return
        if isinstance(payload, dict):
            yield payload
            continue
        raise RuntimeError("SSE payload was not a JSON object")
