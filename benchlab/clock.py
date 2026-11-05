"""Run clocks for benchmark timestamps.

``RealClock`` follows ``time.perf_counter``. ``VirtualClock`` is a discrete-event
clock used by the mock backend so a synthetic run is deterministic and does not
sleep in wall time. Virtual time is an integer number of microseconds.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, cast

T = TypeVar("T")


class Clock(Protocol):
    """Monotonic microsecond clock shared by the runner and the backend."""

    def now_us(self) -> int:
        """Return microseconds since the clock was created."""

    async def sleep_us(self, us: int) -> None:
        """Wait until ``us`` microseconds have passed on this clock."""


class RealClock:
    """Wall-time clock based on ``perf_counter``. Not comparable across processes."""

    def __init__(self) -> None:
        self._origin_ns = time.perf_counter_ns()

    def now_us(self) -> int:
        return (time.perf_counter_ns() - self._origin_ns) // 1_000

    async def sleep_us(self, us: int) -> None:
        if us < 0:
            raise ValueError("sleep duration must be >= 0")
        await asyncio.sleep(us / 1_000_000)


@dataclass(order=True)
class _Wake:
    when: int
    seq: int
    fut: asyncio.Future[None] = field(compare=False)


class VirtualClock:
    """Discrete-event clock.

    Sleeps record an absolute wake time. A driver (see ``run_with_clock``) advances
    time only when every runnable task is blocked on this clock. That lets
    concurrent requests overlap in simulated time.
    """

    def __init__(self) -> None:
        self._t = 0
        self._seq = 0
        self._heap: list[_Wake] = []

    def now_us(self) -> int:
        return self._t

    async def sleep_us(self, us: int) -> None:
        if us < 0:
            raise ValueError("sleep duration must be >= 0")
        if us == 0:
            await asyncio.sleep(0)
            return
        fut = self._schedule(us)
        try:
            await fut
        except asyncio.CancelledError:
            if not fut.done():
                fut.cancel()
            raise

    def _schedule(self, us: int) -> asyncio.Future[None]:
        loop = asyncio.get_running_loop()
        fut = cast(asyncio.Future[None], loop.create_future())
        self._seq += 1
        heapq.heappush(self._heap, _Wake(self._t + us, self._seq, fut))
        return fut

    def advance_next(self) -> bool:
        """Wake the earliest pending sleeper. Return False if nobody is waiting."""
        while self._heap and self._heap[0].fut.done():
            heapq.heappop(self._heap)
        if not self._heap:
            return False
        wake = heapq.heappop(self._heap)
        if wake.when > self._t:
            self._t = wake.when
        if not wake.fut.done():
            wake.fut.set_result(None)
        return True


async def _drain_ready(max_rounds: int = 100_000) -> None:
    """Run queued callbacks until the loop has nothing ready.

    The virtual clock must not advance while a task is still runnable, or a
    request that has not yet registered its sleep would start late. ``_ready``
    is a private asyncio attribute; it is the reliable signal that the ready
    queue is empty, which the public API does not expose.
    """
    loop = cast(asyncio.BaseEventLoop, asyncio.get_running_loop())
    for _ in range(max_rounds):
        ready = loop._ready  # type: ignore[attr-defined]
        if len(ready) == 0:
            return
        await asyncio.sleep(0)
    raise RuntimeError("event loop did not drain; a task may be spinning")


async def run_with_clock(clock: Clock, main: Callable[[Clock], Coroutine[None, None, T]]) -> T:
    """Run ``main(clock)``, driving virtual time when ``clock`` is virtual."""
    if not isinstance(clock, VirtualClock):
        return await main(clock)

    task: asyncio.Task[T] = asyncio.create_task(main(clock))
    steps = 0
    try:
        while not task.done():
            await _drain_ready()
            if task.done():
                break
            if clock.advance_next():
                steps += 1
                if steps > 5_000_000:
                    raise RuntimeError("virtual clock exceeded 5_000_000 steps")
                continue
            await _drain_ready()
            if task.done():
                break
            if not clock.advance_next():
                pending = [t for t in asyncio.all_tasks() if not t.done()]
                raise RuntimeError(
                    "virtual clock stalled with "
                    f"{len(pending)} unfinished tasks waiting on non-simulated I/O"
                )
        return await task
    except BaseException:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        raise
