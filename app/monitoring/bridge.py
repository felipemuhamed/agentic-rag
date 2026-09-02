"""Call async code from the synchronous pipeline.

Every node in the graph is a plain function and the API drives the stream from a
worker thread, while the MCP client is async-only: its tools raise
`NotImplementedError` on `invoke()` and answer only to `ainvoke()`. Rather than
converting the whole pipeline, this owns one event loop on a daemon thread and
submits coroutines to it.

The loop is created lazily and shared, because an MCP session is opened per call
and paying for a fresh loop each time would add latency to every question.
"""

import asyncio
import logging
from collections.abc import Coroutine
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Lock, Thread
from typing import Any, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncBridgeClosed(RuntimeError):
    pass


class AsyncBridge:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._lock = Lock()
        self._closed = False

    def run(self, coroutine: Coroutine[Any, Any, T], *, timeout: float) -> T:
        """Run a coroutine on the background loop and wait for its result.

        A timeout cancels the coroutine before raising, so a hung MCP call does
        not leak a task that keeps the loop busy for the life of the process.
        """
        if timeout <= 0:
            coroutine.close()
            raise ValueError("timeout must be positive")

        try:
            loop = self._ensure_loop()
        except BaseException:
            # Closing keeps a refused submission from warning about a coroutine
            # that was created and never awaited.
            coroutine.close()
            raise
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError(
                f"monitoring call exceeded {timeout:.1f}s"
            ) from error

    def close(self) -> None:
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop, self._thread = None, None
            self._closed = True
        if loop is None:
            return

        def _drain() -> None:
            # Cancel before stopping: a task still pending when the loop closes
            # is destroyed mid-flight and complains on stderr.
            for task in asyncio.all_tasks(loop):
                task.cancel()
            # Stopping in this same callback would prevent the loop from ever
            # delivering the cancellations it was just asked for.
            loop.call_later(0.05, loop.stop)

        loop.call_soon_threadsafe(_drain)
        if thread is not None:
            thread.join(timeout=5)
        loop.close()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise AsyncBridgeClosed("the async bridge is closed")
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()
            thread = Thread(
                target=loop.run_forever,
                name="monitoring-bridge",
                daemon=True,
            )
            thread.start()
            self._loop, self._thread = loop, thread
            return loop
