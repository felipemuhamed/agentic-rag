import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.settings import Settings


class CheckpointerHandle:
    def __init__(self, config: Settings):
        self._config = config
        self._context: AbstractContextManager[Any] | None = None
        self._checkpointer: Any | None = None
        self._lock = Lock()

    def start(self) -> Any:
        with self._lock:
            if self._checkpointer is not None:
                return self._checkpointer
            if self._config.checkpointer_mode == "memory":
                self._checkpointer = InMemorySaver()
                return self._checkpointer

            checkpoint_path = Path(
                self._config.checkpointer_path
            ).expanduser().resolve()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            context = _open_sqlite_checkpointer(checkpoint_path)
            checkpointer = context.__enter__()
            try:
                checkpointer.conn.execute("PRAGMA journal_mode=WAL")
                checkpointer.conn.execute("PRAGMA busy_timeout=5000")
                checkpointer.setup()
            except Exception:
                context.__exit__(None, None, None)
                raise
            self._context = context
            self._checkpointer = checkpointer
            return checkpointer

    def close(self) -> None:
        with self._lock:
            context = self._context
            self._context = None
            self._checkpointer = None
        if context is not None:
            context.__exit__(None, None, None)

    @property
    def started(self) -> bool:
        return self._checkpointer is not None


@contextmanager
def _open_sqlite_checkpointer(path: Path) -> Iterator[Any]:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(str(path), check_same_thread=False)
    with closing(connection):
        yield SqliteSaver(
            connection,
            serde=JsonPlusSerializer(allowed_msgpack_modules=None),
        )
