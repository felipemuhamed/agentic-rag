import json
from dataclasses import dataclass
from typing import Literal


EventName = Literal["status", "sources", "token", "done", "error"]


@dataclass(frozen=True, slots=True)
class StreamEvent:
    event: EventName
    data: dict[str, object]

    def to_sse(self) -> str:
        payload = json.dumps(
            self.data,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"event: {self.event}\ndata: {payload}\n\n"
