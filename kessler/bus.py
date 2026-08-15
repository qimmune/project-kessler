"""A tiny event bus so the UI, the console demo, and the logs all see one story."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Event:
    kind: str          # status | alert | agent1 | agent2 | tool | verdict | error
    text: str
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class Bus:
    def __init__(self, sink: Callable[[Event], None] | None = None):
        self.events: list[Event] = []
        self.sink = sink

    def emit(self, kind: str, text: str, **data) -> Event:
        ev = Event(kind=kind, text=text, data=data)
        self.events.append(ev)
        if self.sink:
            self.sink(ev)
        return ev

    def __iter__(self):
        return iter(self.events)
