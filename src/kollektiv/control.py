"""Thread-safe graph-node controls for the local virtual world."""

import threading
from collections.abc import Callable
from uuid import uuid4


class RunStopped(RuntimeError):
    """A requested, resumable stop at a graph boundary."""


class RunControl:
    def __init__(self, start_paused: bool = False):
        self._condition = threading.Condition()
        self._mode = "paused" if start_paused else "running"
        self._steps = 0
        self._active: dict[str, dict] = {}
        self._waiting = 0
        self.on_change: Callable[[dict], None] | None = None

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "mode": self._mode,
                "paused": self._mode == "paused" and not self._active and not self._steps,
                "waiting": self._waiting > 0,
                "running": self._mode == "running" or bool(self._active),
                "stopped": self._mode == "stopped",
                "active_nodes": list(self._active.values()),
                "steps_pending": self._steps,
            }

    def changed(self) -> None:
        if self.on_change:
            self.on_change(self.snapshot())

    def wait_until_paused(self, timeout: float = 10) -> bool:
        """Wait for a real node boundary, useful for local server shutdown and admin edits."""
        with self._condition:
            return self._condition.wait_for(
                lambda: (
                    self._mode == "paused"
                    and not self._active
                    and not self._steps
                    and self._waiting > 0
                ),
                timeout=timeout,
            )

    def pause(self) -> None:
        with self._condition:
            if self._mode == "stopped":
                raise ValueError("This control has stopped; resume with a fresh RunControl")
            self._mode = "paused"
            self._steps = 0
            self._condition.notify_all()
        self.changed()

    def resume(self) -> None:
        with self._condition:
            if self._mode == "stopped":
                raise ValueError("This control has stopped; resume with a fresh RunControl")
            self._mode = "running"
            self._steps = 0
            self._condition.notify_all()
        self.changed()

    def step(self) -> None:
        with self._condition:
            if self._mode == "stopped":
                raise ValueError("This control has stopped; step with a fresh RunControl")
            if self._active:
                raise ValueError("Pause and wait for the active node before stepping")
            self._mode = "paused"
            self._steps += 1
            self._condition.notify_all()
        self.changed()

    def stop(self) -> None:
        with self._condition:
            self._mode = "stopped"
            self._steps = 0
            self._condition.notify_all()
        self.changed()

    def finish(self) -> None:
        """Mark a terminal runner so inspection no longer reports it as running."""
        with self._condition:
            if self._mode != "stopped":
                self._mode = "finished"
            self._steps = 0
            self._condition.notify_all()
        self.changed()

    def enter(self, node: str, *, cancel: Callable[[], bool] | None = None, **fields) -> str:
        token = uuid4().hex
        with self._condition:
            self._waiting += 1
            try:
                while self._mode == "paused" and self._steps == 0:
                    if cancel and cancel():
                        raise RunStopped("Sub-agent retired while paused")
                    self._condition.notify_all()
                    self._condition.wait(timeout=0.1 if cancel else None)
                if cancel and cancel():
                    raise RunStopped("Sub-agent retired at a graph boundary")
                if self._mode in ("stopped", "finished"):
                    raise RunStopped("Stopped at a graph boundary; progress is resumable")
                if self._mode == "paused":
                    self._steps -= 1
                self._active[token] = {"id": token, "node": node, **fields}
            finally:
                self._waiting -= 1
        self.changed()
        return token

    def leave(self, token: str) -> None:
        with self._condition:
            self._active.pop(token, None)
            self._condition.notify_all()
        self.changed()
