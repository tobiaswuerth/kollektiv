"""Run isolation, SQLite checkpoints, restart recovery and inspectable exports."""

import json
import os
import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import StateSnapshot

from .config import Settings
from .control import RunControl, RunStopped
from .llm import json_text
from .models import Plan, Task
from .tools import record
from .workflow import Workflow
from .workspace import Workspace, atomic_write


class RunBusyError(RuntimeError):
    pass


@contextmanager
def run_lock(path: Path):
    """OS-held lock; an abandoned file does not prevent recovery after a crash."""
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RunBusyError(f"Another process is working on {path.parent}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Collective:
    def __init__(self, directory: Path, settings: Settings, manifest: dict, models=None):
        self.directory = directory.resolve()
        self.settings = settings
        self.manifest = manifest
        self.models = models
        self.workspace = Workspace(self.directory / "artifacts", settings.max_file_bytes)
        self._trace_lock = threading.Lock()
        self._on_event: Callable[[dict], None] | None = None
        self._running_done = threading.Event()
        self._running_done.set()
        self._run_mutex = threading.Lock()
        self._active_control: RunControl | None = None
        self._database_init_lock = threading.Lock()

    @classmethod
    def create(
        cls,
        goal: str,
        runs_dir: str | Path = "runs",
        settings: Settings | None = None,
        models=None,
    ) -> "Collective":
        if not goal.strip():
            raise ValueError("The goal cannot be empty")
        settings = settings or Settings()
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        directory = Path(runs_dir).resolve() / run_id
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "version": 4,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "goal": goal.strip(),
            "settings": settings.model_dump(),
            "status": "created",
        }
        atomic_write(directory / "run.json", json_text(manifest))
        return cls(directory, settings, manifest, models)

    @classmethod
    def open(
        cls,
        directory: str | Path,
        overrides: dict | None = None,
        models=None,
    ) -> "Collective":
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if manifest.get("version") not in (2, 4):
            raise ValueError("This is not a checkpointed Kollektiv run")
        manifest["version"] = 4
        settings = Settings.model_validate({**manifest["settings"], **(overrides or {})})
        return cls(directory, settings, manifest, models)

    def emit(self, kind: str, **fields) -> dict:
        event = {"id": uuid4().hex, "at": datetime.now(UTC).isoformat(), "kind": kind, **fields}
        with self._trace_lock:
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            if self._on_event:
                self._on_event(event)
        return event

    @property
    def graph_config(self) -> dict:
        return {
            "configurable": {"thread_id": self.manifest["run_id"]},
            "max_concurrency": self.settings.max_concurrency,
            "recursion_limit": max(
                1000,
                self.settings.max_tasks
                * self.settings.max_task_attempts
                * self.settings.max_steps
                * 4
                * (self.settings.max_goal_revisions + 1),
            ),
        }

    @contextmanager
    def graph(self, control: RunControl | None = None):
        connection = sqlite3.connect(
            self.directory / "checkpoints.sqlite",
            check_same_thread=False,
            timeout=30,
        )
        try:
            with self._database_init_lock:
                if connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                    connection.execute("PRAGMA journal_mode=WAL")
                saver = SqliteSaver(connection)
                saver.setup()
            workflow = Workflow(self.settings, self.workspace, self.emit, self.models, control)
            yield workflow.build(saver), workflow
        finally:
            connection.close()

    def state(self, include_workers: bool = False) -> dict:
        if not (self.directory / "checkpoints.sqlite").exists():
            return {"status": self.manifest["status"], "goal": self.manifest["goal"], "tasks": {}}
        with self.graph() as (graph, _):
            snapshot = graph.get_state(self.graph_config, subgraphs=include_workers)
            result = snapshot.values or {"status": self.manifest["status"], "tasks": {}}
            if include_workers:
                result = {
                    **result,
                    "worker_states": [
                        {
                            "id": task.id,
                            "name": task.name,
                            "next": list(task.state.next),
                            "values": task.state.values,
                        }
                        for task in snapshot.tasks
                        if isinstance(task.state, StateSnapshot)
                    ],
                }
            return result

    def export(self, state: dict) -> None:
        atomic_write(self.directory / "state.json", json_text(state))
        if "plan" in state:
            atomic_write(self.directory / "plan.json", json_text(state["plan"]))
        atomic_write(
            self.directory / "messages.json", json_text(list(state.get("messages", {}).values()))
        )
        atomic_write(
            self.directory / "memory.json", json_text(list(state.get("notes", {}).values()))
        )
        self.manifest.update(
            status=state.get("status", "created"),
            updated_at=datetime.now(UTC).isoformat(),
            settings=self.settings.model_dump(),
        )
        atomic_write(self.directory / "run.json", json_text(self.manifest))
        report = [
            "# Kollektiv run",
            "",
            f"Status: **{state.get('status', 'created')}**",
            "",
            "## Goal",
            "",
            self.manifest["goal"],
            "",
            "## Result",
            "",
            state.get("final_summary", "Work in progress; inspect events.jsonl for live activity."),
            "",
            "## Tasks",
            "",
        ]
        for task in state.get("tasks", {}).values():
            info = task.get("artifact") or {}
            report.extend(
                [
                    f"- **{task['id']}** ({task['status']}, {task['owner']}): "
                    f"[{task['output']}](artifacts/{task['output']})",
                    f"  - Review: {task.get('feedback') or 'Pending'}",
                    f"  - Words: {info.get('words', 'unknown')}; attempts: {task.get('attempts', 0)}",
                ]
            )
        if self.manifest.get("last_error"):
            report.extend(["", "## Interruption", "", self.manifest["last_error"]])
        atomic_write(self.directory / "report.md", "\n".join(report) + "\n")

    def changed_artifacts(self, state: dict) -> set[str]:
        changed = set()
        for task_id, task in state.get("tasks", {}).items():
            if task["status"] != "completed":
                continue
            try:
                info = self.workspace.inspect(task["output"])
                if info["problems"] or info["sha256"] != task["artifact"]["sha256"]:
                    changed.add(task_id)
            except (OSError, ValueError, KeyError, TypeError):
                changed.add(task_id)
        return changed

    def reopen(
        self, graph, workflow: Workflow, state: dict, task_ids: set[str], reason: str
    ) -> None:
        changes = workflow.revise(
            {
                **state,
                "goal_revision": -1,
                "goal_review": {
                    "feedback": reason,
                    "revise_tasks": sorted(task_ids),
                    "additional_tasks": [],
                },
            }
        )
        changes.update(status="working", goal_review={}, final_summary="", results={})
        graph.update_state(self.graph_config, changes, as_node="revise")

    def run(
        self,
        *,
        retry: bool = False,
        on_event: Callable[[dict], None] | None = None,
        control: RunControl | None = None,
    ) -> dict:
        if not self._run_mutex.acquire(blocking=False):
            raise RunBusyError("This collective is already running")
        self._running_done.clear()
        self._active_control = control
        self._on_event = on_event
        if control:
            control.on_change = lambda snapshot: self.emit("control", **snapshot)
        try:
            return self._run(retry=retry, on_event=on_event, control=control)
        except RunStopped:
            state = {
                **self.state(),
                "status": "stopped",
                "final_summary": "Stopped at a graph boundary. Resume to continue.",
            }
            self.export(state)
            self.emit("run_stopped")
            return state
        finally:
            if control:
                control.finish()
                control.on_change = None
            self._active_control = None
            self._running_done.set()
            self._run_mutex.release()

    def _run(
        self,
        *,
        retry: bool = False,
        on_event: Callable[[dict], None] | None = None,
        control: RunControl | None = None,
    ) -> dict:
        self._on_event = on_event
        with run_lock(self.directory / ".run.lock"), self.graph(control) as (graph, workflow):
            snapshot = graph.get_state(self.graph_config)
            state = snapshot.values
            if state and "plan" in state:
                workflow.validate_plan(Plan.model_validate(state["plan"]))
            changed = self.changed_artifacts(state or {})
            if changed:
                self.reopen(
                    graph,
                    workflow,
                    state,
                    changed,
                    "An accepted artifact changed or disappeared. Rebuild and review it and its dependents.",
                )
                snapshot = graph.get_state(self.graph_config)
                state = snapshot.values
            if state and not snapshot.next:
                if state.get("status") == "completed":
                    self.export(state)
                    return state
                if state.get("status") == "incomplete" and retry:
                    review = state.get("goal_review", {})
                    if review and not review.get("achieved"):
                        revisions = workflow.revise({**state, "goal_revision": -1})
                        revisions.update(status="working", goal_review={}, final_summary="")
                        graph.update_state(self.graph_config, revisions, as_node="revise")
                    else:
                        failed = {k for k, t in state["tasks"].items() if t["status"] == "failed"}
                        self.reopen(
                            graph,
                            workflow,
                            state,
                            failed,
                            "Retry requested; address prior task feedback.",
                        )
                else:
                    self.export(state)
                    return state
            initial = (
                None
                if state
                else {
                    "goal": self.manifest["goal"],
                    "run_id": self.manifest["run_id"],
                    "status": "created",
                    "discussion_round": 0,
                    "contributions": {},
                    "messages": {m["id"]: m for m in self.manifest.get("seed_messages", [])},
                    "notes": {n["id"]: n for n in self.manifest.get("seed_notes", [])},
                    "events": {},
                    "results": {},
                    "tasks": {},
                    "ready": [],
                    "goal_revision": 0,
                    "goal_review": {},
                    "final_summary": "",
                }
            )
            self.manifest.pop("last_error", None)
            try:
                for current in graph.stream(
                    initial,
                    self.graph_config,
                    stream_mode="values",
                ):
                    self.export(current)
                final = graph.get_state(self.graph_config).values
                self.export(final)
                return final
            except RunStopped:
                raise
            except BaseException as exc:
                state = graph.get_state(self.graph_config).values
                self.manifest["last_error"] = f"{type(exc).__name__}: {exc}"
                self.export(state or {"status": "interrupted"})
                self.emit("interrupted", error=self.manifest["last_error"])
                raise

    def admin(self, action: str, control: RunControl | None = None, **fields) -> dict:
        """Apply an explicit operator edit to a quiescent checkpoint, then reschedule work."""
        if not self._running_done.is_set():
            if control is None or control is not self._active_control:
                raise RunBusyError("Pause this run's active control before editing")
            if not control.snapshot()["paused"]:
                raise RunBusyError("Pause and wait for the current node to finish before editing")
            control.stop()
            if not self._running_done.wait(timeout=10):
                raise RunBusyError("The paused runner is still releasing its checkpoint")
        with run_lock(self.directory / ".run.lock"), self.graph() as (graph, workflow):
            state = graph.get_state(self.graph_config).values
            if action == "revise_goal":
                if state:
                    raise ValueError("Edit the goal and team before starting, or create a new run")
                goal = str(fields.get("goal", "")).strip()
                if not goal:
                    raise ValueError("The goal cannot be empty")
                settings = Settings.model_validate(
                    {**self.settings.model_dump(), **fields.get("settings", {})}
                )
                self.settings = settings
                self.workspace.max_bytes = settings.max_file_bytes
                self.manifest.update(goal=goal, settings=settings.model_dump())
                self.export({"status": "created", "goal": goal, "tasks": {}})
                self.emit("admin", action=action)
                return self.state()
            if action in ("inject_message", "share_note"):
                content = str(fields.get("content", "")).strip()
                if not content or len(content) > 6000:
                    raise ValueError("Admin messages and notes need 1 to 6000 characters")
                if action == "inject_message":
                    recipient = fields.get("recipient", "all")
                    if recipient not in [a.name for a in self.settings.team] + ["all"]:
                        raise ValueError("Unknown teammate")
                    entry = record("admin", content, recipient=recipient)
                    key, seed = "messages", "seed_messages"
                else:
                    entry = record("admin", content)
                    key, seed = "notes", "seed_notes"
                self.emit("message" if key == "messages" else "note", **entry)
                changes = {key: {entry["id"]: entry}}
                if not state:
                    self.manifest.setdefault(seed, []).append(entry)
                    self.export({"status": "created", "goal": self.manifest["goal"], "tasks": {}})
                    return self.state()
                route = "revise" if "plan" in state else "begin"
                changes.update(
                    status="working" if "plan" in state else "discussing",
                    goal_review={},
                    final_summary="",
                )
                graph.update_state(self.graph_config, changes, as_node=route)
            elif action in ("retry_task", "reassign_task", "replace_artifact"):
                if "plan" not in state:
                    raise ValueError("No task plan exists yet")
                task_id = fields.get("task_id")
                if action == "replace_artifact":
                    path = fields.get("path", "")
                    task_id = next(
                        (k for k, t in state["tasks"].items() if t["output"] == path), None
                    )
                if task_id not in state["tasks"]:
                    raise ValueError("Unknown task or assigned artifact")
                tasks = {k: dict(v) for k, v in state["tasks"].items()}
                reason = f"Operator requested {action} for {task_id}. Preserve requirements and prior feedback."
                if action == "reassign_task":
                    owner = fields.get("owner")
                    valid = {
                        a.name
                        for a in self.settings.team
                        if "write_file" in a.tools and a.name != self.settings.reviewer
                    }
                    if owner not in valid:
                        raise ValueError(f"Choose a worker with write_file access: {sorted(valid)}")
                    tasks[task_id]["owner"] = owner
                if action == "replace_artifact":
                    self.workspace.write(tasks[task_id]["output"], fields.get("content", ""))
                    reason = "Operator edited this artifact. Read and preserve their changes; correct only unmet acceptance criteria."
                plan = {
                    **state["plan"],
                    "tasks": [
                        Task.model_validate({k: t[k] for k in Task.model_fields}).model_dump()
                        for t in tasks.values()
                    ],
                }
                self.reopen(
                    graph, workflow, {**state, "tasks": tasks, "plan": plan}, {task_id}, reason
                )
            else:
                raise ValueError(f"Unknown admin action: {action}")
            result = graph.get_state(self.graph_config).values
            self.export(result)
            self.emit("admin", action=action, **fields)
            return result
