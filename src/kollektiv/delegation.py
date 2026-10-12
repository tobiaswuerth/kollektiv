"""Durable, bounded agent families with independent workers and parent supervision.

The ledger is deliberately separate from the main graph checkpoint. Replaying a
hire tool call returns the same child, and each child's graph has its own stable
checkpoint thread. Inspection never starts work; only the owning runner calls start.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Agent, Settings, ToolName, validate_hiring_tools
from .control import RunControl, RunStopped
from .workspace import Workspace, relative_path

TERMINAL = frozenset({"completed", "failed", "retired"})
ACTIVE = frozenset({"pending", "running", "awaiting_children", "reviewing", "interrupted"})
DELEGATION_TOOLS = frozenset({"hire", "fire", "monitor_agents", "instruct_agent"})


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


class HireRequest(BaseModel):
    """Everything a parent must specify to create one narrowly scoped worker."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    role: str = Field(min_length=1, max_length=2000)
    instructions: str = Field(min_length=1, max_length=30000)
    skills: list[str] = Field(max_length=50)
    tools: list[ToolName] = Field(max_length=30)
    assignment: str = Field(min_length=1, max_length=30000)
    output: str | None = Field(
        default=None,
        description="Relative output inside the child's unique delegates/name folder, or null for a read-only report",
    )
    acceptance_criteria: list[str] = Field(min_length=1, max_length=30)
    input_files: list[str] = Field(default_factory=list, max_length=100)
    model: str | None = None

    @model_validator(mode="after")
    def validate_supervision(self) -> HireRequest:
        validate_hiring_tools(self.tools)
        return self

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: str | None) -> str | None:
        return relative_path(value) if value is not None else None

    @field_validator("input_files")
    @classmethod
    def validate_inputs(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(relative_path(value) for value in values))


class DelegationManager:
    """One run-owned manager; all ledger changes and artifact gates are serialized."""

    def __init__(
        self,
        directory: Path,
        settings: Settings,
        workspace: Workspace,
        emit: Callable[..., dict],
    ):
        self.directory = directory.resolve()
        self.settings = settings
        self.workspace = workspace
        self.emit = emit
        self.path = self.directory / "delegation.sqlite"
        self._condition = threading.Condition(threading.RLock())
        self._threads: dict[str, threading.Thread] = {}
        self._controls: dict[str, RunControl] = {}
        self._execute: Callable[[dict, RunControl], dict] | None = None
        self._running = False
        self._revision = 0
        self.model_gate = threading.BoundedSemaphore(settings.max_concurrency)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS agents "
                "(id TEXT PRIMARY KEY, name TEXT UNIQUE, request_key TEXT UNIQUE, body TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS messages "
                "(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, recipient TEXT, body TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS writes (operation_key TEXT PRIMARY KEY, body TEXT)"
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _jobs(self) -> dict[str, dict]:
        with self._connection() as connection:
            rows = connection.execute("SELECT body FROM agents ORDER BY rowid").fetchall()
        return {job["name"]: job for (body,) in rows if (job := json.loads(body))}

    def _save(self, job: dict) -> None:
        job["updated_at"] = timestamp()
        with self._connection() as connection:
            connection.execute("UPDATE agents SET body=? WHERE id=?", (json.dumps(job), job["id"]))
        self._revision += 1
        self._condition.notify_all()

    def _message(self, sender: str, recipient: str, content: str, **fields) -> dict:
        message = {
            "id": uuid4().hex,
            "at": timestamp(),
            "sender": sender,
            "recipient": recipient,
            "content": content,
            **fields,
        }
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO messages(id,recipient,body) VALUES(?,?,?)",
                (message["id"], recipient, json.dumps(message)),
            )
        self._revision += 1
        self._condition.notify_all()
        return message

    def _history(self, job: dict, kind: str, **fields) -> dict:
        event = {"id": uuid4().hex, "at": timestamp(), "kind": kind, **fields}
        job["history"] = [*job.get("history", []), event][-100:]
        return event

    def _transition(self, job: dict, status: str, **fields) -> list[tuple[str, dict]]:
        previous = job["status"]
        job.update(status=status, **fields)
        event = self._history(
            job,
            "agent_status",
            agent=job["name"],
            child=job["id"],
            parent=job["parent"],
            level=job["level"],
            previous_status=previous,
            status=status,
            **fields,
        )
        self._save(job)
        report = (
            fields.get("retirement_reason") or fields.get("summary") or fields.get("error") or ""
        )
        output = (
            f"Output: {str(job['output'])[:500]}. "
            if job["output"]
            else "Report available in monitor_agents. "
        )
        message = self._message(
            job["name"],
            job["parent"],
            f"Subagent {job['name']} changed from {previous} to {status}. "
            + output
            + str(report)[:4000],
            system=True,
            child=job["id"],
            status=status,
        )
        return [("agent_status", event), ("message", message)]

    def _publish(self, events: list[tuple[str, dict]]) -> None:
        # Never call external callbacks under the ledger lock: event subscribers
        # inspect the manager and the collective trace has its own lock.
        for kind, event in events:
            self.emit(kind, **{key: value for key, value in event.items() if key != "kind"})

    def get(self, identity: str) -> dict:
        with self._condition:
            jobs = self._jobs()
            job = jobs.get(identity) or next(
                (job for job in jobs.values() if job["id"] == identity), None
            )
            if job is None:
                raise ValueError("Unknown subagent")
            return job

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "agents": self._jobs(),
                "max_depth": 3,
                "max_dynamic_agents": self.settings.max_dynamic_agents,
                "max_hires": self.settings.max_hires,
                "revision": self._revision,
            }

    def actor(self, name: str) -> tuple[Agent, int]:
        static = next((agent for agent in self.settings.team if agent.name == name), None)
        if static is not None:
            return static, 1
        job = self.get(name)
        return Agent.model_validate(job["definition"]), job["level"]

    def child_settings(self, name: str) -> Settings:
        """A fresh model/configuration view; never mutate a shared static team."""
        self.get(name)
        data = self.settings.model_dump()
        data["team"] += [job["definition"] for job in self.snapshot()["agents"].values()]
        return Settings.model_validate(data)

    def _require(self, parent: str, capability: str) -> tuple[Agent, int]:
        agent, level = self.actor(parent)
        if level >= 3 and capability in {"hire", "fire"}:
            raise ValueError("Level-3 subagents cannot hire or fire agents")
        if capability not in agent.tools:
            raise ValueError(f"{parent} does not have the {capability} skill/tool")
        self.check_active(parent)
        return agent, level

    def check_active(self, name: str) -> None:
        """Enforce cancellation at boundaries, including all ancestor retirements."""
        with self._condition:
            jobs = self._jobs()
            current = jobs.get(name)
            while current is not None:
                if current["status"] == "retired":
                    raise RunStopped("This subagent or its parent was fired")
                control = self._controls.get(current["name"])
                if control and control.snapshot()["stopped"]:
                    raise RunStopped("Subagent execution stopped at a resumable boundary")
                current = jobs.get(current["parent"])

    @contextmanager
    def write_guard(self, name: str):
        """Fire returns only after existing writes finish; later writes are rejected."""
        with self._condition:
            self.check_active(name)
            yield

    def guarded_write(
        self, name: str, output: str, content: str, request_key: str
    ) -> tuple[dict, bool]:
        """Idempotent child writes, including a crash between file and graph commits.

        Persist the intent first. A replay checks the actual bytes before deciding
        whether to write, recognizing an atomic replace with an uncommitted receipt.
        The original exact tool-call identity scopes each operation.
        """
        with self.write_guard(name):
            job = self.get(name)
            if output != job["output"]:
                raise ValueError("A subagent can only write its assigned scoped output")
            if not content.strip() or len(content.encode("utf-8")) > self.workspace.max_bytes:
                raise ValueError("Deliverable is empty or exceeds the artifact byte limit")
            key = json.dumps([job["id"], request_key])
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT body FROM writes WHERE operation_key=?", (key,)
                ).fetchone()
                existing = json.loads(row[0]) if row else None
                if existing and (existing["output"] != output or existing["sha256"] != digest):
                    raise ValueError("Replayed write arguments differ from the original operation")
                if not existing:
                    intent = {"output": output, "sha256": digest, "status": "pending"}
                    connection.execute(
                        "INSERT INTO writes(operation_key,body) VALUES(?,?)",
                        (key, json.dumps(intent)),
                    )
            artifact = None
            try:
                current = self.workspace.inspect(output)
                if current["sha256"] == digest:
                    artifact = current
            except (OSError, ValueError):
                pass
            replayed = artifact is not None
            if artifact is None:
                artifact = self.workspace.write(output, content)
            with self._connection() as connection:
                connection.execute(
                    "UPDATE writes SET body=? WHERE operation_key=?",
                    (
                        json.dumps(
                            {
                                "output": output,
                                "sha256": digest,
                                "status": "completed",
                                "artifact": artifact,
                            }
                        ),
                        key,
                    ),
                )
            return artifact, replayed

    def hire(
        self,
        parent: str,
        request: HireRequest | dict,
        request_key: str,
        parent_task_id: str = "",
    ) -> dict:
        request = HireRequest.model_validate(request)
        events = []
        with self._condition:
            _, level = self._require(parent, "hire")
            if not request_key:
                raise ValueError("A stable tool call ID is required to hire an agent")
            key = json.dumps([parent, parent_task_id, request_key])
            jobs = self._jobs()
            existing = next((job for job in jobs.values() if job["request_key"] == key), None)
            if existing is not None:
                if existing["request"] != request.model_dump():
                    raise ValueError("Replayed hire arguments differ from the original request")
                return existing
            if request.name in jobs or request.name in {agent.name for agent in self.settings.team}:
                raise ValueError("Agent names must be unique, including retired agents")
            if level + 1 >= 3 and {"hire", "fire"} & set(request.tools):
                raise ValueError("Level-3 subagents cannot be granted hire or fire")
            writing = bool({"write_file", "assemble_files"} & set(request.tools))
            if writing and request.output is None:
                raise ValueError("A child with writing tools needs an assigned output")
            if not writing and request.output is not None:
                raise ValueError("A read-only child must use output=null")
            if (
                "assemble_files" in request.tools
                and "write_file" not in request.tools
                and not request.input_files
            ):
                raise ValueError("An assembly-only worker needs dependency input_files")
            if len(jobs) >= self.settings.max_hires:
                raise ValueError(f"Run lifetime hire limit reached ({self.settings.max_hires})")
            if (
                sum(job["status"] in ACTIVE for job in jobs.values())
                >= self.settings.max_dynamic_agents
            ):
                raise ValueError(
                    f"Active subagent limit reached ({self.settings.max_dynamic_agents})"
                )
            for path in request.input_files:
                self.workspace.read(path)  # Require actual inputs before a child starts.
            identity = uuid4().hex
            output = (
                relative_path(f"delegates/{request.name}/{request.output}")
                if request.output
                else None
            )
            if output:
                self.workspace.path(output)  # Resolve symlinks before persisting a scope.
            definition = Agent(
                name=request.name,
                role=request.role,
                persona=request.instructions,
                skills=request.skills,
                tools=request.tools,
                model=request.model,
            ).model_dump()
            job = {
                "id": identity,
                "name": request.name,
                "role": request.role,
                "persona": request.instructions,
                "instructions": request.instructions,
                "skills": request.skills,
                "tools": request.tools,
                "model": request.model,
                "definition": definition,
                "parent": parent,
                "parent_id": jobs.get(parent, {}).get("id"),
                "level": level + 1,
                "status": "pending",
                "assignment": request.assignment,
                "output": output,
                "requested_output": request.output,
                "input_files": request.input_files,
                "acceptance_criteria": request.acceptance_criteria,
                "task_id": "child-" + identity,
                "parent_task_id": parent_task_id,
                "thread_id": "child-" + identity,
                "request_key": key,
                "request": request.model_dump(),
                "created_at": timestamp(),
                "updated_at": timestamp(),
                "history": [],
                "interventions": [],
                "summary": "",
                "error": "",
            }
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO agents(id,name,request_key,body) VALUES(?,?,?,?)",
                    (identity, request.name, key, json.dumps(job)),
                )
            events += self._transition(job, "pending")
            events.append(("agent_hired", {**job, "agent": request.name, "child": identity}))
        self._publish(events)
        self._launch(job["name"])
        return self.get(job["name"])

    def _launch(self, name: str) -> None:
        with self._condition:
            if not self._running or not self._execute:
                return
            prior = self._threads.get(name)
            if prior and prior.is_alive():
                return
            job = self.get(name)
            if job["status"] in TERMINAL:
                return
            control = RunControl()
            self._controls[name] = control
            thread = threading.Thread(
                target=self._run_child,
                args=(name, control),
                name=f"kollektiv-child-{name}",
                daemon=True,
            )
            self._threads[name] = thread
            thread.start()

    def _run_child(self, name: str, control: RunControl) -> None:
        try:
            with self._condition:
                job = self.get(name)
                if job["status"] == "retired":
                    return
                events = self._transition(job, "running", error="")
            self._publish(events)
            self.check_active(name)
            assert self._execute is not None
            result = self._execute(self.get(name), control)
            # The callback returns WorkerState, WorkerOutput, or a normalized result.
            task_result = result.get("results", {}).get(job["task_id"], result)
            status = "completed" if task_result.get("status") == "completed" else "failed"
            with self._condition:
                job = self.get(name)
                if job["status"] == "retired":
                    return
                events = self._transition(
                    job,
                    status,
                    summary=task_result.get("summary") or task_result.get("feedback", ""),
                    artifact=task_result.get("artifact"),
                    result=task_result,
                    current_node=None,
                )
            self._publish(events)
        except BaseException as exc:
            with self._condition:
                job = self.get(name)
                if job["status"] == "retired":
                    return
                status = "interrupted"
                events = self._transition(
                    job, status, error=f"{type(exc).__name__}: {exc}", current_node=None
                )
            self._publish(events)
        finally:
            control.finish()
            with self._condition:
                self._condition.notify_all()

    def start(self, execute_child: Callable[[dict, RunControl], dict]) -> None:
        """Attach only inside the main OS-held run lock, then resume unfinished jobs."""
        with self._condition:
            if self._running:
                raise ValueError("Delegation execution is already active")
            self._execute = execute_child
            self._running = True
            pending = [job["name"] for job in self._jobs().values() if job["status"] in ACTIVE]
        for name in pending:
            self._launch(name)

    def stop(self, wait: bool = True) -> None:
        """Quiesce before releasing the owning run lock; interrupted jobs remain resumable."""
        with self._condition:
            self._running = False
            controls = list(self._controls.values())
            threads = list(self._threads.values())
        for control in controls:
            if not control.snapshot()["stopped"]:
                control.stop()
        if wait:
            for thread in threads:
                if thread is not threading.current_thread():
                    thread.join()

    def _owned(self, parent: str, child: str, capability: str, admin: bool) -> dict:
        job = self.get(child)
        if not admin:
            self._require(parent, capability)
            if job["parent"] != parent:
                raise ValueError("An agent can only supervise its own direct children")
        return job

    def fire(self, parent: str, child: str, reason: str = "Work finished", *, admin=False) -> dict:
        events = []
        controls = []
        with self._condition:
            job = self._owned(parent, child, "fire", admin)
            jobs = self._jobs()
            names = {job["name"]}
            while True:
                descendants = {item["name"] for item in jobs.values() if item["parent"] in names}
                if descendants <= names:
                    break
                names |= descendants
            for name in names:
                current = jobs[name]
                if current["status"] != "retired":
                    events += self._transition(current, "retired", retirement_reason=reason)
                    events.append(
                        (
                            "agent_fired",
                            {
                                "agent": name,
                                "child": current["id"],
                                "parent": current["parent"],
                                "reason": reason,
                            },
                        )
                    )
                if control := self._controls.get(name):
                    controls.append(control)
        for control in controls:
            control.stop()
        self._publish(events)
        return self.get(job["name"])

    def instruct(self, parent: str, child: str, content: str, *, admin=False) -> dict:
        if not content.strip() or len(content) > 6000:
            raise ValueError("Intervention instructions need 1 to 6000 characters")
        with self._condition:
            job = self._owned(parent, child, "instruct_agent", admin)
            if job["status"] in TERMINAL:
                raise ValueError("Cannot intervene in a finished or retired subagent")
            entry = {"id": uuid4().hex, "at": timestamp(), "sender": parent, "content": content}
            job["interventions"] = [*job["interventions"], entry][-100:]
            self._history(
                job, "agent_instruction", agent=job["name"], parent=parent, content=content
            )
            self._save(job)
            message = self._message(parent, job["name"], content, system=True, intervention=True)
        self._publish(
            [
                (
                    "agent_instruction",
                    {
                        "agent": job["name"],
                        "child": job["id"],
                        "parent": parent,
                        "content": content,
                    },
                ),
                ("message", message),
            ]
        )
        return {"delivered": True, "agent": job["name"], "instruction": entry}

    def messages_for(self, name: str, limit: int = 100) -> dict[str, dict]:
        with self._condition, self._connection() as connection:
            rows = connection.execute(
                "SELECT body FROM messages WHERE recipient=? ORDER BY seq DESC LIMIT ?",
                (name, limit),
            ).fetchall()
        messages = [json.loads(row[0]) for row in reversed(rows)]
        return {message["id"]: message for message in messages}

    def all_messages(self) -> dict[str, dict]:
        with self._condition, self._connection() as connection:
            rows = connection.execute("SELECT body FROM messages ORDER BY seq").fetchall()
        messages = [json.loads(row[0]) for row in rows]
        return {message["id"]: message for message in messages}

    def allowed_recipients(self, sender: str) -> list[str]:
        actor, _ = self.actor(sender)
        if "send_message" not in actor.tools:
            return []
        jobs = self.snapshot()["agents"]
        targets = [
            job["name"]
            for job in jobs.values()
            if job["parent"] == sender and job["status"] != "retired"
        ]
        if sender in jobs:
            targets.append(jobs[sender]["parent"])
        return targets

    def send_message(self, sender: str, recipient: str, content: str) -> dict:
        if recipient not in self.allowed_recipients(sender):
            raise ValueError("Dynamic messages can only follow direct parent-child channels")
        if not content.strip() or len(content) > 6000:
            raise ValueError("Messages need 1 to 6000 characters")
        self.check_active(sender)
        with self._condition:
            message = self._message(sender, recipient, content)
        self._publish([("message", message)])
        return message

    def context(self, parent: str) -> dict:
        snapshot = self.snapshot()
        jobs = snapshot["agents"]
        fields = (
            "id",
            "name",
            "parent",
            "level",
            "status",
            "role",
            "assignment",
            "output",
            "input_files",
            "acceptance_criteria",
            "current_node",
            "last_tool",
            "summary",
            "error",
            "artifact",
            "created_at",
            "updated_at",
        )
        children = {}
        for name, job in jobs.items():
            if job["parent"] == parent:
                summary = {key: job.get(key) for key in fields}
                if isinstance(summary.get("summary"), str):
                    summary["summary"] = summary["summary"][:3000]
                summary["recent_progress"] = [
                    {key: event.get(key) for key in ("at", "kind", "node", "tool", "status")}
                    for event in job.get("history", [])[-5:]
                ]
                children[name] = summary
        return {
            "revision": snapshot["revision"],
            "children": children,
            "delegation_inbox": list(self.messages_for(parent, 30).values()),
            "your_interventions": jobs.get(parent, {}).get("interventions", [])[-20:],
            "level": jobs.get(parent, {}).get("level", 1),
            "max_depth": 3,
        }

    def wait_for_child_change(self, parent: str, revision: int, timeout: float = 10) -> dict:
        with self._condition:
            self._condition.wait_for(
                lambda: self._revision != revision or not self._running,
                timeout=max(0, min(timeout, 10)),
            )
        return self.context(parent)

    def monitor(self, parent: str, wait_seconds: float = 0, since: int | None = None) -> dict:
        self._require(parent, "monitor_agents")
        if not 0 <= wait_seconds <= 10:
            raise ValueError("wait_seconds must be between 0 and 10")
        if wait_seconds:
            self.wait_for_child_change(
                parent, self._revision if since is None else since, wait_seconds
            )
        return self.context(parent)

    def progress(self, child: str, kind: str, **fields) -> None:
        """The root callback forwards worker events; history survives a page reload."""
        events = []
        with self._condition:
            job = self.get(child)
            if job["status"] == "retired":
                return
            self._history(job, kind, **fields)
            if kind == "node_started":
                job["current_node"] = fields.get("node")
            elif kind in {"node_finished", "node_failed"}:
                job["current_node"] = None
            if kind == "tool_result":
                job["last_tool"] = fields.get("tool")
            node = str(fields.get("node", ""))
            status = (
                "awaiting_children"
                if "await_children" in node
                else "reviewing"
                if kind == "node_started" and node.endswith("review")
                else "running"
            )
            if kind == "node_started" and status != job["status"]:
                events += self._transition(job, status)
            else:
                self._save(job)
            events.append(
                (
                    "agent_progress",
                    {
                        "agent": job["name"],
                        "child": job["id"],
                        "parent": job["parent"],
                        "level": job["level"],
                        "status": job["status"],
                        "current_node": job.get("current_node"),
                        "last_tool": job.get("last_tool"),
                        "event": {"kind": kind, **fields},
                    },
                )
            )
        self._publish(events)

    def checkpoint(self, child: str, values: dict) -> None:
        """A small inspectable cache; execution resumes the real LangGraph checkpoint."""
        with self._condition:
            job = self.get(child)
            if job["status"] == "retired":
                return
            job["worker_state"] = {
                key: values.get(key)
                for key in ("attempt", "steps", "feedback", "finished", "review")
            }
            self._save(job)

    def pending_children(self, parent: str) -> list[dict]:
        return [
            job
            for job in self.snapshot()["agents"].values()
            if job["parent"] == parent and job["status"] in ACTIVE
        ]

    def raise_if_interrupted(self, parent: str) -> None:
        with self._condition:
            for job in self.pending_children(parent):
                thread = self._threads.get(job["name"])
                if job["status"] == "interrupted" and not (thread and thread.is_alive()):
                    raise RuntimeError(
                        f"Subagent {job['name']} was interrupted; resume this collective to continue "
                        f"its checkpoint. {job.get('error', '')}"
                    )
