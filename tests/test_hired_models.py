"""Native hires inherit local model configuration; legacy interrupted jobs stay resumable."""

import copy
import json
import threading

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from ollama import ResponseError
from test_dynamic_workflow import (
    MANAGER_TOOLS,
    FamilyModels,
    create_family,
    finish_runner,
    request,
    start_runner,
)
from test_web import eventually

from kollektiv import Collective, Settings
from kollektiv.control import RunControl
from kollektiv.delegation import DelegationManager
from kollektiv.llm import Models
from kollektiv.tools import Toolkit
from kollektiv.webapp import ControlRoomServer, RunManager
from kollektiv.workspace import Workspace


@pytest.fixture
def local_ledger(tmp_path):
    settings = Settings(model="gemma4:e2b")
    settings.agent("builder").model = "gemma4:e4b"
    settings.agent("builder").tools += MANAGER_TOOLS
    workspace = Workspace(tmp_path / "artifacts")
    events = []

    def emit(kind, **fields):
        event = {"kind": kind, **fields}
        events.append(event)
        return event

    manager = DelegationManager(tmp_path, settings, workspace, emit)
    yield manager, settings, workspace, events
    manager.stop()


def test_children_and_grandchildren_use_the_immediate_parents_effective_model(local_ledger):
    manager, settings, _, _ = local_ledger
    original = settings.model_dump()
    first = manager.hire("builder", request(nested=True), "first", "main")
    second = manager.hire("assistant", request("leaf"), "second", first["task_id"])
    assert first["level"] == 2 and second["level"] == 3
    for child in (first, second):
        assert child["model"] == "gemma4:e4b"
        assert child["definition"]["model"] == "gemma4:e4b"
        view = manager.child_settings(child["name"])
        assert Models(view, lambda *_a, **_k: {}).client(child["name"]).model == "gemma4:e4b"
    assert settings.model_dump() == original
    assert settings.model == "gemma4:e2b"


def test_hire_without_static_parent_override_inherits_the_selected_run_model(local_ledger):
    manager, settings, _, _ = local_ledger
    settings.agent("builder").model = None
    settings.model = "gemma4:e4b"
    job = manager.hire("builder", request(), "selected-run-model", "main")
    assert job["model"] == "gemma4:e4b"
    assert manager.inherited_model("assistant") == "gemma4:e4b"


@pytest.mark.parametrize("invented_model", ["GPT-4o", "gpt-4o", "unconfigured-local:latest", {}])
def test_native_hire_hides_model_selection_and_ignores_forged_or_historical_values(
    local_ledger, invented_model
):
    manager, settings, workspace, _ = local_ledger
    tools = Toolkit(
        settings,
        workspace,
        "builder",
        "final.md",
        lambda *_a, **_k: {},
        delegation=manager,
        task_id="main",
    ).build()
    hire = next(tool for tool in tools if tool.name == "hire")
    assert "model" not in hire.tool_call_schema.model_json_schema()["properties"]
    assert "model" not in convert_to_openai_tool(hire)["function"]["parameters"]["properties"]
    legacy_request = {**request(), "model": invented_model}
    reply = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "hire",
                "args": legacy_request,
                "id": "stable-native-hire",
                "type": "tool_call",
            }
        ],
    )
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    compiled = graph.compile()
    first = json.loads(compiled.invoke({"messages": [reply]})["messages"][-1].content)
    replay = json.loads(compiled.invoke({"messages": [reply]})["messages"][-1].content)
    job = manager.get("assistant")
    assert first["id"] == replay["id"] == job["id"]
    assert job["model"] == job["definition"]["model"] == "gemma4:e4b"
    assert job["request"]["model"] is None
    assert len(manager.snapshot()["agents"]) == 1
    assert (
        Models(manager.child_settings("assistant"), lambda *_a, **_k: {}).client("assistant").model
        == "gemma4:e4b"
    )


def test_historical_native_hire_replay_preserves_original_request_model_and_child_identity(
    local_ledger,
):
    manager, settings, workspace, _ = local_ledger
    legacy_request = {**request(), "model": "GPT-4o"}
    original = manager.hire("builder", legacy_request, "legacy-native-id", "main")
    tools = Toolkit(
        settings,
        workspace,
        "builder",
        "final.md",
        lambda *_a, **_k: {},
        delegation=manager,
        task_id="main",
    ).build()
    reply = AIMessage(
        content="",
        tool_calls=[
            {"name": "hire", "args": legacy_request, "id": "legacy-native-id", "type": "tool_call"}
        ],
    )
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    replay = json.loads(graph.compile().invoke({"messages": [reply]})["messages"][-1].content)
    persisted = manager.get("assistant")
    assert replay["id"] == persisted["id"] == original["id"]
    assert persisted["request"] == original["request"]
    assert persisted["request"]["model"] == "GPT-4o"
    assert persisted["model"] == "gemma4:e4b"
    assert len(manager.snapshot()["agents"]) == 1


def test_legacy_model_reset_preserves_request_identity_checkpoint_and_interruption_evidence(
    local_ledger,
):
    manager, _, workspace, events = local_ledger
    legacy_request = {**request(), "model": "GPT-4o"}
    created = manager.hire("builder", legacy_request, "legacy-call", "main")
    workspace.write(created["output"], "Already checkpointed COMPLETE evidence.")
    with manager._condition:
        legacy = manager.get("assistant")
        legacy.update(
            model="GPT-4o",
            status="interrupted",
            error="model GPT-4o not found (status code: 404)",
            worker_state={"attempt": 1, "steps": 2, "finished": False},
        )
        legacy["definition"]["model"] = "GPT-4o"
        manager._save(legacy)
    original = copy.deepcopy(manager.get("assistant"))
    reset = manager.reset_child_model(original["id"])
    assert reset["model"] == reset["definition"]["model"] == "gemma4:e4b"
    for field in (
        "id",
        "task_id",
        "thread_id",
        "request",
        "request_key",
        "worker_state",
        "status",
        "error",
        "output",
    ):
        assert reset[field] == original[field]
    assert workspace.read(original["output"]) == "Already checkpointed COMPLETE evidence."
    assert any(event["kind"] == "agent_model_reset" for event in events)
    assert any(
        "configured model gemma4:e4b" in message["content"]
        for message in manager.messages_for("builder").values()
    )
    replay = manager.hire("builder", legacy_request, "legacy-call", "main")
    assert replay["id"] == original["id"] and replay["request"] == original["request"]
    assert replay["model"] == "gemma4:e4b"


def test_model_reset_rejects_active_execution_and_terminal_child(local_ledger):
    manager, _, _, _ = local_ledger
    manager.hire("builder", request(), "active", "main")
    started, release = threading.Event(), threading.Event()

    def execute(_job, _control):
        started.set()
        assert release.wait(5)
        return {"status": "completed", "summary": "Finished fixture work."}

    manager.start(execute)
    try:
        assert started.wait(5)
        with pytest.raises(ValueError, match="Stop the collective"):
            manager.reset_child_model("assistant")
    finally:
        release.set()
        manager.stop()
    assert manager.get("assistant")["status"] == "completed"
    with pytest.raises(ValueError, match="unfinished"):
        manager.reset_child_model("assistant")


class LegacyBackendFamilyModels(FamilyModels):
    """Create an authentic interrupted checkpoint, then simulate an old bad model cache."""

    def __init__(self):
        super().__init__()
        self.poisoned = False
        self.effective_models = []

    def act(self, agent, messages, tools):
        prompt = messages[1].content
        if agent == "assistant" and prompt.startswith("YOUR TASK:\n"):
            # The assistant's first write and tool result have already been saved to
            # its real LangGraph thread when it reaches this second model call.
            written = any(
                getattr(message, "name", None) == "write_file" for message in messages[2:]
            )
            if written and not self.poisoned:
                self.poisoned = True
                with self.manager._condition:
                    job = self.manager.get("assistant")
                    job["model"] = job["definition"]["model"] = "GPT-4o"
                    job["request"]["model"] = "GPT-4o"
                    self.manager._save(job)
            effective = self.manager.inherited_model("assistant")
            self.effective_models.append(effective)
            if effective == "GPT-4o":
                raise ResponseError("model 'GPT-4o' not found", status_code=404)
        reply = super().act(agent, messages, tools)
        if agent == "builder":
            for tool_call in reply.tool_calls:
                if tool_call["name"] == "hire":
                    # Old checkpoints can retain this argument after the schema is fixed.
                    tool_call["args"]["model"] = "GPT-4o"
        return reply


def test_http_model_repair_resumes_actual_child_checkpoint_with_one_write_and_same_hire(tmp_path):
    models = LegacyBackendFamilyModels()
    run = create_family(tmp_path, models)
    run.settings.model = "gemma4:e4b"
    thread, outcome = start_runner(run, RunControl())
    thread.join(15)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), RuntimeError)
    assert "was interrupted" in str(outcome["error"])
    original = copy.deepcopy(run.delegation.get("assistant"))
    assert original["status"] == "interrupted"
    assert original["model"] == original["request"]["model"] == "GPT-4o"
    original_artifact = run.workspace.inspect(original["output"])
    assert models.writes["assistant"] == 1

    manager = RunManager(tmp_path, models=models)
    server = ControlRoomServer(("127.0.0.1", 0), manager)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=10) as client:
            run_id = run.manifest["run_id"]
            path = f"/api/runs/{run_id}"
            managed = manager.get(run_id)
            models.manager = managed.collective.delegation
            response = client.post(
                path + "/admin", json={"action": "reset_agent_model", "agent_id": original["id"]}
            )
            assert response.status_code == 200, response.text
            repaired = managed.collective.delegation.get("assistant")
            for field in ("id", "task_id", "thread_id", "request", "request_key", "worker_state"):
                assert repaired[field] == original[field]
            assert repaired["model"] == repaired["definition"]["model"] == "gemma4:e4b"
            assert managed.collective.workspace.inspect(original["output"]) == original_artifact
            response = client.post(path + "/control", json={"action": "resume"})
            assert response.status_code == 200, response.text
            eventually(lambda: not managed.running, timeout=15)
            assert managed.error is None
            final = client.get(path).json()
            assert final["state"]["status"] == "completed"
            child = final["delegation"]["agents"]["assistant"]
            assert child["id"] == original["id"] and child["thread_id"] == original["thread_id"]
            assert child["model"] == "gemma4:e4b"
            assert child["request"]["model"] == "GPT-4o"
            assert models.writes["assistant"] == 1 and models.writes["builder"] == 1
            assert models.effective_models == ["gemma4:e4b", "GPT-4o", "gemma4:e4b"]
            assert managed.collective.workspace.inspect(original["output"]) == original_artifact
            replay_key = json.loads(original["request_key"])
            replay = managed.collective.delegation.hire(
                "builder", original["request"], replay_key[2], replay_key[1]
            )
            assert replay["id"] == original["id"]
            written_events = [
                event
                for event in final["events"]
                if event["kind"] == "artifact_written" and event.get("agent") == "assistant"
            ]
            assert len(written_events) == 1
    finally:
        manager.close()
        for managed in list(manager._runs.values()):
            if managed.thread:
                managed.thread.join(5)
        server.shutdown()
        server.server_close()
        server_thread.join(2)

    # A further CLI/API reopen still observes the same completed configuration.
    reopened = Collective.open(run.directory, models=models)
    models.manager = reopened.delegation
    thread, outcome = start_runner(reopened, RunControl())
    assert finish_runner(thread, outcome)["status"] == "completed"
    assert models.writes["assistant"] == 1
