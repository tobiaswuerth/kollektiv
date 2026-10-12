import json
import threading
import time

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from kollektiv.config import Agent, Settings
from kollektiv.control import RunStopped
from kollektiv.delegation import DelegationManager, HireRequest
from kollektiv.tools import Toolkit
from kollektiv.workspace import Workspace


@pytest.fixture
def ledger(tmp_path):
    settings = Settings()
    settings.agent("builder").tools += ["hire", "fire", "monitor_agents", "instruct_agent"]
    events = []
    workspace = Workspace(tmp_path / "artifacts")

    def emit(kind, **fields):
        event = {"kind": kind, **fields}
        events.append(event)
        return event

    manager = DelegationManager(tmp_path, settings, workspace, emit)
    yield manager, settings, workspace, events
    manager.stop()


def request(name="assistant", **overrides):
    return HireRequest.model_validate(
        {
            "name": name,
            "role": "Specialist author",
            "instructions": "Produce clear complete prose. Respect the parent's corrections.",
            "skills": ["writing", "validation"],
            "tools": ["read_file", "write_file", "check_file", "send_message"],
            "assignment": "Write a concrete explanation of dependency graphs.",
            "output": "explanation.md",
            "acceptance_criteria": ["Explain a DAG and give one actual three-node example."],
            **overrides,
        }
    )


def wait_status(manager, name, expected, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(name)
        if job["status"] in expected:
            return job
        manager.wait_for_child_change(job["parent"], manager.snapshot()["revision"], 0.05)
    raise AssertionError(f"Expected {expected}, got {manager.get(name)['status']}")


def test_hire_exact_contract_runs_independently_and_preserves_static_settings(ledger):
    manager, settings, workspace, events = ledger
    original = settings.model_dump()
    started, proceed = threading.Event(), threading.Event()

    def execute(job, control):
        started.set()
        assert proceed.wait(5)
        with manager.write_guard(job["name"]):
            artifact = workspace.write(job["output"], "A DAG has no cycles: A -> B -> C.")
        return {"status": "completed", "artifact": artifact, "feedback": "All criteria passed."}

    manager.start(execute)
    job = manager.hire("builder", request(), "call-1", "root-task")
    assert started.wait(5)
    assert job["level"] == 2
    assert job["parent"] == "builder"
    assert job["output"] == "delegates/assistant/explanation.md"
    assert manager.get(job["id"])["definition"] == {
        "name": "assistant",
        "role": "Specialist author",
        "persona": request().instructions,
        "skills": ["writing", "validation"],
        "tools": ["read_file", "write_file", "check_file", "send_message"],
        "model": settings.model,
    }
    assert settings.model_dump() == original
    assert manager.child_settings("assistant").agent("assistant").persona == request().instructions
    assert manager.get("assistant")["status"] == "running"
    proceed.set()
    finished = wait_status(manager, "assistant", {"completed"})
    assert finished["artifact"]["sha256"] == workspace.inspect(job["output"])["sha256"]
    inbox = manager.messages_for("builder")
    assert any(message["status"] == "running" for message in inbox.values())
    assert any(message["status"] == "completed" for message in inbox.values())
    assert any(
        event["kind"] == "agent_status" and event["status"] == "completed" for event in events
    )


def test_replayed_and_concurrent_hire_has_one_child_and_one_execution(ledger):
    manager, _, workspace, _ = ledger
    executions = []

    def execute(job, _control):
        executions.append(job["id"])
        workspace.write(job["output"], "Exactly one durable child execution.")
        return {"status": "completed"}

    manager.start(execute)
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(manager.hire("builder", request(), "stable-id", "task-a"))
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wait_status(manager, "assistant", {"completed"})
    replay = manager.hire("builder", request(), "stable-id", "task-a")
    assert {job["id"] for job in results} == {replay["id"]}
    assert executions == [replay["id"]]
    assert len(manager.snapshot()["agents"]) == 1
    with pytest.raises(ValueError, match="differ"):
        manager.hire("builder", request(role="Changed role"), "stable-id", "task-a")


def test_legacy_model_argument_does_not_invalidate_identical_hire_replay(ledger):
    manager, _, _, _ = ledger
    original = manager.hire("builder", request(model="GPT-4o"), "legacy-call", "same-task")
    replayed = manager.hire("builder", request(model=None), "legacy-call", "same-task")
    assert replayed["id"] == original["id"]
    assert replayed["request"] == original["request"]
    assert replayed["request"]["model"] == "GPT-4o"
    assert replayed["model"] == manager.settings.model
    with pytest.raises(ValueError, match="arguments differ"):
        manager.hire(
            "builder",
            request(model=None, assignment="A different assignment."),
            "legacy-call",
            "same-task",
        )


def test_real_toolnode_injects_call_id_and_enforces_exact_tools(ledger):
    manager, settings, workspace, _ = ledger
    toolkit = Toolkit(
        settings,
        workspace,
        "builder",
        "root.md",
        lambda *_a, **_k: {},
        delegation=manager,
        task_id="main",
    )
    tools = toolkit.build()
    hire_tool = next(tool for tool in tools if tool.name == "hire")
    assert "tool_call_id" not in hire_tool.tool_call_schema.model_json_schema()["properties"]
    assert "model" not in hire_tool.tool_call_schema.model_json_schema()["properties"]
    reply = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "hire",
                "args": request().model_dump(),
                "id": "stable-tool-id",
                "type": "tool_call",
            }
        ],
    )
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    compiled = graph.compile()
    first = compiled.invoke({"messages": [reply]})
    second = compiled.invoke({"messages": [reply]})
    assert (
        json.loads(first["messages"][-1].content)["id"]
        == json.loads(second["messages"][-1].content)["id"]
    )
    child_tools = Toolkit(
        manager.child_settings("assistant"),
        workspace,
        "assistant",
        manager.get("assistant")["output"],
        lambda *_a, **_k: {},
        delegation=manager,
    ).build()
    assert {tool.name for tool in child_tools} == set(request().tools)


def test_level_three_cannot_receive_or_execute_hire_fire_even_if_ledger_is_tampered(ledger):
    manager, _, workspace, _ = ledger
    manager.hire(
        "builder",
        request("middle", tools=["write_file", "hire", "fire", "monitor_agents", "instruct_agent"]),
        "middle",
    )
    for capability in ("hire", "fire"):
        with pytest.raises(ValueError, match="Level-3"):
            manager.hire(
                "middle",
                request(
                    "bad-leaf", tools=["write_file", capability, "monitor_agents", "instruct_agent"]
                ),
                "bad-" + capability,
            )
    leaf = manager.hire("middle", request("leaf"), "leaf")
    assert leaf["level"] == 3
    # The boundary checks depth independently of capability exposure and schema validation.
    with manager._condition:
        poisoned = manager.get("leaf")
        poisoned["definition"]["tools"] += ["hire", "fire", "monitor_agents", "instruct_agent"]
        manager._save(poisoned)
    with pytest.raises(ValueError, match="Level-3"):
        manager.hire("leaf", request("great-grandchild"), "forbidden")
    with pytest.raises(ValueError, match="Level-3"):
        manager.fire("leaf", "middle")
    tools = Toolkit(
        manager.child_settings("leaf"),
        workspace,
        "leaf",
        leaf["output"],
        lambda *_a, **_k: {},
        delegation=manager,
    ).build()
    assert not {"hire", "fire"} & {tool.name for tool in tools}
    assert len(manager.snapshot()["agents"]) == 2


def test_monitor_automatic_progress_and_intervention_persist_across_reload(ledger):
    manager, settings, workspace, _ = ledger
    manager.hire("builder", request(), "call")
    manager.progress("assistant", "node_started", node="worker.act", task="a")
    manager.progress("assistant", "tool_result", tool="read_file", content="An observed fact")
    manager.instruct("builder", "assistant", "Use the corrected assumption from A to B.")
    context = manager.monitor("builder")
    assert context["children"]["assistant"]["current_node"] == "worker.act"
    assert context["children"]["assistant"]["last_tool"] == "read_file"
    assert any(
        event["kind"] == "tool_result"
        for event in context["children"]["assistant"]["recent_progress"]
    )
    reopened = DelegationManager(manager.directory, settings, workspace, lambda *_a, **_k: {})
    assert reopened.get("assistant")["status"] == "running"
    assert reopened.context("assistant")["your_interventions"][0]["content"].startswith(
        "Use the corrected"
    )
    assert any(
        message.get("intervention") for message in reopened.messages_for("assistant").values()
    )
    assert reopened.get("assistant")["history"] == manager.get("assistant")["history"]
    # Merely opening/inspecting the ledger never attaches or starts execution.
    assert not reopened._threads


def test_direct_parent_child_messages_ignore_static_graph_without_opening_sibling_channels(ledger):
    manager, settings, workspace, _ = ledger
    settings.communication_links = []
    manager.hire("builder", request("one"), "one")
    manager.hire("builder", request("two"), "two")
    first = manager.send_message("one", "builder", "I need clarification.")
    second = manager.send_message("builder", "one", "Use a three-node example.")
    assert first["id"] in manager.messages_for("builder")
    assert second["id"] in manager.messages_for("one")
    with pytest.raises(ValueError, match="parent-child"):
        manager.send_message("one", "two", "Cross-family shortcut")
    child_tools = {
        tool.name: tool
        for tool in Toolkit(
            manager.child_settings("one"),
            workspace,
            "one",
            manager.get("one")["output"],
            lambda *_a, **_k: {},
            delegation=manager,
        ).build()
    }
    with pytest.raises(ValueError, match="parent-child"):
        child_tools["send_message"].invoke(
            {"recipient": "reviewer", "content": "Unapproved shortcut"}
        )
    result = json.loads(
        child_tools["send_message"].invoke(
            {"recipient": "all", "content": "A bounded family broadcast."}
        )
    )
    assert result["recipients"] == ["builder"]


def test_supervision_permissions_and_caps_are_backend_enforced(ledger):
    manager, settings, _, _ = ledger
    with pytest.raises(ValueError, match="does not have"):
        manager.hire("researcher", request(), "x")
    settings.max_dynamic_agents = 1
    settings.max_hires = 2
    manager.hire("builder", request(), "first")
    with pytest.raises(ValueError, match="Active subagent limit"):
        manager.hire("builder", request("second"), "second")
    with pytest.raises(ValueError, match="does not have"):
        manager.fire("researcher", "assistant")
    settings.agent("researcher").tools += ["fire", "instruct_agent"]
    with pytest.raises(ValueError, match="own direct children"):
        manager.fire("researcher", "assistant")
    with pytest.raises(ValueError, match="own direct children"):
        manager.instruct("researcher", "assistant", "Unapproved edit")
    manager.fire("builder", "assistant")
    manager.hire("builder", request("second"), "second")
    manager.fire("builder", "second")
    with pytest.raises(ValueError, match="lifetime hire limit"):
        manager.hire("builder", request("third"), "third")


def test_fire_cascades_and_blocks_future_tool_writes_while_retaining_artifacts(ledger):
    manager, _, workspace, _ = ledger
    parent = manager.hire(
        "builder",
        request("middle", tools=["write_file", "hire", "fire", "monitor_agents", "instruct_agent"]),
        "middle",
    )
    leaf = manager.hire("middle", request("leaf"), "leaf")
    workspace.write(leaf["output"], "Already finished evidence remains inspectable.")
    fired = manager.fire("builder", "middle", "The assignment has been superseded.")
    assert fired["status"] == "retired"
    assert manager.get("leaf")["status"] == "retired"
    for job in (parent, leaf):
        with pytest.raises(RunStopped, match="fired"):
            with manager.write_guard(job["name"]):
                workspace.write(job["output"], "Forbidden late write")
    assert workspace.read(leaf["output"]) == "Already finished evidence remains inspectable."
    assert any(
        message["status"] == "retired" for message in manager.messages_for("builder").values()
    )


def test_fire_waits_for_inflight_write_and_then_prevents_every_later_write(ledger):
    manager, _, workspace, _ = ledger
    job = manager.hire("builder", request(), "write-test")
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()

    def writer():
        with manager.write_guard("assistant"):
            entered.set()
            assert release.wait(5)
            workspace.write(job["output"], "Atomic completed write.")

    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(5)
    firing = threading.Thread(target=lambda: (manager.fire("builder", "assistant"), retired.set()))
    firing.start()
    assert not retired.wait(0.05)
    release.set()
    thread.join()
    firing.join()
    assert retired.is_set()
    with pytest.raises(RunStopped):
        with manager.write_guard("assistant"):
            workspace.write(job["output"], "Cannot overwrite after fire.")
    assert workspace.read(job["output"]) == "Atomic completed write."


def test_stop_marks_child_interrupted_and_restart_reuses_its_identity(ledger):
    manager, settings, workspace, _ = ledger
    started = threading.Event()
    executions = []

    def interrupted(job, control):
        executions.append(job["id"])
        started.set()
        control.pause()
        control.enter("worker.act")
        raise AssertionError("Stopped child must never execute the next model call")

    manager.start(interrupted)
    original = manager.hire("builder", request(), "resume")
    assert started.wait(5)
    manager.stop()
    assert manager.get("assistant")["status"] == "interrupted"
    reopened = DelegationManager(manager.directory, settings, workspace, lambda *_a, **_k: {})

    def resumed(job, _control):
        executions.append(job["id"])
        with reopened.write_guard(job["name"]):
            artifact = workspace.write(job["output"], "Resumed at the same scoped output.")
        return {"status": "completed", "artifact": artifact}

    reopened.start(resumed)
    try:
        finished = wait_status(reopened, "assistant", {"completed"})
        assert finished["id"] == original["id"]
        assert executions == [original["id"], original["id"]]
        assert reopened.hire("builder", request(), "resume")["id"] == original["id"]
    finally:
        reopened.stop()


@pytest.mark.parametrize("tools", [[], ["read_file"], ["read_file", "send_message"]])
def test_readonly_subagent_preserves_exact_permissions_and_returns_actual_summary(ledger, tools):
    manager, _, workspace, _ = ledger

    def execute(job, _control):
        assert job["output"] is None
        return {
            "status": "completed",
            "summary": "Verified fact: A -> B -> C is acyclic.",
            "artifact": None,
            "feedback": "Independent reviewer accepted.",
        }

    manager.start(execute)
    manager.hire("builder", request(tools=tools, output=None), "readonly")
    finished = wait_status(manager, "assistant", {"completed"})
    assert finished["tools"] == tools
    assert finished["artifact"] is None
    assert finished["summary"] == "Verified fact: A -> B -> C is acyclic."
    assert finished["result"]["feedback"] == "Independent reviewer accepted."
    exposed = Toolkit(
        manager.child_settings("assistant"),
        workspace,
        "assistant",
        None,
        lambda *_a, **_k: {},
        delegation=manager,
    ).build()
    assert {tool.name for tool in exposed} == set(tools)
    assert any(
        finished["summary"] in message["content"]
        for message in manager.messages_for("builder").values()
    )


def test_writer_output_and_readonly_output_consistency(ledger):
    manager, _, _, _ = ledger
    with pytest.raises(ValueError, match="writing tools needs an assigned output"):
        manager.hire("builder", request(output=None), "writer-without-path")
    with pytest.raises(ValueError, match="read-only child must use output=null"):
        manager.hire("builder", request(tools=[]), "readonly-with-path")
    readonly_manager = manager.hire(
        "builder",
        request("middle", tools=["hire", "fire", "monitor_agents", "instruct_agent"], output=None),
        "middle",
    )
    assert readonly_manager["output"] is None
    assert manager.hire("middle", request("leaf"), "leaf")["level"] == 3


def test_guarded_write_replays_after_pending_file_commit_without_duplicate_writes(
    ledger, monkeypatch
):
    manager, settings, workspace, _ = ledger
    job = manager.hire("builder", request(), "hire")
    writes = []
    original = workspace.write

    def observed(path, content):
        writes.append(path)
        return original(path, content)

    monkeypatch.setattr(workspace, "write", observed)
    first, replayed = manager.guarded_write(
        "assistant", job["output"], "Complete durable artifact.", "stable-write-id"
    )
    assert not replayed
    # Simulate a crash after the atomic artifact replace but before its receipt.
    with manager._connection() as connection:
        connection.execute(
            "UPDATE writes SET body=?",
            (
                json.dumps(
                    {"output": job["output"], "sha256": first["sha256"], "status": "pending"}
                ),
            ),
        )
    reopened = DelegationManager(manager.directory, settings, workspace, lambda *_a, **_k: {})
    second, replayed = reopened.guarded_write(
        "assistant", job["output"], "Complete durable artifact.", "stable-write-id"
    )
    assert replayed
    assert first == second
    assert writes == [job["output"]]
    with pytest.raises(ValueError, match="arguments differ"):
        reopened.guarded_write(
            "assistant", job["output"], "Different replay payload.", "stable-write-id"
        )
    with pytest.raises(ValueError, match="assigned scoped output"):
        reopened.guarded_write("assistant", "someone-else.md", "Forbidden escape.", "another-id")
    manager.fire("builder", "assistant")
    with pytest.raises(RunStopped):
        reopened.guarded_write(
            "assistant", job["output"], "Complete durable artifact.", "stable-write-id"
        )


def test_tool_write_receipt_suppresses_duplicate_artifact_events(ledger):
    manager, _, workspace, _ = ledger
    job = manager.hire("builder", request(), "hire")
    events = []
    toolkit = Toolkit(
        manager.child_settings("assistant"),
        workspace,
        "assistant",
        job["output"],
        lambda kind, **fields: events.append({"kind": kind, **fields}),
        delegation=manager,
        task_id=job["task_id"],
        write_key="tool-call-id",
    )
    write = next(tool for tool in toolkit.build() if tool.name == "write_file")
    first = write.invoke({"content": "One actual atomic write."})
    second = write.invoke({"content": "One actual atomic write."})
    assert first == second
    assert len(events) == 1
    assert events[0]["kind"] == "artifact_written"


def test_unexpected_child_error_is_resumable_and_propagates_to_parent(ledger):
    manager, _, _, _ = ledger

    def failed_connection(_job, _control):
        raise ConnectionError("Model temporarily unavailable")

    manager.start(failed_connection)
    manager.hire("builder", request(), "child")
    job = wait_status(manager, "assistant", {"interrupted"})
    assert "ConnectionError" in job["error"]
    manager._threads["assistant"].join(timeout=5)
    with pytest.raises(RuntimeError, match="resume this collective"):
        manager.raise_if_interrupted("builder")
    manager.fire("builder", "assistant")
    manager.raise_if_interrupted("builder")


def test_live_recovery_thread_is_not_mistaken_for_a_dead_interrupted_child(ledger, monkeypatch):
    manager, settings, workspace, _ = ledger
    job = manager.hire("builder", request(), "recovery")
    with manager._condition:
        manager._transition(job, "interrupted", error="Previous model connection failed")
    reopened = DelegationManager(manager.directory, settings, workspace, lambda *_a, **_k: {})
    entered, release = threading.Event(), threading.Event()
    original_run = reopened._run_child

    def delayed_transition(name, control):
        entered.set()
        assert release.wait(5)
        original_run(name, control)

    monkeypatch.setattr(reopened, "_run_child", delayed_transition)
    reopened.start(lambda _job, _control: {"status": "completed"})
    try:
        assert entered.wait(5)
        assert reopened.get("assistant")["status"] == "interrupted"
        assert reopened._threads["assistant"].is_alive()
        reopened.raise_if_interrupted("builder")
        release.set()
        wait_status(reopened, "assistant", {"completed"})
    finally:
        release.set()
        reopened.stop()


def test_fire_keeps_completed_readonly_findings_and_bounds_notification_context(ledger):
    manager, _, _, _ = ledger
    report = "Verified analytical findings: " + "one real observation " * 1000

    def execute(_job, _control):
        return {"status": "completed", "summary": report, "feedback": "Independently reviewed."}

    manager.start(execute)
    manager.hire("builder", request(output=None, tools=[]), "long-report")
    wait_status(manager, "assistant", {"completed"})
    retired = manager.fire("builder", "assistant", "Evidence integrated; release this specialist.")
    assert retired["summary"] == report
    assert retired["result"]["summary"] == report
    assert retired["result"]["feedback"] == "Independently reviewed."
    assert retired["retirement_reason"] == "Evidence integrated; release this specialist."
    assert len(manager.context("builder")["children"]["assistant"]["summary"]) == 3000
    assert all(
        len(message["content"]) <= 6000 for message in manager.messages_for("builder").values()
    )
    assert any(
        "Evidence integrated" in message["content"]
        for message in manager.messages_for("builder").values()
    )


@pytest.mark.parametrize("path", ["../escape.md", "C:/escape.md", "/escape.md", "a//b"])
def test_hire_artifact_paths_cannot_escape_scoped_workspace(ledger, path):
    manager, _, _, _ = ledger
    with pytest.raises(ValueError):
        manager.hire("builder", request(output=path), "escape")
    assert not manager.snapshot()["agents"]


@pytest.mark.parametrize(
    "tools", [["hire"], ["hire", "monitor_agents"], ["hire", "instruct_agent"]]
)
def test_hire_permission_requires_explicit_monitor_and_intervention_helpers(tools):
    with pytest.raises(ValueError, match="hire skill requires explicitly granting"):
        Agent(name="incomplete", role="Manager", persona="Supervise carefully.", tools=tools)
    with pytest.raises(ValueError, match="hire skill requires explicitly granting"):
        request(tools=tools, output=None)
    exact = [*tools, *[name for name in ("monitor_agents", "instruct_agent") if name not in tools]]
    assert (
        Agent(name="complete", role="Manager", persona="Supervise carefully.", tools=exact).tools
        == exact
    )


def test_assembly_accepts_reviewed_direct_child_and_preflight_results_after_retirement(ledger):
    manager, settings, workspace, _ = ledger
    settings.agent("builder").tools.append("assemble_files")

    def execute(job, _control):
        artifact = workspace.write(job["output"], "Complete accepted specialist evidence.")
        return {"status": "completed", "artifact": artifact}

    manager.start(execute)
    manager.hire("builder", request(), "preflight", "discussion:1:builder")
    child = wait_status(manager, "assistant", {"completed"})
    assert manager.approved_child_inputs("builder", "final-deliverable") == [child["output"]]
    toolkit = Toolkit(
        settings,
        workspace,
        "builder",
        "final.md",
        lambda *_a, **_k: {},
        delegation=manager,
        task_id="final-deliverable",
    )
    assembly = next(tool for tool in toolkit.build() if tool.name == "assemble_files")
    assembly.invoke({"paths": [child["output"]], "prefix": "# Verified report\n\n"})
    assert (
        workspace.read("final.md") == "# Verified report\n\nComplete accepted specialist evidence."
    )
    manager.fire("builder", "assistant", "Integrate approved result and release worker.")
    assert manager.approved_child_inputs("builder", "another-deliverable") == [child["output"]]
    workspace.write(child["output"], "Operator changed this after independent approval.")
    assert manager.approved_child_inputs("builder", "another-deliverable") == []
    with pytest.raises(ValueError, match="independently accepted"):
        assembly.invoke({"paths": [child["output"]]})
    assert workspace.read("final.md").startswith("# Verified report")


def test_assembly_rejects_unfinished_failed_sibling_and_grandchild_results(ledger):
    manager, settings, workspace, _ = ledger
    settings.agent("builder").tools.append("assemble_files")
    middle = manager.hire(
        "builder",
        request(
            "middle", tools=["assemble_files", "hire", "fire", "monitor_agents", "instruct_agent"]
        ),
        "middle",
    )
    leaf = manager.hire("middle", request("leaf"), "leaf")
    artifact = workspace.write(leaf["output"], "Finished-looking text without parent ownership.")
    assert manager.approved_child_inputs("middle") == []
    with manager._condition:
        manager._transition(
            manager.get("leaf"),
            "completed",
            artifact=artifact,
            result={"status": "completed", "artifact": artifact},
        )
    assert manager.approved_child_inputs("middle") == [leaf["output"]]
    assert manager.approved_child_inputs("builder") == []
    assembly = next(
        tool
        for tool in Toolkit(
            settings, workspace, "builder", "final.md", lambda *_a, **_k: {}, delegation=manager
        ).build()
        if tool.name == "assemble_files"
    )
    with pytest.raises(ValueError, match="independently accepted"):
        assembly.invoke({"paths": [leaf["output"]]})
    with manager._condition:
        manager._transition(
            manager.get("middle"),
            "failed",
            artifact=workspace.write(middle["output"], "Unaccepted middle output."),
            result={"status": "failed"},
        )
    assert manager.approved_child_inputs("builder") == []
    manager.fire("builder", "middle")
    assert manager.approved_child_inputs("builder") == []
    with pytest.raises(ValueError, match="independently accepted"):
        assembly.invoke({"paths": [middle["output"]]})
