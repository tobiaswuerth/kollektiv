"""Real graph/tool/checkpoint integration for bounded, supervised agent families."""

import json
import threading
from collections import Counter
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from kollektiv import Collective, Settings
from kollektiv.control import RunControl
from kollektiv.models import Contribution, GoalReview, MessageDraft, Plan, Task, TaskReview
from kollektiv.tools import Toolkit

FILE_TOOLS = ["list_files", "read_file", "write_file", "check_file", "send_message"]
MANAGER_TOOLS = ["hire", "fire", "monitor_agents", "instruct_agent"]


def request(name="assistant", *, nested=False):
    return {
        "name": name,
        "role": "Exact delegated analytical role",
        "instructions": "ORIGINAL PERSONA: reason carefully and produce complete observed evidence.",
        "skills": ["analysis", "precise writing"],
        "tools": FILE_TOOLS + (MANAGER_TOOLS if nested else []),
        "assignment": "Write COMPLETE findings from your actual inputs and supervised child.",
        "output": "notes.md",
        "acceptance_criteria": ["Contains complete findings, rather than placeholders"],
        "input_files": [],
    }


def call(name, arguments):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex, "type": "tool_call"}],
    )


class FamilyModels:
    """Only model decisions are scripted; all state, tools and review inputs are real."""

    def __init__(self, *, nested=False, interrupt=False, hold=None, retire=False):
        self.nested = nested
        self.interrupt = interrupt
        self.interrupted = False
        self.hold = hold
        self.retire = retire
        self.manager = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.prompts = []
        self.reviews = []
        self.tool_sets = {}
        self.writes = Counter()
        self._lock = threading.Lock()

    def system(self, agent):
        return SystemMessage(content=agent)

    def structured(self, agent, schema, prompt, validate=None):
        with self._lock:
            self.prompts.append((agent, schema.__name__, prompt))
        if schema is Contribution:
            result = Contribution(
                findings="Use complete observed artifacts and independent review.",
                messages=[
                    MessageDraft(
                        recipient="coordinator" if agent == "builder" else "builder",
                        content="Use exact assigned artifacts.",
                    )
                ],
            )
        elif schema is Plan:
            result = Plan(
                summary="Supervise one bounded family, then integrate its actual evidence.",
                success_criteria=["A final parent artifact exists with observed findings"],
                tasks=[
                    Task(
                        id="main",
                        title="Integrate delegated evidence",
                        phase="Produce",
                        description="Hire a specialist, observe its outcome, and write the result.",
                        owner="builder",
                        output="combined.md",
                        acceptance_criteria=["Contains the actual supervised outcome"],
                    )
                ],
            )
        elif schema is TaskReview:
            self.reviews.append(prompt)
            result = TaskReview(
                decision="accept",
                feedback="Actual assigned output inspected against its criteria.",
                reflection="Keep supervision and artifact evidence explicit.",
            )
        elif schema is GoalReview:
            self.reviews.append(prompt)
            result = GoalReview(decision="complete", feedback="Actual outputs inspected together.")
        else:
            raise AssertionError(schema)
        if validate:
            validate(result)
        return result

    def _write(self, agent, content):
        self.writes[agent] += 1
        return call("write_file", {"content": content})

    def act(self, agent, messages, tools):
        prompt = messages[1].content
        with self._lock:
            self.prompts.append((agent, "act", prompt))
            self.tool_sets.setdefault(agent, set()).update(tool.name for tool in tools)
        if not prompt.startswith("YOUR TASK:\n"):
            return AIMessage(content="No additional staffing is needed before planning.")
        history = [message for message in messages[2:] if isinstance(message, ToolMessage)]
        used = {message.name for message in history}
        if "write_file" in used:
            if agent == "assistant" and self.interrupt and not self.interrupted:
                self.interrupted = True
                raise ConnectionError("simulated child connection lost AFTER checkpointed write")
            if agent == "leaf" and self.hold == "leaf-after-write":
                self.entered.set()
                assert self.release.wait(10), "test must release the in-flight leaf model call"
                return self._write(agent, "FORBIDDEN LATE OVERWRITE")
            return AIMessage(content="Complete artifact ready for independent review.")
        if agent == "leaf":
            if "hire" not in used:
                # Intentionally forge an unavailable fourth-level hire tool call.
                return call("hire", request("forbidden-fourth-level"))
            return self._write(agent, "# Leaf\nLEAF COMPLETE: verified bounded-family evidence.")
        if agent == "assistant":
            if self.hold == "assistant-before-write" and "read_file" not in used:
                self.entered.set()
                assert self.release.wait(10), "test must release the specialist's first call"
                return call("read_file", {"path": "seed.md"})
            if self.nested:
                if "hire" not in used:
                    return call("hire", request("leaf"))
                leaf = self.manager.get("leaf")
                if leaf["status"] not in ("completed", "retired"):
                    return call("monitor_agents", {"wait_seconds": 0.05})
                if "read_file" not in used:
                    return call("read_file", {"path": leaf["output"]})
            source = next(
                (message.content for message in reversed(history) if message.name == "read_file"),
                "No inputs were required.",
            )
            correction = " ADJUSTED" if "CORRECTED ASSUMPTION" in prompt else ""
            return self._write(agent, f"# Assistant\nCHILD COMPLETE{correction}\n{source}")
        if agent == "builder":
            if "hire" not in used:
                return call("hire", request(nested=self.nested))
            child = self.manager.get("assistant")
            if child["status"] == "interrupted":
                return AIMessage(
                    content="Wait for the interrupted child rather than fabricate work."
                )
            if child["status"] not in ("completed", "retired"):
                return call("monitor_agents", {"wait_seconds": 0.05})
            if child["status"] == "retired" and not child.get("artifact"):
                return self._write(
                    agent, "# Final\nFINAL RETIRED: canceled work was not fabricated."
                )
            if "read_file" not in used:
                return call("read_file", {"path": child["output"]})
            if self.retire and "fire" not in used:
                return call("fire", {"child": "assistant", "reason": "Verified work is complete."})
            content = next(
                message.content for message in reversed(history) if message.name == "read_file"
            )
            return self._write(agent, f"# Final\nFINAL COMPLETE\n{content}")
        raise AssertionError(f"Unexpected acting agent: {agent}")


class ReadOnlyFamilyModels(FamilyModels):
    def __init__(self, child_tools):
        super().__init__()
        self.child_tools = child_tools

    def act(self, agent, messages, tools):
        prompt = messages[1].content
        if not prompt.startswith("YOUR TASK:\n"):
            return super().act(agent, messages, tools)
        with self._lock:
            self.prompts.append((agent, "act", prompt))
            self.tool_sets.setdefault(agent, set()).update(tool.name for tool in tools)
        history = [message for message in messages[2:] if isinstance(message, ToolMessage)]
        used = {message.name for message in history}
        if agent == "assistant":
            if "read_file" in self.child_tools and "read_file" not in used:
                return call("read_file", {"path": "seed.md"})
            source = next(
                (message.content for message in history if message.name == "read_file"),
                "The supplied scenario has exactly two choices.",
            )
            return AIMessage(content=f"READONLY FINDINGS: COMPLETE analysis. {source}")
        if agent == "builder":
            if "write_file" in used:
                return AIMessage(content="Final parent artifact integrates the actual findings.")
            if "hire" not in used:
                child_request = request()
                child_request.update(
                    role="Exact read-only risk analyst",
                    instructions="READONLY PERSONA: reason from the supplied evidence; do not write files.",
                    tools=self.child_tools,
                    output=None,
                    assignment="Analyze a supplied scenario with exactly two choices and report complete findings.",
                    input_files=["seed.md"] if "read_file" in self.child_tools else [],
                )
                return call("hire", child_request)
            child = self.manager.get("assistant")
            if child["status"] != "completed":
                return call("monitor_agents", {"wait_seconds": 0.05})
            return self._write(agent, f"# Final\nFINAL COMPLETE\n{child['summary']}")
        raise AssertionError(agent)


class StaffingFamilyModels(FamilyModels):
    def __init__(self, *, interrupt_preflight_review=False):
        super().__init__()
        self.interrupt_preflight_review = interrupt_preflight_review
        self.preflight_interrupted = False

    def act(self, agent, messages, tools):
        prompt = messages[1].content
        if agent == "scout":
            self.prompts.append((agent, "act", prompt))
            self.tool_sets[agent] = {tool.name for tool in tools}
            return AIMessage(
                content="PREFLIGHT COMPLETE: the scoped mission has one parent artifact."
            )
        if agent == "coordinator" and prompt.startswith("SHARED GOAL:\n"):
            assert all(isinstance(message, BaseMessage) for message in messages), (
                "staffing history must deserialize persisted message dictionaries before model invocation"
            )
            self.prompts.append((agent, "staff.act", prompt))
            history = [message for message in messages[2:] if isinstance(message, ToolMessage)]
            if not any(message.name == "hire" for message in history):
                child_request = request("scout")
                child_request.update(
                    instructions="Analyze mission scope without producing a file.",
                    tools=[],
                    output=None,
                    assignment="Explain the single final parent artifact and preserve the requirements.",
                )
                return call("hire", child_request)
            if self.manager.get("scout")["status"] != "completed":
                return call("monitor_agents", {"wait_seconds": 0.05})
            assert "PREFLIGHT COMPLETE" in prompt
            return AIMessage(content="Use the actual preflight findings when creating the plan.")
        return super().act(agent, messages, tools)

    def structured(self, agent, schema, prompt, validate=None):
        if schema is TaskReview and self.interrupt_preflight_review:
            task_data = json.JSONDecoder().raw_decode(prompt.split("TASK:\n", 1)[1])[0]
            if task_data["owner"] == "scout" and not self.preflight_interrupted:
                self.preflight_interrupted = True
                raise ConnectionError("simulated preflight review connection failure")
        if schema is Plan:
            assert "PREFLIGHT COMPLETE" in prompt
        return super().structured(agent, schema, prompt, validate)


def create_family(tmp_path, models):
    settings = Settings(max_concurrency=1, max_steps=64, discussion_rounds=1)
    settings.team = [agent for agent in settings.team if agent.name != "researcher"]
    settings.agent("builder").tools += MANAGER_TOOLS
    run = Collective.create(
        "Produce an actual supervised family outcome.", tmp_path, settings, models
    )
    models.manager = run.delegation
    return run


def start_runner(run, control=None, on_event=None):
    outcome = {}

    def invoke():
        try:
            outcome["result"] = run.run(control=control, on_event=on_event)
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    return thread, outcome


def finish_runner(thread, outcome, timeout=15):
    thread.join(timeout)
    assert not thread.is_alive(), "real graph execution must finish without nested-worker deadlock"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def test_real_nested_family_reviews_files_and_rejects_forged_fourth_level_hire(tmp_path):
    models = FamilyModels(nested=True, retire=True)
    run = create_family(tmp_path, models)
    control = RunControl()
    thread, outcome = start_runner(run, control)
    try:
        state = finish_runner(thread, outcome)
    finally:
        if thread.is_alive():
            control.stop()
            thread.join(5)
    assert state["status"] == "completed"
    jobs = state["delegation"]["agents"]
    assert set(jobs) == {"assistant", "leaf"}
    assert jobs["assistant"]["parent"] == "builder" and jobs["assistant"]["level"] == 2
    assert jobs["leaf"]["parent"] == "assistant" and jobs["leaf"]["level"] == 3
    assert jobs["assistant"]["status"] == jobs["leaf"]["status"] == "retired"
    assert "hire" in models.tool_sets["assistant"]
    assert not {"hire", "fire"} & models.tool_sets["leaf"]
    child = run.delegation.child_settings("assistant").agent("assistant")
    assert child.role == request(nested=True)["role"]
    assert child.persona == request(nested=True)["instructions"]
    assert child.skills == request(nested=True)["skills"]
    assert child.tools == request(nested=True)["tools"]
    assert models.writes == {"leaf": 1, "assistant": 1, "builder": 1}
    for name in ("assistant", "leaf"):
        assert "COMPLETE" in run.workspace.read(jobs[name]["output"])
        assert any(jobs[name]["output"] in prompt for prompt in models.reviews)
    assert "LEAF COMPLETE" in run.workspace.read("combined.md")
    assert (run.directory / "delegate-checkpoints.sqlite").exists()
    events = [
        json.loads(line) for line in (run.directory / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        event["kind"] == "tool_result"
        and event.get("agent") == "leaf"
        and event.get("tool") == "hire"
        and event.get("status") == "error"
        for event in events
    )
    assert any(event["kind"] == "agent_progress" for event in events)
    inbox = run.delegation.messages_for("builder").values()
    assert any(message.get("system") and message.get("status") == "completed" for message in inbox)
    root_prompts = [
        prompt for agent, kind, prompt in models.prompts if agent == "builder" and kind == "act"
    ]
    assert any(
        '"current_node"' in prompt and '"recent_progress"' in prompt for prompt in root_prompts
    )
    assert any("changed from" in prompt and "completed" in prompt for prompt in root_prompts)


def test_child_checkpointed_write_survives_connection_failure_and_collective_resume(tmp_path):
    models = FamilyModels(interrupt=True)
    run = create_family(tmp_path, models)
    thread, outcome = start_runner(run, RunControl())
    thread.join(15)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), RuntimeError)
    assert "was interrupted" in str(outcome["error"])
    original = run.delegation.get("assistant")
    assert original["status"] == "interrupted"
    assert models.writes["assistant"] == 1
    assert "CHILD COMPLETE" in run.workspace.read(original["output"])
    resumed = Collective.open(run.directory, models=models)
    models.manager = resumed.delegation
    thread, outcome = start_runner(resumed, RunControl())
    state = finish_runner(thread, outcome)
    assert state["status"] == "completed"
    assert resumed.delegation.get("assistant")["id"] == original["id"]
    assert resumed.delegation.get("assistant")["thread_id"] == original["thread_id"]
    assert models.writes["assistant"] == 1
    assert models.writes["builder"] == 1
    assert len(state["delegation"]["agents"]) == 1


def test_parent_instruction_reaches_paused_child_next_prompt_and_artifact(tmp_path):
    models = FamilyModels(hold="assistant-before-write")
    run = create_family(tmp_path, models)
    run.workspace.write("seed.md", "COMPLETE original supplied evidence.")
    control = RunControl()
    thread, outcome = start_runner(run, control)
    try:
        assert models.entered.wait(10)
        control.pause()
        models.release.set()
        assert control.wait_until_paused(10)
        tools = {
            tool.name: tool
            for tool in Toolkit(
                run.settings,
                run.workspace,
                "builder",
                "combined.md",
                run.emit,
                delegation=run.delegation,
                task_id="main",
            ).build()
        }
        result = json.loads(
            tools["instruct_agent"].invoke(
                {
                    "child": "assistant",
                    "content": "CORRECTED ASSUMPTION: mark the final evidence ADJUSTED.",
                }
            )
        )
        assert result["delivered"]
        assert run.delegation.get("assistant")["interventions"]
        control.resume()
        state = finish_runner(thread, outcome)
    finally:
        models.release.set()
        if thread.is_alive():
            control.stop()
            thread.join(5)
    assert state["status"] == "completed"
    assert "ADJUSTED" in run.workspace.read("delegates/assistant/notes.md")
    prompts = [
        prompt for agent, kind, prompt in models.prompts if agent == "assistant" and kind == "act"
    ]
    assert "CORRECTED ASSUMPTION" not in prompts[0]
    assert "CORRECTED ASSUMPTION" in prompts[-1]


def test_firing_live_family_blocks_late_writes_and_preserves_completed_leaf_artifact(tmp_path):
    models = FamilyModels(nested=True, hold="leaf-after-write")
    run = create_family(tmp_path, models)
    control = RunControl()
    thread, outcome = start_runner(run, control)
    try:
        assert models.entered.wait(10)
        leaf = run.delegation.get("leaf")
        original = run.workspace.read(leaf["output"])
        fired = run.delegation.fire("builder", "assistant", "Cancel the whole active family.")
        assert fired["status"] == "retired"
        assert run.delegation.get("leaf")["status"] == "retired"
        models.release.set()
        state = finish_runner(thread, outcome)
    finally:
        models.release.set()
        if thread.is_alive():
            control.stop()
            thread.join(5)
    assert state["status"] == "completed"
    assert run.workspace.read(leaf["output"]) == original
    assert "FORBIDDEN" not in run.workspace.read(leaf["output"])
    assert "delegates/assistant/notes.md" not in run.workspace.list()
    assert "FINAL RETIRED" in run.workspace.read("combined.md")


def test_global_pause_step_stop_and_resume_preserve_child_identity_and_checkpoint(tmp_path):
    models = FamilyModels(hold="assistant-before-write")
    run = create_family(tmp_path, models)
    run.workspace.write("seed.md", "COMPLETE evidence retained across pause and stop.")
    control = RunControl()
    events = []
    thread, outcome = start_runner(run, control, events.append)
    try:
        assert models.entered.wait(10)
        control.pause()
        models.release.set()
        assert control.wait_until_paused(10)
        original = run.delegation.get("assistant")
        started_before = sum(event["kind"] == "node_started" for event in events)
        control.step()
        assert control.wait_until_paused(10)
        started_after = sum(event["kind"] == "node_started" for event in events)
        assert started_after == started_before + 1
        assert not run.workspace.path("combined.md").exists()
        control.stop()
        stopped = finish_runner(thread, outcome)
    finally:
        models.release.set()
        if thread.is_alive():
            control.stop()
            thread.join(5)
    assert stopped["status"] == "stopped"
    assert run.delegation.get("assistant")["status"] == "interrupted"
    resumed = Collective.open(run.directory, models=models)
    models.manager = resumed.delegation
    fresh_control = RunControl(start_paused=True)
    resumed_events = []
    thread, outcome = start_runner(resumed, fresh_control, resumed_events.append)
    try:
        assert fresh_control.wait_until_paused(10)
        assert not any(event["kind"] == "node_started" for event in resumed_events)
        assert resumed.delegation.get("assistant")["id"] == original["id"]
        assert resumed.delegation.get("assistant")["thread_id"] == original["thread_id"]
        fresh_control.resume()
        state = finish_runner(thread, outcome)
    finally:
        if thread.is_alive():
            fresh_control.stop()
            thread.join(5)
    assert state["status"] == "completed"
    assert models.writes["assistant"] == models.writes["builder"] == 1
    assert "COMPLETE evidence retained" in run.workspace.read("combined.md")


@pytest.mark.parametrize("child_tools", [[], ["read_file", "check_file"]])
def test_read_only_child_keeps_exact_permissions_and_returns_reviewed_findings_to_parent(
    tmp_path, child_tools
):
    models = ReadOnlyFamilyModels(child_tools)
    run = create_family(tmp_path, models)
    run.workspace.write("seed.md", "COMPLETE supplied source has exactly two choices.")
    control = RunControl()
    thread, outcome = start_runner(run, control)
    try:
        state = finish_runner(thread, outcome)
    finally:
        if thread.is_alive():
            control.stop()
            thread.join(5)
    assert state["status"] == "completed"
    child = run.delegation.get("assistant")
    assert child["output"] is None
    assert child["status"] == "completed"
    assert child["tools"] == child_tools
    assert models.tool_sets["assistant"] == set(child_tools)
    assert models.writes["assistant"] == 0
    assert "READONLY FINDINGS" in child["summary"]
    assert "READONLY FINDINGS" in run.workspace.read("combined.md")
    assert not any(path.startswith("delegates/") for path in run.workspace.list())
    assert any(
        "ACTUAL FINAL RESPONSE" in prompt and "READONLY FINDINGS" in prompt
        for prompt in models.reviews
    )
    assert any(
        "DELEGATED WORK" in prompt and "READONLY FINDINGS" in prompt for prompt in models.reviews
    )
    assert any(
        message.get("system") and "READONLY FINDINGS" in message["content"]
        for message in run.delegation.messages_for("builder").values()
    )
    assert any(
        agent == "assistant" and kind == "act" and "READONLY PERSONA" in prompt
        for agent, kind, prompt in models.prompts
    )


def test_coordinator_can_hire_a_reviewed_preflight_specialist_before_planning(tmp_path):
    models = StaffingFamilyModels()
    run = create_family(tmp_path, models)
    run.settings.agent("coordinator").tools += MANAGER_TOOLS
    control = RunControl()
    thread, outcome = start_runner(run, control)
    try:
        state = finish_runner(thread, outcome)
    finally:
        if thread.is_alive():
            control.stop()
            thread.join(5)
    assert state["status"] == "completed"
    scout = run.delegation.get("scout")
    assert scout["parent"] == "coordinator"
    assert scout["parent_task_id"] == "discussion:1:coordinator"
    assert scout["status"] == "completed" and scout["output"] is None
    assert models.tool_sets["scout"] == set()
    assert "PREFLIGHT COMPLETE" in scout["summary"]
    events = [
        json.loads(line) for line in (run.directory / "events.jsonl").read_text().splitlines()
    ]
    hired_index = next(
        index
        for index, event in enumerate(events)
        if event["kind"] == "agent_hired" and event["name"] == "scout"
    )
    plan_index = next(
        index for index, event in enumerate(events) if event["kind"] == "plan_created"
    )
    assert hired_index < plan_index
    assert any(
        "ACTUAL FINAL RESPONSE" in prompt and "PREFLIGHT COMPLETE" in prompt
        for prompt in models.reviews
    )


def test_preplanning_hire_checkpoint_resumes_same_specialist_without_repeating_findings(tmp_path):
    models = StaffingFamilyModels(interrupt_preflight_review=True)
    run = create_family(tmp_path, models)
    run.settings.agent("coordinator").tools += MANAGER_TOOLS
    thread, outcome = start_runner(run, RunControl())
    thread.join(15)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), RuntimeError)
    assert "Subagent scout was interrupted" in str(outcome["error"])
    original = run.delegation.get("scout")
    assert original["status"] == "interrupted"
    resumed = Collective.open(run.directory, models=models)
    models.manager = resumed.delegation
    thread, outcome = start_runner(resumed, RunControl())
    state = finish_runner(thread, outcome)
    assert state["status"] == "completed"
    scout = resumed.delegation.get("scout")
    assert scout["id"] == original["id"] and scout["thread_id"] == original["thread_id"]
    assert scout["status"] == "completed"
    assert sum(agent == "scout" and kind == "act" for agent, kind, _ in models.prompts) == 1
    assert "PREFLIGHT COMPLETE" in scout["summary"]
