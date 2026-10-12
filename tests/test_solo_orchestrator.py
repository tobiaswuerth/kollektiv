"""A singleton orchestrator creates every author and integrates accepted child work."""

import json
from collections import Counter
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from kollektiv import Collective, Settings
from kollektiv.config import Agent
from kollektiv.models import Contribution, GoalReview, Plan, Task, TaskReview

MANAGER_TOOLS = [
    "list_files",
    "read_file",
    "check_file",
    "assemble_files",
    "hire",
    "fire",
    "monitor_agents",
    "instruct_agent",
    "send_message",
]
CHILD_CONTENT = "# Specialist findings\n\nOne orchestrator hired this writer dynamically. A -> B -> C is acyclic.\n"
FINAL_PREFIX = "# Final result\n\n"


def tool_call(name, args):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": uuid4().hex, "type": "tool_call"}],
    )


class SoloModels:
    def __init__(self, *, preflight=False, interrupt=False):
        self.preflight = preflight
        self.interrupt = interrupt
        self.interrupted = False
        self.manager = None
        self.agent_calls = set()
        self.tool_sets = {}
        self.write_decisions = Counter()
        self.reviews = []

    def system(self, agent):
        self.agent_calls.add(agent)
        return SystemMessage(content=agent)

    def structured(self, agent, schema, prompt, validate=None):
        self.agent_calls.add(agent)
        assert agent == "orchestrator"
        if schema is Contribution:
            result = Contribution(
                findings="Hire the exact required specialist, then integrate reviewed actual work.",
                messages=[],
            )
        elif schema is Plan:
            result = Plan(
                summary="One orchestrator owns integration; all authors are hired dynamically.",
                success_criteria=["result.md contains the approved writer's complete text"],
                tasks=[
                    Task(
                        id="integrate",
                        title="Integrate hired writer",
                        phase="Deliver",
                        description="Hire, supervise and retire a writer; assemble the actual approved child file into result.md.",
                        owner="orchestrator",
                        output="result.md",
                        acceptance_criteria=[
                            "Preserves the full approved child artifact verbatim after the final heading."
                        ],
                    )
                ],
            )
        elif schema is TaskReview:
            self.reviews.append(prompt)
            assert CHILD_CONTENT.rstrip() in prompt
            result = TaskReview(
                decision="accept",
                feedback="Actual complete file inspected against the full criteria.",
                reflection="Only accepted child evidence is used in integration.",
            )
        elif schema is GoalReview:
            self.reviews.append(prompt)
            assert FINAL_PREFIX + CHILD_CONTENT.rstrip() in prompt
            result = GoalReview(
                decision="complete",
                feedback="The actual result contains the complete delegated work; all requirements were checked.",
            )
        else:
            raise AssertionError(schema)
        if validate:
            validate(result)
        return result

    @staticmethod
    def child_request():
        return {
            "name": "scribe",
            "role": "Precisely scoped author",
            "instructions": "Write the complete observed result. Do not hire or fire anyone.",
            "skills": ["writing", "graph analysis"],
            "tools": ["read_file", "write_file", "check_file"],
            "assignment": "Explain that one orchestrator hired the writer and give the acyclic graph A -> B -> C.",
            "output": "source.md",
            "input_files": [],
            "acceptance_criteria": [
                "Contains the complete orchestrator explanation and graph example."
            ],
        }

    def act(self, agent, messages, tools):
        self.agent_calls.add(agent)
        self.tool_sets.setdefault(agent, set()).update(tool.name for tool in tools)
        prompt = messages[1].content
        history = [message for message in messages[2:] if isinstance(message, ToolMessage)]
        used = {message.name for message in history}
        if agent == "scribe":
            if "write_file" not in used:
                self.write_decisions[agent] += 1
                return tool_call("write_file", {"content": CHILD_CONTENT})
            if self.interrupt and not self.interrupted:
                self.interrupted = True
                raise ConnectionError("Singleton test: interrupted after durable child write")
            return AIMessage(content="The actual specialist output is ready for review.")
        assert agent == "orchestrator"
        is_staff = not prompt.startswith("YOUR TASK:\n")
        if is_staff and not self.preflight:
            return AIMessage(content="Hire the writer during the final integration task.")
        if "scribe" not in self.manager.snapshot()["agents"]:
            return tool_call("hire", self.child_request())
        child = self.manager.get("scribe")
        if child["status"] == "interrupted":
            return AIMessage(content="Actual child work was interrupted; preserve and resume it.")
        if child["status"] not in {"completed", "retired"}:
            return AIMessage(content="Waiting for actual accepted child work.")
        if is_staff:
            if child["status"] != "retired":
                return tool_call(
                    "fire",
                    {
                        "child": "scribe",
                        "reason": "Accepted preflight work is available for integration.",
                    },
                )
            return AIMessage(
                content="Accepted preflight source is available; plan its integration into result.md."
            )
        if "assemble_files" not in used:
            assert "assemble_files" in {tool.name for tool in tools}
            return tool_call("assemble_files", {"paths": [child["output"]], "prefix": FINAL_PREFIX})
        if child["status"] != "retired":
            return tool_call(
                "fire", {"child": "scribe", "reason": "Accepted source has been integrated."}
            )
        return AIMessage(content="Complete actual result.md assembled and the worker retired.")


def solo_settings():
    return Settings(
        team=[
            Agent(
                name="orchestrator",
                role="General orchestrator",
                persona="Hire exactly the specialists necessary, supervise their actual work and integrate accepted artifacts.",
                skills=["planning", "hire", "fire", "integration"],
                tools=MANAGER_TOOLS,
            )
        ],
        coordinator="orchestrator",
        reviewer="orchestrator",
        communication_links=[],
        discussion_rounds=1,
        max_concurrency=1,
        max_steps=20,
        max_tasks=4,
        max_task_attempts=1,
        max_goal_revisions=0,
    )


@pytest.mark.parametrize("preflight", [False, True])
def test_single_orchestrator_hires_every_author_and_assembles_accepted_work(tmp_path, preflight):
    models = SoloModels(preflight=preflight)
    run = Collective.create(
        "Deliver result.md using a dynamically hired writer, preserving the accepted source verbatim.",
        tmp_path,
        solo_settings(),
        models=models,
    )
    models.manager = run.delegation
    result = run.run()
    assert result["status"] == "completed"
    assert [agent.name for agent in run.settings.team] == ["orchestrator"]
    assert models.agent_calls == {"orchestrator", "scribe"}
    assert "write_file" not in models.tool_sets["orchestrator"]
    assert {
        "hire",
        "fire",
        "monitor_agents",
        "instruct_agent",
        "assemble_files",
    } <= models.tool_sets["orchestrator"]
    assert models.tool_sets["scribe"] == {"read_file", "write_file", "check_file"}
    assert models.write_decisions == {"scribe": 1}
    assert run.workspace.read("result.md") == FINAL_PREFIX + CHILD_CONTENT
    child = run.delegation.get("scribe")
    assert child["status"] == "retired"
    assert child["result"]["status"] == "completed"
    assert run.workspace.read(child["output"]) == CHILD_CONTENT
    assert run.delegation.approved_child_inputs("orchestrator", "integrate") == [child["output"]]
    assert len(models.reviews) == 3  # Child review, integration review, full goal audit.
    events = [
        json.loads(line)
        for line in (run.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert Counter(event["agent"] for event in events if event["kind"] == "artifact_written") == {
        "scribe": 1,
        "orchestrator": 1,
    }
    assert result["tasks"]["integrate"]["owner"] == "orchestrator"
    assert result["tasks"]["integrate"]["status"] == "completed"


def test_single_orchestrator_resumes_its_hired_writer_without_duplicate_work(tmp_path):
    models = SoloModels(interrupt=True)
    run = Collective.create(
        "Deliver result.md from an accepted dynamically hired source.",
        tmp_path,
        solo_settings(),
        models=models,
    )
    models.manager = run.delegation
    with pytest.raises(RuntimeError, match="Subagent scribe was interrupted"):
        run.run()
    original = run.delegation.get("scribe")
    assert original["status"] == "interrupted"
    assert run.workspace.read(original["output"]) == CHILD_CONTENT
    resumed = Collective.open(run.directory, models=models)
    models.manager = resumed.delegation
    result = resumed.run()
    assert result["status"] == "completed"
    assert resumed.delegation.get("scribe")["id"] == original["id"]
    assert len(resumed.delegation.snapshot()["agents"]) == 1
    assert models.write_decisions == {"scribe": 1}
    assert resumed.workspace.read("result.md") == FINAL_PREFIX + CHILD_CONTENT
    events = [
        json.loads(line)
        for line in (run.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert Counter(event["agent"] for event in events if event["kind"] == "artifact_written") == {
        "scribe": 1,
        "orchestrator": 1,
    }
