"""Small teams validate and execute without hidden coordinator/reviewer agents."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from kollektiv import Collective, Settings
from kollektiv.config import Agent
from kollektiv.models import Contribution, GoalReview, MessageDraft, Plan, Task, TaskReview


def test_single_agent_infers_both_roles_and_two_agents_keep_explicit_roles():
    agent = Agent(name="orchestrator", role="Orchestrator", persona="Complete the user's task.")
    single = Settings(team=[agent])
    assert single.coordinator == single.reviewer == "orchestrator"
    assert single.task_owners() == ["orchestrator"]
    with pytest.raises(ValidationError, match="coordinator and reviewer"):
        Settings(team=[agent], reviewer="missing")
    with pytest.raises(ValidationError, match="at least 1"):
        Settings(team=[])
    reviewer = Agent(
        name="critic", role="Reviewer", persona="Review the actual result.", tools=["read_file"]
    )
    pair = Settings(team=[agent, reviewer], coordinator=agent.name, reviewer=reviewer.name)
    assert pair.task_owners() == [agent.name]


class SingleAgentModels:
    def __init__(self):
        self.actors = set()
        self.reviews = []

    def system(self, agent):
        self.actors.add(agent)
        return SystemMessage(content=agent)

    def structured(self, agent, schema, prompt, validate=None):
        self.actors.add(agent)
        if schema is Contribution:
            result = Contribution(
                findings="Produce the requested complete file.",
                messages=[MessageDraft(recipient="all", content="Use result.md.")],
            )
        elif schema is Plan:
            result = Plan(
                summary="Produce one complete file.",
                success_criteria=["Actual file is complete"],
                tasks=[
                    Task(
                        id="result",
                        title="Result",
                        phase="Produce",
                        description="Write the complete result.",
                        owner=agent,
                        output="result.md",
                        acceptance_criteria=["Contains COMPLETE evidence"],
                    )
                ],
            )
        elif schema is TaskReview:
            self.reviews.append(prompt)
            assert "fresh-context self-review" in prompt
            assert "FILE result.md\nCOMPLETE evidence" in prompt
            result = TaskReview(
                decision="accept",
                feedback="Actual file checked in a fresh self-review.",
                reflection="Review the real artifact.",
            )
        elif schema is GoalReview:
            assert "COMPLETE evidence" in prompt
            result = GoalReview(decision="complete", feedback="Actual requested file inspected.")
        else:
            raise AssertionError(schema)
        if validate:
            validate(result)
        return result

    def act(self, agent, messages, tools):
        self.actors.add(agent)
        if any(
            isinstance(message, ToolMessage) and message.name == "write_file"
            for message in messages
        ):
            return AIMessage(content="Actual complete result is ready for review.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"content": "COMPLETE evidence"},
                    "id": uuid4().hex,
                    "type": "tool_call",
                }
            ],
        )


def test_one_plain_author_executes_and_resumes_with_explicit_self_review(tmp_path):
    models = SingleAgentModels()
    settings = Settings(
        team=[
            Agent(name="solo", role="Solo author", persona="Write and check the requested result.")
        ],
        discussion_rounds=1,
    )
    run = Collective.create(
        "Write result.md with COMPLETE evidence.", tmp_path, settings, models=models
    )
    assert run.run()["status"] == "completed"
    assert models.actors == {"solo"}
    assert run.workspace.read("result.md") == "COMPLETE evidence"
    assert len(models.reviews) == 1
    assert Collective.open(run.directory, models=models).run()["status"] == "completed"
    assert len(models.reviews) == 1
