"""Exercise real LangGraph scheduling, native ToolNode calls, and SQLite recovery."""

import json
from collections import Counter
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from kollektiv import Collective, Settings
from kollektiv.models import Contribution, GoalReview, MessageDraft, Plan, Task, TaskReview


def task(task_id="design", output="design.md", dependencies=(), owner="researcher"):
    return Task(
        id=task_id,
        title=task_id,
        phase="Build",
        description=f"Create {output}",
        owner=owner,
        output=output,
        dependencies=list(dependencies),
        acceptance_criteria=["Contains COMPLETE content consistent with dependency files"],
    )


class ScriptedModels:
    """Deterministic model responses, with the production graph and tools left intact."""

    def __init__(
        self,
        *,
        reject_once=False,
        always_reject=False,
        interrupt=None,
        revise=False,
        recipient="researcher",
    ):
        self.writes = Counter()
        self.calls = Counter()
        self.prompts = []
        self.reject_once = reject_once
        self.always_reject = always_reject
        self.interrupt = interrupt
        self.interrupted = False
        self.revise = revise
        self.recipient = recipient

    def system(self, agent):
        return SystemMessage(content=agent)

    def structured(self, agent, schema, prompt, validate=None):
        self.calls[schema.__name__] += 1
        self.prompts.append((agent, schema.__name__, prompt))
        if schema is Contribution:
            target = self.recipient if agent == "builder" else "builder"
            result = Contribution(
                findings=f"{agent} recommends consistent COMPLETE artifacts.",
                messages=[MessageDraft(recipient=target, content=f"HELLO FROM {agent}")],
            )
        elif schema is Plan:
            result = Plan(
                summary="Build design then a consistent story",
                success_criteria=["Both files complete"],
                tasks=[task(), task("story", "story.md", ["design"], "builder")],
            )
        elif schema is TaskReview:
            rejected = "\nBAD\n" in prompt
            result = TaskReview(
                decision="revise" if rejected else "accept",
                feedback="Replace BAD with COMPLETE" if rejected else "Verified COMPLETE content",
                reflection="Follow the shared design and read dependencies.",
            )
        elif schema is GoalReview:
            if self.revise and self.calls["GoalReview"] == 1:
                result = GoalReview(
                    decision="revise",
                    feedback="Strengthen design; add an integration summary.",
                    revise_tasks=["design"],
                    additional_tasks=[task("summary", "summary.md", ["story"], "builder")],
                )
            else:
                result = GoalReview(
                    decision="complete", feedback="All required artifacts inspected and consistent."
                )
        else:
            raise AssertionError(schema)
        if validate:
            validate(result)
        return result

    def act(self, agent, messages, tools):
        prompt = messages[1].content
        task_data = json.JSONDecoder().raw_decode(prompt.split("YOUR TASK:\n", 1)[1])[0]
        task_id = task_data["id"]
        self.calls[f"act:{task_id}"] += 1
        self.prompts.append((agent, "act", prompt))
        history = messages[2:]
        # Keep each attempt's tool sequence distinct after reviewer feedback.
        last_reset = max(
            (i for i, m in enumerate(history) if isinstance(m, HumanMessage)), default=-1
        )
        recent = history[last_reset + 1 :]
        written = any(isinstance(m, ToolMessage) and m.name == "write_file" for m in recent)
        read = any(isinstance(m, ToolMessage) and m.name == "read_file" for m in recent)
        if written:
            if self.interrupt == task_id and not self.interrupted:
                self.interrupted = True
                raise ConnectionError("simulated model connection lost after writing")
            return AIMessage(content="Artifact created. Ready for independent review.")
        if task_data.get("input_files") and not read:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": task_data["input_files"][0]},
                        "id": uuid4().hex,
                        "type": "tool_call",
                    }
                ],
            )
        self.writes[task_id] += 1
        bad = self.always_reject or (self.reject_once and self.writes[task_id] == 1)
        content = "BAD" if bad else f"# {task_id}\nCOMPLETE\nVersion {self.writes[task_id]}"
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"content": content},
                    "id": uuid4().hex,
                    "type": "tool_call",
                },
                {
                    "name": "send_message",
                    "args": {"recipient": "all", "content": f"DECISION {task_id}"},
                    "id": uuid4().hex,
                    "type": "tool_call",
                },
                {
                    "name": "share_note",
                    "args": {"content": f"MEMORY {task_id}"},
                    "id": uuid4().hex,
                    "type": "tool_call",
                },
            ],
        )


def make_run(tmp_path, models, **kwargs):
    return Collective.create(
        "Create design.md and a consistent story.md", tmp_path, Settings(**kwargs), models=models
    )


def test_team_messages_dependency_inputs_and_review_repairs(tmp_path):
    models = ScriptedModels(reject_once=True)
    run = make_run(tmp_path, models, max_concurrency=2)
    state = run.run()
    assert state["status"] == "completed"
    assert models.writes == {"design": 2, "story": 2}
    assert all(t["status"] == "completed" for t in state["tasks"].values())
    builder_round_two = [
        p for a, s, p in models.prompts if a == "builder" and s == "Contribution" and "round 2" in p
    ][0]
    assert "HELLO FROM researcher" in builder_round_two
    story_prompt = next(p for a, s, p in models.prompts if s == "act" and '"id": "story"' in p)
    assert '"input_files": [\n    "design.md"' in story_prompt
    assert "DECISION design" in story_prompt
    assert "MEMORY design" in story_prompt
    review = next(p for a, s, p in models.prompts if s == "TaskReview" and '"id": "story"' in p)
    assert "FILE design.md" in review and "FILE story.md" in review
    events = [
        json.loads(line) for line in (run.directory / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["kind"] == "task_review" and not e["accepted"] for e in events)
    assert (run.directory / "checkpoints.sqlite").is_file()
    assert (run.directory / "report.md").is_file()
    assert json.loads((run.directory / "state.json").read_text())["status"] == "completed"


def test_resume_after_tool_write_preserves_completed_tasks_and_worker_checkpoint(tmp_path):
    models = ScriptedModels(interrupt="story")
    run = make_run(tmp_path, models)
    with pytest.raises(ConnectionError):
        run.run()
    assert run.state()["tasks"]["design"]["status"] == "completed"
    assert run.workspace.path("story.md").is_file()
    before = models.calls["Contribution"]
    reopened = Collective.open(run.directory, models=models)
    result = reopened.run()
    assert result["status"] == "completed"
    assert models.calls["Contribution"] == before
    assert models.writes == {"design": 1, "story": 1}
    calls = models.calls.copy()
    assert reopened.run()["status"] == "completed"
    assert models.calls == calls


def test_goal_revisions_invalidate_descendants_and_add_missing_deliverables(tmp_path):
    models = ScriptedModels(revise=True)
    run = make_run(tmp_path, models)
    result = run.run()
    assert result["status"] == "completed"
    assert result["goal_revision"] == 1
    assert models.writes == {"design": 2, "story": 2, "summary": 1}
    assert result["tasks"]["summary"]["dependencies"] == ["story"]
    assert len(result["plan"]["tasks"]) == 3


def test_exhausted_attempts_report_incomplete_and_retry_is_explicit(tmp_path):
    models = ScriptedModels(always_reject=True)
    run = make_run(tmp_path, models, max_task_attempts=2)
    result = run.run()
    assert result["status"] == "incomplete"
    assert result["tasks"]["design"]["status"] == "failed"
    assert result["tasks"]["story"]["status"] == "pending"
    assert models.calls["GoalReview"] == 0
    assert models.writes["design"] == 2
    calls = models.calls.copy()
    assert Collective.open(run.directory, models=models).run()["status"] == "incomplete"
    assert models.calls == calls
    models.always_reject = False
    assert Collective.open(run.directory, models=models).run(retry=True)["status"] == "completed"


def test_changed_accepted_file_rebuilds_it_and_dependents(tmp_path):
    models = ScriptedModels()
    run = make_run(tmp_path, models)
    assert run.run()["status"] == "completed"
    run.workspace.write("design.md", "Someone changed the accepted design")
    result = Collective.open(run.directory, models=models).run()
    assert result["status"] == "completed"
    assert models.writes == {"design": 2, "story": 2}
    assert result["tasks"]["design"]["artifact"] == run.workspace.inspect("design.md")


@pytest.mark.parametrize("dependencies", [["missing"], ["design"]])
def test_plan_rejects_unknown_and_self_dependencies(dependencies):
    with pytest.raises(ValidationError):
        Plan(summary="x", success_criteria=["x"], tasks=[task(dependencies=dependencies)])


def test_plan_rejects_cycles_duplicate_paths_and_directory_conflicts():
    for tasks in (
        [task(dependencies=["story"]), task("story", "story.md", ["design"])],
        [task(), task("other", "DESIGN.md")],
        [task(output="chapters"), task("other", "chapters/story.md")],
    ):
        with pytest.raises(ValidationError):
            Plan(summary="x", success_criteria=["x"], tasks=tasks)


def test_runs_with_same_goal_have_isolated_artifacts_and_checkpoints(tmp_path):
    first = make_run(tmp_path, ScriptedModels())
    second = make_run(tmp_path, ScriptedModels())
    assert first.directory != second.directory
    first.run()
    assert second.state()["status"] == "created"
    assert not second.workspace.list()


def test_custom_personas_and_ownership_validation(tmp_path):
    settings = Settings()
    settings.team[1].name = "architect"
    settings.team[1].persona = "Prefer compact architectural notes."
    # Built-in owners are rejected when they do not exist in a custom roster.
    run = Collective.create(
        "A goal", tmp_path, settings, models=ScriptedModels(recipient="architect")
    )
    with pytest.raises(ValueError, match="owners"):
        run.run()
    assert run.state()["status"] == "discussing"


def test_parallel_workers_join_without_losing_messages_memory_or_artifacts(tmp_path):
    import threading

    barrier = threading.Barrier(2)

    class ParallelModels(ScriptedModels):
        def structured(self, agent, schema, prompt, validate=None):
            if schema is not Plan:
                return super().structured(agent, schema, prompt, validate)
            result = Plan(
                summary="Parallel design and style, then story",
                success_criteria=["Three complete artifacts"],
                tasks=[
                    task(),
                    task("style", "style.md", owner="builder"),
                    task("story", "story.md", ["design", "style"], "builder"),
                ],
            )
            if validate:
                validate(result)
            return result

        def act(self, agent, messages, tools):
            data = json.JSONDecoder().raw_decode(messages[1].content.split("YOUR TASK:\n", 1)[1])[0]
            if data["id"] in ("design", "style") and not self.calls[f"act:{data['id']}"]:
                barrier.wait(timeout=5)
            return super().act(agent, messages, tools)

    models = ParallelModels()
    run = make_run(tmp_path, models, max_concurrency=2)
    result = run.run()
    assert result["status"] == "completed"
    assert set(result["results"]) == {"design", "style", "story"}
    assert models.writes == {"design": 1, "style": 1, "story": 1}
    assert {n["content"] for n in result["notes"].values()} >= {"MEMORY design", "MEMORY style"}
    prompt = next(p for _, s, p in models.prompts if s == "act" and '"id": "story"' in p)
    assert "MEMORY design" in prompt and "MEMORY style" in prompt
