"""Large manuscripts use real scoped tools, independent reviews and recovery."""

import hashlib
import json
from collections import Counter
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage, messages_from_dict, messages_to_dict
from pydantic import ValidationError
from test_collective import ScriptedModels

from kollektiv import Collective, Settings
from kollektiv.models import GoalReview, Plan, Task, TaskReview
from kollektiv.tools import Toolkit
from kollektiv.workflow import Workflow
from kollektiv.workspace import Workspace


def assembly_settings(**overrides):
    settings = Settings(**overrides)
    settings.agent("builder").tools.append("assemble_files")
    return settings


def assembly_tool(workspace, *, output="manuscript.md", inputs=None, settings=None, events=None):
    toolkit = Toolkit(
        settings or assembly_settings(),
        workspace,
        "builder",
        output,
        lambda kind, **fields: (events if events is not None else []).append(
            {"kind": kind, **fields}
        ),
        input_files=inputs or [],
    )
    return {tool.name: tool for tool in toolkit.build()}


def chapter(index):
    return f"# Chapter {index}\n\n" + " ".join(f"c{index}w{word}" for word in range(1250)) + "\n"


def test_assembly_preserves_entire_long_manuscript_order_and_reports_only_metrics(tmp_path):
    workspace = Workspace(tmp_path)
    contents = [chapter(index) for index in range(1, 11)]
    paths = [f"chapters/chapter_{index:02}.md" for index in range(1, 11)]
    for path, content in zip(paths, contents, strict=True):
        workspace.write(path, content)
    events = []
    tool = assembly_tool(workspace, inputs=paths, events=events)["assemble_files"]
    # Plan dependency ordering and publication ordering may differ. Backslashes are
    # normalized, while their exact complete text and requested order are retained.
    ordered = [paths[9], *paths[:9]]
    prefix = "# The Clockwork Archive\n\n"
    separator = "\n\n---\n\n"
    result_text = tool.invoke(
        {
            "paths": [path.replace("/", "\\") for path in ordered],
            "prefix": prefix,
            "separator": separator,
        }
    )
    expected = prefix + separator.join([contents[9], *contents[:9]])
    assert workspace.read("manuscript.md") == expected
    metrics = json.loads(result_text)
    assert metrics["words"] >= 12500
    assert metrics["sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert metrics["bytes"] == len(expected.encode())
    assert len(result_text) < 500
    assert "c10w1249" not in result_text
    assert events == [
        {"kind": "artifact_written", "agent": "builder", "sources": ordered, **metrics}
    ]


@pytest.mark.parametrize(
    ("paths", "inputs", "error"),
    [
        (["other.md"], ["one.md"], "dependency"),
        (["../outside.md"], ["one.md"], "safe relative"),
        (["D:/outside.md"], ["one.md"], "safe relative"),
        (["one.md", "one.md"], ["one.md"], "unique"),
        (["one.md", "ONE.md"], ["one.md", "ONE.md"], "unique"),
        (["manuscript.md"], ["manuscript.md"], "own assigned output"),
        (["MANUSCRIPT.md"], ["MANUSCRIPT.md"], "own assigned output"),
    ],
)
def test_invalid_assembly_never_changes_the_existing_output(tmp_path, paths, inputs, error):
    workspace = Workspace(tmp_path)
    workspace.write("one.md", "A valid source")
    original = workspace.write("manuscript.md", "Preserve the prior manuscript")
    events = []
    tool = assembly_tool(workspace, inputs=inputs, events=events)["assemble_files"]
    with pytest.raises(ValueError, match=error):
        tool.invoke({"paths": paths})
    assert workspace.inspect("manuscript.md") == original
    assert events == []


def test_missing_or_oversize_sources_leave_existing_output_intact(tmp_path):
    workspace = Workspace(tmp_path, max_bytes=1024)
    workspace.write("one.md", "a" * 700)
    workspace.write("two.md", "b" * 700)
    original = workspace.write("manuscript.md", "Prior version")
    events = []
    tool = assembly_tool(workspace, inputs=["one.md", "two.md", "missing.md"], events=events)[
        "assemble_files"
    ]
    with pytest.raises(ValueError, match="exceeds"):
        tool.invoke({"paths": ["one.md", "two.md"]})
    with pytest.raises(OSError):
        tool.invoke({"paths": ["one.md", "missing.md"]})
    assert workspace.inspect("manuscript.md") == original
    assert events == []
    assert not list(workspace.root.rglob(".kollektiv-*"))


def test_assembly_sources_cannot_escape_through_a_dependency_symlink(tmp_path):
    workspace = Workspace(tmp_path / "artifacts")
    outside = tmp_path / "outside.md"
    outside.write_text("Outside data", encoding="utf-8")
    try:
        (workspace.root / "source.md").symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit creation of symlinks")
    original = workspace.write("manuscript.md", "Prior version")
    tool = assembly_tool(workspace, inputs=["source.md"])["assemble_files"]
    with pytest.raises(ValueError, match="leaves"):
        tool.invoke({"paths": ["source.md"]})
    assert workspace.inspect("manuscript.md") == original


def test_assembly_respects_tool_permissions_task_scope_and_strict_arguments(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.write("one.md", "Source")
    assert "assemble_files" not in assembly_tool(workspace, output=None, inputs=["one.md"])
    assert "assemble_files" not in assembly_tool(workspace, inputs=[])
    assert "assemble_files" not in assembly_tool(workspace, inputs=["one.md"], settings=Settings())
    tool = assembly_tool(workspace, inputs=["one.md"])["assemble_files"]
    for arguments in (
        {"paths": []},
        {"paths": ["one.md"], "output": "someone_elses.md"},
        {"paths": ["one.md"], "prefix": "x" * 50001},
    ):
        with pytest.raises(ValidationError):
            tool.invoke(arguments)
    assert not workspace.path("manuscript.md").exists()


@pytest.mark.parametrize(
    "names", [("write_file", "assemble_files"), ("assemble_files", "assemble_files")]
)
def test_one_output_write_per_step_includes_assembly(tmp_path, names):
    workspace = Workspace(tmp_path)
    workspace.write("one.md", "Source")
    original = workspace.write("manuscript.md", "Prior version")
    events = []

    def emit(kind, **fields):
        event = {"id": uuid4().hex, "kind": kind, **fields}
        events.append(event)
        return event

    workflow = Workflow(assembly_settings(), workspace, emit)
    reply = AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": {"content": "Overwrite"} if name == "write_file" else {"paths": ["one.md"]},
                "id": uuid4().hex,
                "type": "tool_call",
            }
            for name in names
        ],
    )
    result = workflow.execute(
        {
            "task": {
                "id": "publish",
                "owner": "builder",
                "output": "manuscript.md",
                "input_files": ["one.md"],
            },
            "reply": messages_to_dict([reply])[0],
        }
    )
    assert workspace.inspect("manuscript.md") == original
    messages = messages_from_dict(result["history"])
    assert len(messages) == 2
    assert all(message.status == "error" and "only one" in message.content for message in messages)
    assert not any(event["kind"] == "artifact_written" for event in events)


class LongManuscriptModels(ScriptedModels):
    """Supply chapter text; leave real ToolNode writes, reviews and persistence intact."""

    paths = [f"chapters/chapter_{index:02}.md" for index in range(1, 11)]
    contents = [chapter(index) for index in range(1, 11)]
    prefix = "# The Clockwork Archive\n\n"

    def __init__(self, *, interrupt_after_assembly=False):
        super().__init__()
        self.interrupt_after_assembly = interrupt_after_assembly
        self.assembly_calls = 0
        self.reviewed_manuscripts = []

    @property
    def manuscript(self):
        return self.prefix + "\n\n".join(self.contents)

    def structured(self, agent, schema, prompt, validate=None):
        if schema is Plan:
            tasks = [
                Task(
                    id=f"chapter_{index:02}",
                    title=f"Chapter {index}",
                    phase="Chapters",
                    description=f"Write chapter {index}",
                    owner="builder",
                    output=path,
                    acceptance_criteria=["Contains at least 1250 words of complete chapter text"],
                )
                for index, path in enumerate(self.paths, 1)
            ]
            result = Plan(
                summary="Write ten chapters then assemble the complete manuscript",
                success_criteria=["Complete manuscript of at least 12500 words"],
                tasks=[
                    *tasks,
                    Task(
                        id="publish",
                        title="Publish complete manuscript",
                        phase="Publication",
                        description="Assemble all approved chapters in numeric order",
                        owner="builder",
                        output="manuscript.md",
                        dependencies=[task.id for task in tasks],
                        acceptance_criteria=[
                            "All ten complete chapters in order; at least 12500 words"
                        ],
                    ),
                ],
            )
        elif schema is TaskReview:
            if '"id": "publish"' in prompt:
                assert f"FILE manuscript.md\n{self.manuscript}\nEND FILE manuscript.md" in prompt
                for path, content in zip(self.paths, self.contents, strict=True):
                    assert f"FILE {path}\n{content}\nEND FILE {path}" in prompt
                self.reviewed_manuscripts.append(prompt)
            result = TaskReview(
                decision="accept",
                feedback="Full actual file meets its criteria.",
                reflection="Assemble complete approved chapter files.",
            )
        elif schema is GoalReview:
            assert f"FILE manuscript.md\n{self.manuscript}\nEND FILE manuscript.md" in prompt
            for path, content in zip(self.paths, self.contents, strict=True):
                assert f"FILE {path}\n{content}\nEND FILE {path}" in prompt
            self.reviewed_manuscripts.append(prompt)
            result = GoalReview(
                decision="complete", feedback="All ten chapters and full manuscript inspected."
            )
        else:
            return super().structured(agent, schema, prompt, validate)
        self.calls[schema.__name__] += 1
        self.prompts.append((agent, schema.__name__, prompt))
        if validate:
            validate(result)
        return result

    def act(self, agent, messages, tools):
        data = json.JSONDecoder().raw_decode(messages[1].content.split("YOUR TASK:\n", 1)[1])[0]
        self.calls[f"act:{data['id']}"] += 1
        history = messages[2:]
        if any(
            isinstance(message, ToolMessage)
            and message.name == "assemble_files"
            and message.status != "error"
            for message in history
        ):
            if self.interrupt_after_assembly and not self.interrupted:
                self.interrupted = True
                raise ConnectionError("interrupted after checkpointed assembly")
            return AIMessage(content="Complete manuscript assembled; ready for review.")
        if any(
            isinstance(message, ToolMessage) and message.name == "write_file" for message in history
        ):
            return AIMessage(content="Complete chapter written; ready for review.")
        if data["id"] == "publish":
            assert "assemble_files" in {tool.name for tool in tools}
            assert data["input_files"] == self.paths
            self.assembly_calls += 1
            name, arguments = "assemble_files", {"paths": self.paths, "prefix": self.prefix}
        else:
            assert "assemble_files" not in {tool.name for tool in tools}
            index = self.paths.index(data["output"])
            self.writes[data["id"]] += 1
            name, arguments = "write_file", {"content": self.contents[index]}
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": arguments,
                    "id": uuid4().hex,
                    "type": "tool_call",
                }
            ],
        )


def test_large_manuscript_real_graph_full_reviews_and_resume_without_reassembly(tmp_path):
    models = LongManuscriptModels(interrupt_after_assembly=True)
    settings = assembly_settings(
        max_tasks=12, discussion_rounds=1, context_chars=350000, num_ctx=131072
    )
    run = Collective.create(
        "Write a complete 50-page, 12500-word novel in manuscript.md",
        tmp_path,
        settings,
        models=models,
    )
    with pytest.raises(ConnectionError, match="checkpointed assembly"):
        run.run()
    assert all(
        task["status"] == "completed"
        for task_id, task in run.state()["tasks"].items()
        if task_id != "publish"
    )
    assert run.workspace.read("manuscript.md") == models.manuscript
    reopened = Collective.open(run.directory, models=models)
    state = reopened.run()
    assert state["status"] == "completed"
    assert models.assembly_calls == 1
    assert models.writes == Counter({f"chapter_{index:02}": 1 for index in range(1, 11)})
    info = reopened.workspace.inspect("manuscript.md")
    assert info["words"] >= 12500
    assert state["tasks"]["publish"]["artifact"] == info
    assert len(models.reviewed_manuscripts) == 2
    assert all(len(prompt) < settings.context_chars for prompt in models.reviewed_manuscripts)
    events = [
        json.loads(line) for line in (run.directory / "events.jsonl").read_text().splitlines()
    ]
    assembly_events = [
        event for event in events if event["kind"] == "artifact_written" and event.get("sources")
    ]
    assert len(assembly_events) == 1
    assert assembly_events[0]["sources"] == models.paths
    assert assembly_events[0]["sha256"] == info["sha256"]
    calls = models.calls.copy()
    assert Collective.open(run.directory, models=models).run()["status"] == "completed"
    assert models.calls == calls


def test_full_manuscript_review_fails_when_context_cannot_cover_actual_files(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.write("chapter.md", chapter(1))
    workspace.write("manuscript.md", chapter(1))
    models = LongManuscriptModels()
    workflow = Workflow(Settings(context_chars=4000), workspace, lambda *_a, **_k: {}, models)
    with pytest.raises(ValueError, match="Artifact context needs"):
        workflow.review_task(
            {
                "task": {
                    "id": "publish",
                    "owner": "builder",
                    "output": "manuscript.md",
                    "input_files": ["chapter.md"],
                },
                "goal": "A long manuscript",
                "attempt": 1,
            }
        )
    assert models.calls["TaskReview"] == 0


def test_accepted_chapter_changes_rebuild_it_and_the_assembled_manuscript(tmp_path):
    models = LongManuscriptModels()
    run = Collective.create(
        "Write a complete 50-page, 12500-word novel in manuscript.md",
        tmp_path,
        assembly_settings(max_tasks=12, discussion_rounds=1, context_chars=350000, num_ctx=131072),
        models=models,
    )
    assert run.run()["status"] == "completed"
    assert models.assembly_calls == 1
    run.workspace.write(models.paths[0], "An external change invalidates its prior approval.")
    reopened = Collective.open(run.directory, models=models)
    state = reopened.run()
    assert state["status"] == "completed"
    assert models.writes["chapter_01"] == 2
    assert all(models.writes[f"chapter_{index:02}"] == 1 for index in range(2, 11))
    assert models.assembly_calls == 2
    assert reopened.workspace.read("manuscript.md") == models.manuscript
    assert state["tasks"]["publish"]["artifact"] == reopened.workspace.inspect("manuscript.md")
