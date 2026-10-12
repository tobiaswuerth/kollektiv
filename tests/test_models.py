from langchain_core.messages import AIMessage
from pydantic import BaseModel

from kollektiv import Settings
from kollektiv.llm import ModelOutputError, Models
from kollektiv.models import Plan


class Ready(BaseModel):
    value: str


def test_structured_output_repairs_with_feedback_and_preserves_original_request():
    class Runnable:
        def __init__(self):
            self.inputs = []

        def invoke(self, messages):
            self.inputs.append(list(messages))
            if len(self.inputs) == 1:
                return {"raw": AIMessage(content="{}"), "parsed": {}, "parsing_error": None}
            return {
                "raw": AIMessage(content='{"value":"ready"}'),
                "parsed": {"value": "ready"},
                "parsing_error": None,
            }

    class Client:
        def __init__(self):
            self.runnable = Runnable()

        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self.runnable

    models = Models(Settings(), lambda *_a, **_k: {})
    client = Client()
    models._clients["gemma4:e2b"] = client
    assert models.structured("coordinator", Ready, "ORIGINAL REQUIREMENT").value == "ready"
    assert client.runnable.inputs[1][1].content == "ORIGINAL REQUIREMENT"
    assert "failed validation" in client.runnable.inputs[1][-1].content


def test_planning_schema_constrains_task_owners_to_real_workers():
    class Client:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            raise RuntimeError("captured schema")

    import pytest

    models = Models(Settings(), lambda *_a, **_k: {})
    client = Client()
    models._clients["gemma4:e2b"] = client
    with pytest.raises(RuntimeError, match="captured"):
        models.structured("coordinator", Plan, "A full goal")
    assert client.schema["$defs"]["Task"]["properties"]["owner"]["enum"] == [
        "researcher",
        "builder",
    ]
    assert client.schema["properties"]["tasks"]["maxItems"] == 24


def test_token_limit_is_resumable_error_instead_of_partial_approval():
    import pytest

    model = Models(Settings(), lambda *_a, **_k: {})
    with pytest.raises(ModelOutputError, match="num_predict"):
        model.record(
            AIMessage(content="partial", response_metadata={"done_reason": "length"}),
            "builder",
            "tools",
        )


def test_plan_coalescing_preserves_criteria_and_rewrites_handoffs():
    from test_collective import task

    model = Models(Settings(), lambda *_a, **_k: {})
    draft = task().model_dump()
    edit = task("edit", "design.md", ["design"], "builder").model_dump()
    edit["acceptance_criteria"] = ["Make the design precise and internally consistent"]
    story = task("story", "story.md", ["edit"], "builder").model_dump()
    raw = {
        "summary": "A shared design and story",
        "success_criteria": ["All requirements"],
        "tasks": [draft, edit, story],
    }
    normalized = Plan.model_validate(model.coalesce_plan(raw))
    assert len(normalized.tasks) == 2
    assert (
        "Make the design precise and internally consistent"
        in normalized.tasks[0].acceptance_criteria
    )
    assert normalized.tasks[1].dependencies == ["design"]
    assert normalized.success_criteria == ["All requirements"]
