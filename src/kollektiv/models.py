"""Small model-facing schemas and JSON-only graph state."""

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


def merge_records(left: dict | None, right: dict | None) -> dict:
    """Keyed reducers make parallel writes and checkpoint replay idempotent."""
    return {**(left or {}), **(right or {})}


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MessageDraft(Schema):
    recipient: str = Field(description="An exact teammate name or 'all' for a broadcast")
    content: str = Field(min_length=1, max_length=6000)


class Contribution(Schema):
    findings: str = Field(
        description="Concrete recommendations, facts, uncertainties and conventions"
    )
    messages: list[MessageDraft] = Field(default_factory=list, max_length=5)


class Task(Schema):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,60}$")
    title: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    description: str = Field(min_length=1)
    owner: str = Field(description="Exact name of an agent with write_file access")
    output: str = Field(description="One relative deliverable path, e.g. chapters/chapter_1.md")
    dependencies: list[str] = Field(default_factory=list, description="IDs of prerequisite tasks")
    acceptance_criteria: list[str] = Field(min_length=1)

    @field_validator("output")
    @classmethod
    def valid_output(cls, value: str) -> str:
        from .workspace import relative_path

        return relative_path(value)


class Plan(Schema):
    summary: str
    success_criteria: list[str] = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_dependencies(self) -> "Plan":
        ids = [t.id for t in self.tasks]
        paths = [t.output.casefold() for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("Task IDs must be unique")
        if len(set(paths)) != len(paths):
            raise ValueError("Each file must have exactly one owner; task outputs must be unique")
        if any(a.startswith(b + "/") for a in paths for b in paths if a != b):
            raise ValueError("A deliverable cannot be the parent directory of another deliverable")
        visited: set[str] = set()
        while len(visited) < len(ids):
            ready = [
                t.id for t in self.tasks if t.id not in visited and set(t.dependencies) <= visited
            ]
            if not ready:
                raise ValueError(
                    "Dependencies contain an unknown task, a self-reference, or a cycle"
                )
            visited.update(ready)
        return self


class TaskReview(Schema):
    feedback: str = Field(description="Evidence of criteria met, or specific corrections to make")
    reflection: str = Field(description="Reusable lessons and decisions to share with the team")
    decision: Literal["accept", "revise"] = Field(
        description="After auditing the evidence: accept if ALL task criteria are met and no corrections are needed; otherwise revise"
    )

    @computed_field
    @property
    def accepted(self) -> bool:
        return self.decision == "accept"


class GoalReview(Schema):
    feedback: str = Field(description="Assess every goal requirement using the actual artifacts")
    revise_tasks: list[str] = Field(
        default_factory=list, description="Existing task IDs needing changes"
    )
    additional_tasks: list[Task] = Field(
        default_factory=list, description="Missing deliverables, if any"
    )
    decision: Literal["complete", "revise"] = Field(
        description="After auditing all artifacts: complete if ALL goal requirements are met; otherwise revise"
    )

    @computed_field
    @property
    def achieved(self) -> bool:
        return self.decision == "complete"


class GraphState(TypedDict, total=False):
    goal: str
    run_id: str
    status: str
    discussion_round: int
    contributions: Annotated[dict[str, dict], merge_records]
    messages: Annotated[dict[str, dict], merge_records]
    notes: Annotated[dict[str, dict], merge_records]
    events: Annotated[dict[str, dict], merge_records]
    plan: dict
    tasks: dict[str, dict]
    ready: list[str]
    results: Annotated[dict[str, dict], merge_records]
    goal_revision: int
    goal_review: dict
    final_summary: str


class DiscussionState(TypedDict):
    goal: str
    run_id: str
    agent_name: str
    discussion_round: int
    contributions: dict[str, dict]
    messages: dict[str, dict]
    notes: dict[str, dict]


class WorkerOutput(TypedDict, total=False):
    results: Annotated[dict[str, dict], merge_records]
    messages: Annotated[dict[str, dict], merge_records]
    notes: Annotated[dict[str, dict], merge_records]
    events: Annotated[dict[str, dict], merge_records]


class WorkerState(WorkerOutput, total=False):
    goal: str
    run_id: str
    task: dict
    plan_summary: str
    attempt: int
    steps: int
    history: Annotated[list, operator.add]
    reply: Any
    finished: bool
    feedback: str
    review: dict
