from pydantic import BaseModel, Field, field_validator

from kollektiv.data_models.models_phase3_deliverables import (
    DeliverableFile,
    ProjectPhaseWithDeliverables,
    ProjectWithDeliverables,
)


class Task(BaseModel):
    task_name: str = Field(description="A short and descriptive title for the task.")
    description: str = Field(
        description="A detailed explanation providing context for the task."
    )
    required_inputs: list[str] = Field(
        description="A list of files or documents previously produced that are required for this phase."
    )
    deliverable_file: DeliverableFile = Field(
        description="A file or document that is expected to be produced as part of this task."
    )


class ProjectPhaseWithTasks(ProjectPhaseWithDeliverables):
    tasks: list[Task] = Field(
        description="A list of tasks, each representing a step within the project phase."
    )

    @staticmethod
    def from_phase(phase: ProjectPhaseWithDeliverables) -> "ProjectPhaseWithTasks":
        return ProjectPhaseWithTasks(
            phase_name=phase.phase_name,
            description=phase.description,
            deliverable_files=phase.deliverable_files,
            required_inputs=phase.required_inputs,
            tasks=[],
        )


class TaskList(BaseModel):
    tasks: list[Task] = Field(
        description="A list of tasks, each representing a step within the project phase."
    )


class ProjectWithTasks(BaseModel):
    overarching_goal: str = Field(
        description="The primary objective of the project plan."
    )
    description: str = Field(
        description="A detailed explanation providing context for the project plan."
    )
    project_phases: list[ProjectPhaseWithTasks] = Field(
        description="A list of phases, each representing a step toward achieving the overarching goal."
    )

    @staticmethod
    def from_plan(plan: ProjectWithDeliverables) -> "ProjectWithTasks":
        return ProjectWithTasks(
            overarching_goal=plan.overarching_goal,
            description=plan.description,
            project_phases=[
                ProjectPhaseWithTasks.from_phase(p) for p in plan.project_phases
            ],
        )
