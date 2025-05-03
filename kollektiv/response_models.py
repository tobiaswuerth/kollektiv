from pydantic import BaseModel, Field


class Phase(BaseModel):
    phase_name: str = Field(description="A short and descriptive title for the phase.")
    description: str = Field(
        description="A detailed explanation providing context for the phase."
    )


class ProblemDeconstructionTree(BaseModel):
    overarching_goal: str = Field(
        description="The primary objective of the problem deconstruction tree."
    )
    description: str = Field(
        description="A detailed explanation providing context for the problem deconstruction tree."
    )

    project_phases: list[Phase] = Field(
        description="A list of phases, each representing a step toward achieving the overarching goal."
    )


class ProjectPhase(Phase):
    required_inputs: list[str] = Field(
        description="A list of files or documents previously produced that are required for this phase."
    )
    deliverable_files: list[str] = Field(
        description="A list of files or documents that are expected to be produced as part of this phase."
    )


class ProjectPlan(BaseModel):
    overarching_goal: str = Field(
        description="The primary objective of the project plan."
    )
    description: str = Field(
        description="A detailed explanation providing context for the project plan."
    )

    project_phases: list[ProjectPhase] = Field(
        description="A list of phases, each representing a step toward achieving the overarching goal."
    )

    def from_tree(self, tree: ProblemDeconstructionTree) -> "ProjectPlan":
        return ProjectPlan(
            overarching_goal=tree.overarching_goal,
            description=tree.description,
            project_phases=[
                ProjectPhase(
                    id=id(phase),
                    phase_name=phase.phase_name,
                    description=phase.description,
                    deliverable_files=[],
                    required_inputs=[],
                )
                for phase in tree.project_phases
            ],
        )
