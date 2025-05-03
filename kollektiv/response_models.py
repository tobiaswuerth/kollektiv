from pydantic import BaseModel, Field


class Phase(BaseModel):
    phase_name: str = Field(
        description="A short and descriptive title for the phase."
    )
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
