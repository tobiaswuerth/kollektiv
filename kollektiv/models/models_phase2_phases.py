from pydantic import BaseModel, Field


class ProjectPhase(BaseModel):
    phase_name: str = Field(description="A short and descriptive title for the phase.")
    description: str = Field(
        description="A detailed explanation providing context for the phase."
    )


class Project(BaseModel):
    overarching_goal: str = Field(
        description="The main goal of the problem deconstruction tree."
    )
    description: str = Field(
        description="An explanation providing context for the problem deconstruction tree."
    )
    project_phases: list[ProjectPhase] = Field(
        description="A list of phases, each outlining a step toward achieving the main goal."
    )
