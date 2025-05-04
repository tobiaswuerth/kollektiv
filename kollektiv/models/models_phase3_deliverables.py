from pydantic import BaseModel, Field, field_validator

from kollektiv.models.models_phase2_phases import ProjectPhase


class DeliverableFile(BaseModel):
    file_name: str = Field(
        description=(
            "The name of the file or document that is expected to be produced as part of this phase. "
            "Valid file extensions are [.txt, .md, .json]."
        )
    )
    description: str = Field(
        description=(
            "A short and descriptive explanation of the file's content and scope. "
            "This should provide enough context to understand the purpose of the file and what to find in it."
        )
    )

    @field_validator("file_name")
    def validate_file_name(cls, value: str) -> str:
        valid_extensions = {".txt", ".md", ".json"}
        if not any(value.endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"File name must end with one of the valid extensions: {valid_extensions}."
            )
        return value


class ProjectPhaseWithDeliverables(ProjectPhase):
    required_inputs: list[str] = Field(
        description="A list of files or documents previously produced that are required for this phase."
    )
    deliverable_files: list[DeliverableFile] = Field(
        description="A list of files or documents that are expected to be produced as part of this phase."
    )


class ProjectWithDeliverables(BaseModel):
    overarching_goal: str = Field(
        description="The primary objective of the project plan."
    )
    description: str = Field(
        description="A detailed explanation providing context for the project plan."
    )
    project_phases: list[ProjectPhaseWithDeliverables] = Field(
        description="A list of phases, each representing a step toward achieving the overarching goal."
    )
