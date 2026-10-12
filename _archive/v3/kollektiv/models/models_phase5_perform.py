from pydantic import BaseModel, Field


class ResultEvaluation(BaseModel):
    continue_with_next_task: bool = Field(
        description=(
            "Indicates whether the assistant should proceed to the next task. "
            "Set to `true` if the current task is successfully completed. "
            "Set to `false` if the task needs further work or improvement. "
            "The assistant should evaluate its work against the task completion criteria before deciding: "
            "- Choose `true` when confident that all requirements have been met. "
            "- Choose `false` to revise and improve the current solution. "
        )
    )
