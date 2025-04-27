import pydantic
from typing import Optional, Dict
import json
from json import JSONEncoder, JSONDecoder
import os

from .tool import (
    Tool,
    Function,
    ResponseStatus,
    RESPONSE_OK,
    RESPONSE_NOT_FOUND,
)


class Task(pydantic.BaseModel):
    unique_name: str
    description: str
    date: str
    status: str
    priority: str

    def __str__(self):
        return f"Task({self.unique_name})"


class TaskEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Task):
            return {
                "unique_name": obj.unique_name,
                "description": obj.description,
                "date": obj.date,
                "status": obj.status,
                "priority": obj.priority,
            }
        return super().default(obj)


class TaskDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Dict) -> Task:
        return Task(**obj)


class GetTasksInput(pydantic.BaseModel):
    pass

    def __str__(self):
        return f"GetTasksInput"


class GetTasksOutput(pydantic.BaseModel):
    status: ResponseStatus
    tasks: Optional[list["Task"]]


class UpdateTaskInput(pydantic.BaseModel):
    task: Task

    def __str__(self):
        return f"UpdateTaskInput(task={self.task})"


class UpdateTaskOutput(pydantic.BaseModel):
    status: ResponseStatus


class DeleteTaskInput(pydantic.BaseModel):
    unique_name: str

    def __str__(self):
        return f"DeleteTaskInput(unique_name={self.unique_name})"


class DeleteTaskOutput(pydantic.BaseModel):
    status: ResponseStatus


class Planner(Tool):

    def __init__(self, output_dir: str):
        name = "Planner"
        description = "A tool to manage personal tasks and plans."
        super().__init__(name, description)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.register_function(
            Function(
                name="get_tasks",
                description="Get a list of all pending tasks.",
                func=self.get_tasks,
            )
        )
        self.register_function(
            Function(
                name="update_task",
                description="Create/Update a task.",
                func=self.update_task,
            )
        )
        self.register_function(
            Function(
                name="delete_task",
                description="Delete a task.",
                func=self.delete_task,
            )
        )

    def get_file_path(self, agent_id: int) -> str:
        return os.path.join(self.output_dir, f"tasks_{agent_id}.json")

    def load_state(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f, cls=TaskDecoder)

    def save_state(self, path: str, state: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4, cls=TaskEncoder)

    def get_tasks(self, agent, input_: GetTasksInput) -> GetTasksOutput:
        file_path = self.get_file_path(agent.name)
        if not os.path.exists(file_path):
            return GetTasksOutput(status=RESPONSE_OK, tasks=[])

        state = self.load_state(file_path)
        return GetTasksOutput(status=RESPONSE_OK, tasks=state)

    def update_task(self, agent, input_: UpdateTaskInput) -> UpdateTaskOutput:
        file_path = self.get_file_path(agent.name)
        state = [] if not os.path.exists(file_path) else self.load_state(file_path)

        found_existing = [t for t in state if t.unique_name == input_.task.unique_name]
        if found_existing:
            state.remove(found_existing[0])

        state.append(input_.task)
        self.save_state(file_path, state)
        return UpdateTaskOutput(status=RESPONSE_OK)

    def delete_task(self, agent, input_: DeleteTaskInput) -> DeleteTaskOutput:
        file_path = self.get_file_path(agent.name)
        if not os.path.exists(file_path):
            return DeleteTaskOutput(status=RESPONSE_NOT_FOUND)

        state = self.load_state(file_path)
        found_existing = [t for t in state if t.unique_name == input_.unique_name]
        if not found_existing:
            return DeleteTaskOutput(status=RESPONSE_NOT_FOUND)

        state.remove(found_existing[0])
        self.save_state(file_path, state)
        return DeleteTaskOutput(status=RESPONSE_OK)
