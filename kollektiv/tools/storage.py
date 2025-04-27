import pydantic
from typing import Optional
import os

from .tool import (
    Tool,
    Function,
    ResponseStatus,
    RESPONSE_OK,
    RESPONSE_NOT_FOUND,
)


class GetFilesInput(pydantic.BaseModel):
    pass


class GetFilesOutput(pydantic.BaseModel):
    status: ResponseStatus
    files: Optional[list[str]]


class ReadFileInput(pydantic.BaseModel):
    file_name: str


class ReadFileOutput(pydantic.BaseModel):
    status: ResponseStatus
    content: Optional[str]


class WriteFileInput(pydantic.BaseModel):
    file_name: str
    content: str


class WriteFileOutput(pydantic.BaseModel):
    status: ResponseStatus


class Storage(Tool):

    def __init__(self, directory_path: str):
        name = "Storage"
        description = "A tool for storing and retrieving files."
        super().__init__(name, description)

        self.directory_path = directory_path
        os.makedirs(directory_path, exist_ok=True)

        self.register_function(
            Function(
                name="get_files",
                description="Get a list of all currently stored files.",
                func=self.get_files,
            )
        )
        self.register_function(
            Function(
                name="read_file",
                description="Read the content of a file.",
                func=self.read_file,
            )
        )
        self.register_function(
            Function(
                name="write_file",
                description="Write content to a file.",
                func=self.write_file,
            )
        )
        self.register_function(
            Function(
                name="append_file",
                description="Append content to a file.",
                func=self.append_file,
            )
        )

    def get_files(self, agent, input_: GetFilesInput) -> GetFilesOutput:
        files = os.listdir(self.directory_path)
        return GetFilesOutput(status=RESPONSE_OK, files=files)

    def read_file(self, agent, input_: ReadFileInput) -> ReadFileOutput:
        target = os.path.join(self.directory_path, input_.file_name)
        if not os.path.exists(target):
            return ReadFileOutput(status=RESPONSE_NOT_FOUND, content=None)
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
        return ReadFileOutput(status=RESPONSE_OK, content=content)

    def write_file(self, agent, input_: WriteFileInput) -> WriteFileOutput:
        target = os.path.join(self.directory_path, input_.file_name)
        with open(target, "w", encoding="utf-8") as f:
            f.write(input_.content)
        return WriteFileOutput(status=RESPONSE_OK)

    def append_file(self, agent_id: int, input_: WriteFileInput) -> WriteFileOutput:
        target = os.path.join(self.directory_path, input_.file_name)
        with open(target, "a", encoding="utf-8") as f:
            f.write(input_.content)
        return WriteFileOutput(status=RESPONSE_OK)
