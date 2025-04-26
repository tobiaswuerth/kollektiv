import pydantic

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
    files: list[str]


class ReadFileInput(pydantic.BaseModel):
    file_name: str


class ReadFileOutput(pydantic.BaseModel):
    status: ResponseStatus
    content: str


class WriteFileInput(pydantic.BaseModel):
    file_name: str
    content: str


class WriteFileOutput(pydantic.BaseModel):
    status: ResponseStatus


class Storage(Tool):

    def __init__(self):
        name = "Storage"
        description = "A tool for storing and retrieving files."
        super().__init__(name, description)

        self.state = {}

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

    def get_files(self, agent_id: int, input_: GetFilesInput) -> GetFilesOutput:
        files = self.state.get(agent_id, [])
        return GetFilesOutput(files=files)

    def read_file(self, agent_id: int, input_: ReadFileInput) -> ReadFileOutput:
        files = self.state.get(agent_id, [])
        if input_.file_name not in files:
            return ReadFileOutput(status=RESPONSE_NOT_FOUND, content=None)
        return ReadFileOutput(
            status=RESPONSE_OK, content=self.state[agent_id][input_.file_name]
        )

    def write_file(self, agent_id: int, input_: WriteFileInput) -> WriteFileOutput:
        if agent_id not in self.state:
            self.state[agent_id] = {}
        self.state[agent_id][input_.file_name] = input_.content
        return WriteFileOutput(status=RESPONSE_OK)
