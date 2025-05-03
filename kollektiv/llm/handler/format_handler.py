from .handle import Handler
from pydantic import BaseModel
from typing import Any


class FormatHandler(Handler):

    def __init__(self, format: BaseModel, retry_attempts: int = 3):
        assert issubclass(format, BaseModel), "Format must be a subclass of BaseModel."
        self.format = format
        super().__init__(retry_attempts)

    def _prepare_instructions(self) -> str:
        return (
            "Your normal response MUST be formatted as JSON that conforms to this schema:\n\n"
            f"```json\n{self.format.model_json_schema()}\n```\n\n"
            "For example:\n"
            "For the schema:"
            "`{{'properties': {{'foo': {{'title': 'Foo', 'description': 'a list of strings', "
            "'type': 'array', 'items': {{'type': 'string'}}}}}}, 'required': ['foo']}}`"
            "the response\n"
            "```json\n{{'foo': ['bar', 'baz']}}\n```\n"
            "is a well-formatted instance of the schema."
        )

    def _resolve(self, response: str) -> Any:
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return self.format.model_validate_json(response, strict=True)
