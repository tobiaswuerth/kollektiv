from pydantic import BaseModel
from typing import Any

from .messages import SystemMessage


class FormatHandler:

    def __init__(self, format: BaseModel, retry_attempts: int = 3):
        assert issubclass(format, BaseModel), "Format must be a subclass of BaseModel."
        self.format = format
        self.instructions = self._prepare_instructions(format)
        self.retry_attempts = retry_attempts
        self.attempt = 0

    def _prepare_instructions(self, format: BaseModel) -> str:
        return (
            "Your normal response MUST be formatted as JSON that conforms to this schema:\n\n"
            f"```json\n{format.model_json_schema()}\n```\n\n"
            "For example:\n"
            "For the schema:"
            "`{{'properties': {{'foo': {{'title': 'Foo', 'description': 'a list of strings', "
            "'type': 'array', 'items': {{'type': 'string'}}}}}}, 'required': ['foo']}}`"
            "the response\n"
            "```json\n{{'foo': ['bar', 'baz']}}\n```\n"
            "is a well-formatted instance of the schema."
        )

    def resolve(self, response: str) -> tuple[bool, SystemMessage | Any]:
        try:
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            response = self.format.model_validate_json(response, strict=True)
            return True, response
        except Exception as e:
            self.attempt += 1
            if self.attempt >= self.retry_attempts:
                raise e
            return False, SystemMessage(
                (
                    f"!! [ERROR]: {e}\n"
                    f"!! If you see this message, it means that your output did not adhere to the requested format.\n"
                    f"!! Retrying attempt {self.attempt + 1} of {self.retry_attempts}."
                )
            )
