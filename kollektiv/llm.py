import ollama
import pydantic
import random
from typing import Self
from langchain_core.output_parsers import PydanticOutputParser

from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def _print_title(self) -> str:
        return print(f" {self.role.capitalize()} ".center(50, "="))

    def print(self, do=True) -> Self:
        if not do:
            return self

        self._print_title()
        print(self.content)
        return self


class UserMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="user", content=content)


class AssistantMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="assistant", content=content)


class LLMClient:
    def __init__(self, model_name: str = "phi4:latest") -> None:
        self.model_name = model_name

    def chat(
        self,
        message: str,
        message_history: list[Message] = [],
        format: pydantic.BaseModel = None,
        format_retries: int = 3,
        verbose: bool = True,
    ) -> tuple[Message, list[Message]]:

        if format:
            assert issubclass(format, pydantic.BaseModel)
            parser = PydanticOutputParser(pydantic_object=format)
            message_history.append(
                UserMessage(parser.get_format_instructions()).print(verbose)
            )

        user_message = UserMessage(message).print(verbose)
        message_history.append(user_message)
        model_input = message_history.copy()

        for attempt in range(format_retries):
            response = ollama.chat(
                self.model_name,
                messages=[m.__dict__ for m in model_input],
                stream=verbose,
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "seed": random.randint(0, 2**30 - 1),
                },
            )

            if verbose:
                AssistantMessage("")._print_title()
                chunks = []
                for chunk in response:
                    chunk = chunk.message.content
                    print(chunk, end="", flush=True)
                    chunks.append(chunk)
                print()
                response = "".join(chunks)
                ai_message = AssistantMessage(response)
            else:
                response = response.message.content
                ai_message = AssistantMessage(response).print(verbose)

            if format:
                try:
                    if response.startswith("<think>"):
                        response = response.split("</think>")[-1].strip()
                        if response.startswith("```json") and response.endswith("```"):
                            response = response[7:-3].strip()
                    response = format.model_validate_json(response)
                except pydantic.ValidationError as e:
                    model_input.append(ai_message)
                    model_input.append(
                        UserMessage(
                            f"Validation error: {e}\n"
                            f"Retry attempt {attempt + 1} of {format_retries}..."
                        ).print(verbose)
                    )
                    continue

            message_history.append(ai_message)
            return response, message_history
