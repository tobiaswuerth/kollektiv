import ollama
import pydantic
import random
from typing import Self

from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def print(self, do=True) -> Self:
        if not do:
            return self

        role = self.role.capitalize()
        print(f" {role} ".center(50, "="))
        print(self.content)
        return self


class SystemMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="system", content=content)


class UserMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="user", content=content)


class AssistantMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="assistant", content=content)


class ToolMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="tool", content=content)


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
        user_message = UserMessage(message).print(verbose)
        message_history.append(user_message)

        model_input = message_history.copy()

        for attempt in range(format_retries):
            response = ollama.chat(
                self.model_name,
                messages=[m.__dict__ for m in model_input],
                stream=False,
                options={
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "seed": random.randint(0, 2**30 - 1),
                },
                format=format.model_json_schema() if format else None,
            )
            response = response.message.content
            ai_message = AssistantMessage(response).print(verbose)

            if format:
                try:
                    response = format.model_validate_json(response)
                except pydantic.ValidationError as e:
                    model_input.append(ai_message)
                    model_input.append(
                        SystemMessage(
                            f"Validation error: {e}\n"
                            f"Retry attempt {attempt + 1} of {format_retries}..."
                        ).print(verbose)
                    )
                    continue

            message_history.append(ai_message)
            return response, message_history
