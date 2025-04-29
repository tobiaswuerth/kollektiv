import ollama
import pydantic
import random

from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def pretty_print(self) -> None:
        role = self.role.capitalize()
        print(f" {role} ".center(50, "="))
        print(self.content)


class LLMClient:
    def __init__(self, model_name: str = "phi4:latest") -> None:
        self.model_name = model_name

    def chat(
        self,
        message: str,
        message_history: list[Message] = [],
        format: pydantic.BaseModel = None,
        format_retries: int = 3,
        print_result: bool = True,
    ) -> tuple[Message, list[Message]]:
        user_message = Message(role="user", content=message)
        message_history.append(user_message)
        if print_result:
            user_message.pretty_print()

        model_input = message_history.copy()

        for attempt in range(format_retries):
            response = ollama.chat(
                self.model_name,
                messages=[m.__dict__ for m in model_input],
                stream=False,
                options={
                    "temperature": 0.5,
                    "top_p": 0.9,
                    # "num_ctx": 10000,
                    "seed": random.randint(0, 2**30 - 1),
                },
                format=format.model_json_schema() if format else None,
            )
            response = response.message.content
            ai_message = Message(role="assistant", content=response)

            if format:
                try:
                    response = format.model_validate_json(response)
                except pydantic.ValidationError as e:
                    model_input.append(ai_message)
                    model_input.append(
                        Message(
                            role="system",
                            content=(
                                f"Validation error: {e}\n"
                                f"Retry attempt {attempt + 1} of {format_retries}..."
                            ),
                        )
                    )
                    if print_result:
                        model_input[-2].pretty_print()
                        model_input[-1].pretty_print()
                    continue

            message_history.append(ai_message)
            if print_result:
                ai_message.pretty_print()
            return response, message_history
