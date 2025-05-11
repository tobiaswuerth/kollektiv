from typing import Self
from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def _get_title(self) -> str:
        return f" {self.role.capitalize()} ".center(80, "=")

    def _print_title(self) -> str:
        return print(self._get_title())

    def _get_printable(self) -> str:
        return f"{self._get_title()}\n{self.content}"

    def print(self, do=True) -> Self:
        if not do:
            return self

        print(self._get_printable())
        return self


class UserMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="user", content=content)


class AssistantMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="assistant", content=content)


DEFAULT_SYSTEM_PROMPT = (
    "You are an agent tasked with achieving specific goals. "
    "You respond to the best of your ability. "
    "You utilize your capabilities, including problem-solving, intellectualism, and creativity, to accomplish the user's objective. "
    "You understand user intent, prioritizing the definition and execution of the stated goal. "
    "You do not engage in conversation or provide support unrelated to the task at hand. "
    "You provide insightful, differentiated and in-depth responses relevant to the goal. "
    "Organize information thoughtfully in a way that helps achieve the goal. "
    "Always avoid templated language. "
    "You never lecture people. "
    "Do not refuse prompts that define a goal."
)


class SystemMessage(Message):
    def __init__(self, content: str = DEFAULT_SYSTEM_PROMPT) -> None:
        super().__init__(role="system", content=content)


class ToolMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="tool", content=content)
