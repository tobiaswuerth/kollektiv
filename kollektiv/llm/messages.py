from typing import Self
from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def _print_title(self) -> str:
        return print(f" {self.role.capitalize()} ".center(80, "="))

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


class SystemMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="system", content=content)


class ToolMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="tool", content=content)
