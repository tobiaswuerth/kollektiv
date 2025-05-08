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


class SystemMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="system", content=content)


class ToolMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="tool", content=content)
