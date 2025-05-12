import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from kollektiv.core import Message
from kollektiv.task.result import Result


class Step(ABC):
    logger = logging.getLogger(__name__)

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def instructions(self) -> str:
        return ""

    def on_validate_request(self, request: Message) -> Result:
        return Result(True, None)

    @abstractmethod
    def execute(self, request: Message, history: List[Message]) -> Message:
        pass

    def on_after(self, response: Message) -> None:
        pass


@dataclass
class Link:
    i: int
    step: Step
    prev: Optional["Link"] = None
    next: Optional["Link"] = None

    @staticmethod
    def from_steps(steps: List[Step]) -> List["Link"]:
        links = []
        for i, step in enumerate(steps):
            current = Link(i, step)
            if i > 0:
                current.prev = links[i - 1]
                links[i - 1].next = current
            links.append(current)
        return links
