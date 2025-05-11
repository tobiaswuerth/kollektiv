from abc import ABC, abstractmethod
from typing import Any

from kollektiv.core import ToolMessage


class Handler(ABC):
    def __init__(self, retry_attempts: int = 3):
        self.retry_attempts = retry_attempts
        self.attempt = 0
        self.instructions = self._prepare_instructions()

    @abstractmethod
    def _prepare_instructions(self) -> str:
        """Prepare instructions for the handler."""
        pass

    @abstractmethod
    def consider(self, response: str) -> bool:
        """Check if the response is valid for this handler."""
        pass

    @abstractmethod
    def _invoke(self, response: str) -> Any:
        """Perform a single attempt to resolve the response."""
        pass

    def invoke(self, response: str) -> tuple[bool, Any]:
        """Handle retry logic and delegate resolution to `_resolve_once`."""
        try:
            if not self.consider(response):
                raise RuntimeError(f"Not considered response valid for handler {self.__class__.__name__}")
            result = self._invoke(response)
            return True, result
        except Exception as e:
            self.attempt += 1
            if self.attempt >= self.retry_attempts:
                raise e
            return False, ToolMessage(
                (
                    f"!! [ERROR]: {e}\n"
                    f"!! If you see this message, it means that your output did not adhere to the requested format.\n"
                    f"!! Retrying attempt {self.attempt + 1} of {self.retry_attempts}."
                )
            )
