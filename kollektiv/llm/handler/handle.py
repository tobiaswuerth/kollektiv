from abc import ABC, abstractmethod
from typing import Any

from ..messages import ToolMessage


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
    def _resolve(self, response: str) -> Any:
        """Perform a single attempt to resolve the response."""
        pass

    def resolve(self, response: str) -> tuple[bool, Any]:
        """Handle retry logic and delegate resolution to `_resolve_once`."""
        try:
            result = self._resolve(response)
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
