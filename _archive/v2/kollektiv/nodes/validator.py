from typing import Literal, Generic, TypeVar, get_args
from langchain_core.messages import SystemMessage
from langgraph.types import Command
from langchain_core.tools import tool
from pydantic import BaseModel


TBaseModel = TypeVar("T", bound=BaseModel)


class TypedValidatorNode(Generic[TBaseModel]):
    def __init__(
        self,
        success_path: str,
        failure_path: str,
        retry: int = 3,
    ):
        self.success_path: str = success_path
        self.failure_path: str = failure_path
        self.retry: int = retry
        self.attempts: int = 0

    @property
    def f_cond_paths(self):
        def conditional_path_func(state):
            assert 'path' in state, "State must contain 'path' key"
            return state["path"]

        destinations = tuple([self.success_path, self.failure_path])
        conditional_path_func.__annotations__["return"] = Literal[destinations]
        return conditional_path_func

    def __call__(self, state) -> dict:
        messages = state.get("messages", [])
        assert len(messages) > 0, "No messages in state"

        response = messages[-1].content
        try:
            self.model = get_args(self.__orig_class__)[0]
            response = self.model.model_validate_json(response)
            return {
                "messages": [SystemMessage(response)],
                "path": self.success_path,
            }

        except Exception as e:
            self.attempts += 1
            if self.attempts >= self.retry:
                raise RuntimeError(
                    f"Max validation attempts reached: {self.attempts}"
                ) from e
            return {
                "messages": [
                    SystemMessage(
                        (
                            f"!! SYSTEM ERROR - INVALID RESPONSE FORMAT !!\n"
                            f"Exception: {str(e)}"
                        )
                    )
                ],
                "path": self.failure_path,
            }
