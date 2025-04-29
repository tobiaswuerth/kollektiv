from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field


class DecisionResponse(BaseModel):
    message: str = Field(description="Your thinking process before deciding")
    decision: str = Field(description="Your decision keyword")


class DecisionNode:
    def __init__(
        self,
        llm: BaseChatModel,
        decision_routes: dict[str, str],
        retry: int = 3,
    ):
        self.llm: BaseChatModel = llm
        self.route_map: dict[str, str] = decision_routes
        self.retry: int = retry

    @property
    def f_cond_paths(self):
        def conditional_path_func(state):
            assert 'path' in state, "State must contain 'path' key"
            return state["path"]

        destinations = tuple(self.route_map.values())
        conditional_path_func.__annotations__["return"] = Literal[destinations]
        return conditional_path_func

    @property
    def invalid_option(self):
        return SystemMessage(
            (
                f"!! SYSTEM ERROR - INVALID RESPONSE FORMAT !!\n"
                f"Valid options are: [ {', '.join(self.route_map)} ] - Try again."
            )
        )

    def __call__(self, state) -> dict:
        new_msgs = []
        parser = PydanticOutputParser(pydantic_object=DecisionResponse)

        for _ in range(self.retry):
            new_msgs.append(
                SystemMessage(
                    content=(
                        "You must decide now how you want to proceed. "
                        f"Your decision MUST be one of these options:\n[ {', '.join(self.route_map)} ]\n"
                        "Respond with a structured JSON output according to this format:\n"
                        f"{parser.get_format_instructions()}"
                    )
                )
            )

            response = self.llm.invoke(state["messages"] + new_msgs)
            new_msgs.append(response)

            try:
                parsed_response = parser.parse(response.content)
                if parsed_response.decision not in self.route_map:
                    new_msgs.append(self.invalid_option)
                    continue

                return {
                    "messages": new_msgs,
                    "path": self.route_map[parsed_response.decision],
                }
            except Exception:
                new_msgs.append(self.invalid_option)
                continue

        raise ValueError("DecisionNode: Max retries exceeded.")
