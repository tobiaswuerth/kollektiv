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
        self.decision_routes: dict[str, str] = decision_routes
        self.retry: int = retry

    @property
    def f_cond_paths(self):
        def conditional_path_func(state):
            return state["messages"][-1].content

        paths = tuple(self.decision_routes.values())
        conditional_path_func.__annotations__['return'] = Literal[paths]   
        return conditional_path_func

    @property
    def invalid_option(self):
        return SystemMessage((
            f"!! SYSTEM ERROR - INVALID RESPONSE FORMAT !!\n"
            f"Valid options are: [ {', '.join(self.decision_routes)} ] - Try again."
        ))

    def __call__(self, state) -> str:
        new_messages = []
        parser = PydanticOutputParser(pydantic_object=DecisionResponse)

        for _ in range(self.retry):
            new_messages.append(SystemMessage(content=(
                "You must decide now how you want to proceed. "
                f"Your decision MUST be one of these options:\n[ {', '.join(self.decision_routes)} ]\n"
                "Respond with a structured JSON output according to this format:\n"
                f"{parser.get_format_instructions()}"
            )))
            
            response = self.llm.invoke(state["messages"] + new_messages)
            new_messages.append(response)
            
            try:
                parsed_response = parser.parse(response.content)
                if parsed_response.decision not in self.decision_routes:
                    new_messages.append(self.invalid_option)
                    continue
                
                new_messages.append(SystemMessage(self.decision_routes[parsed_response.decision]))
                return {
                    "messages": new_messages,
                }
            except Exception:
                new_messages.append(self.invalid_option)
                continue

        raise ValueError("DecisionNode: Max retries exceeded.")
    
    