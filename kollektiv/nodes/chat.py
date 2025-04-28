from langchain_core.language_models import BaseChatModel


class ChatNode:
    def __init__(self, llm: BaseChatModel):
        self.llm: BaseChatModel = llm

    def __call__(self, state) -> dict:
        message = self.llm.invoke(state["messages"])
        assert len(message.tool_calls) <= 1
        return {"messages": [message]}
