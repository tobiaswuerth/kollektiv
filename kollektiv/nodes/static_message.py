from langchain_core.messages import HumanMessage


class MessageNode:
    def __init__(self, msg: str):
        self.msg: str = msg

    def __call__(self, state) -> str:
        return {"messages": [HumanMessage(self.msg)]}
