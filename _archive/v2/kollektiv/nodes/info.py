from langchain_core.messages import HumanMessage


class InfoNode:
    def __init__(self, msg: str):
        self.msg: str = msg

    def __call__(self, state) -> dict:
        return {"messages": [HumanMessage(self.msg)]}
