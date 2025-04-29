from .llm import LLMClient, Message
from .node import Node

import pydantic

class MyTypedDict(pydantic.BaseModel):
    thoughts: list[str]

class System:

    def __init__(self, goal: str):
        self.goal = goal

    def run(self):
        # Initialize the system with the goal
        print(f"System initialized with goal:\n{self.goal}")

        llm = LLMClient()

        history = []
        while True:
            # Get user input
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the system.")
                break

            _, history = llm.chat(user_input, message_history=history, format=MyTypedDict)
