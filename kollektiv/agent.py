import pydantic
import json

from .tools import Tool, Function
from .llm import LLMClient
from .role import Role

class AgentAction(pydantic.BaseModel):
    target: str
    input: dict

class AgentResponse(pydantic.BaseModel):
    actions: list[AgentAction]


class Agent:
    def __init__(self, llm:LLMClient, name:str, role: Role):
        self.id = id(self)
        self.llm:LLMClient = llm
        self.name:str = name
        self.role:Role = role

        self.tools:list[Tool] = []

        self.inbox:list[str] = []
        self.inbox_archive: list[str] = []

        self.tool_invoke_history: list[tuple[str, str]] = []


    def __str__(self):
        return f"Agent(name={self.name}, id={self.id}, role={self.role.name})"

    def __repr__(self):
        return str(self)

    def update(self, current_system_state: dict):
        prompt = f"""
You are {self.name}, a {self.role} agent.
Your team is {current_system_state["your_team"]}.
Your team goal is: {current_system_state["team_goal"]}.
The system time is: {current_system_state["system_time"]}.

In the past, you have received the following messages:
{'<none>' if not self.inbox_archive else self.inbox_archive}

You have performed the following actions:
{'<none>' if not self.tool_invoke_history else self.tool_invoke_history}

You currently have the following new messages:
{'<none>' if not self.inbox else self.inbox}

What would you like to do?

You can utilize the following tools:
{'<none>' if not self.tools else [tool.get_json_schema() for tool in self.tools]}

You must select and plan out one or more tool usages and respond in JSON format by providing this information:
"target": "<tool_name>.<function_name>"
"input": <function_input>

You will receive your responses and potentially new messages in the next tick.
Make sure you keep up with your tasks not just chat around.
"""
        self.inbox_archive.extend(self.inbox)
        self.inbox = []

        # print(f"Agent {self.name} state:\n{prompt}")
        response = self.llm.generate(prompt, format=AgentResponse)
        print(f"Agent {self.name} response:\n{response}")
        
        for action in response.actions:
            target = action.target
            input_ = action.input
            tool_name, function_name = target.split(".")
            tool = [t for t in self.tools if t.name == tool_name][0]
            function:Function = tool.functions[function_name]
            result = function.func(self.id, function.func_TIn(**input_))
            
            self.tool_invoke_history.append({
                "target": target,
                "input": input_,
                "output": result,
                "system_time": current_system_state["system_time"],
            })

        print("-" * 80)
