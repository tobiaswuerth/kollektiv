from .tools import Tool


class Agent:
    def __init__(self, name, role: list[str], tools: list[Tool]):
        self.id = id(self)
        self.name = name
        self.role = role
        self.tools = tools

        

    def __str__(self):
        return f"Agent({self.name})"
    def __repr__(self):
        return str(self)

    def update(self, current_system_state: dict):
        print(f"Agent {self.name} received system state: {current_system_state}")
