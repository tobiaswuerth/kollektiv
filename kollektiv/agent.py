from .tools import Tool


class Agent:
    def __init__(self, name, role: list[str]):
        self.id = id(self)
        self.name = name
        self.role = role

        self.tools:list[Tool] = []
        self.inbox:list[str] = []
        

    def __str__(self):
        id = self.id
        name = self.name
        return f"Agent({name=}, {id=})"

    def __repr__(self):
        return str(self)

    def update(self, current_system_state: dict):
        print(f"Agent {self.name} received system state: {current_system_state}")
