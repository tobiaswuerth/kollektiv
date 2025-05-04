from dataclasses import dataclass
import datetime

from .agent import Agent
from .tools import Tool


@dataclass
class SystemState:
    time: datetime.datetime


class System:
    def __init__(self, goal: str, agents: list[Agent], tools: list[Tool]):
        self.goal = goal
        self.agents: list[Agent] = agents
        self.tools: list[Tool] = tools

        self.time_started = datetime.datetime.now() + datetime.timedelta(days=-365)
        self.cycle = 0

        for agent in self.agents:
            agent.tools = self.tools
            agent.agents = self.agents
            agent.system_goal = self.goal

    @property
    def current_day(self) -> int:
        return self.time_started + datetime.timedelta(days=self.cycle)

    def get_current_daytime(self, hour: int) -> datetime.datetime:
        return self.current_day.replace(hour=hour, minute=0, second=0, microsecond=0)

    def create_system_state(self, hour: int) -> SystemState:
        return SystemState(time=self.get_current_daytime(hour=hour))

    def tick(self):
        # each day in the agents life consists of the following cycle:
        # 1. prepare the agent
        # 2. plan its next action
        # 3. perform the action
        # 4. reflect on the action
        
        state = self.create_system_state(hour=9)
        _ = [tool.update_system_state(state) for tool in self.tools]
        _ = [agent.prepare(state) for agent in self.agents]

        state = self.create_system_state(hour=10)
        _ = [tool.update_system_state(state) for tool in self.tools]
        _ = [agent.plan(state) for agent in self.agents]

        state = self.create_system_state(hour=13)
        _ = [tool.update_system_state(state) for tool in self.tools]
        _ = [agent.act(state) for agent in self.agents]

        state = self.create_system_state(hour=17)
        _ = [tool.update_system_state(state) for tool in self.tools]
        _ = [agent.reflect(state) for agent in self.agents]

        self.cycle += 1
