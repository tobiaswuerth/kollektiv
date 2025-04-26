from .agent import Agent


class System:
    def __init__(self, goal: str, agents: list[Agent]):
        self.goal = goal
        self.time = 0
        self.agents: list[Agent] = agents

    def get_agent_state(self, agent: Agent):
        return {
            "your_team": [a for a in self.agents if a != agent],
            "team_goal": self.goal,
            "system_time": self.time,
        }

    def tick(self):
        self.time += 1

        for agent in self.agents:
            state = self.get_agent_state(agent)
            agent.update(state)

        print(f"System tick: {self.time}")
