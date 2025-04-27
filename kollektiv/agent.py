import pydantic
import datetime

from .tools import Tool, Function
from .llm import LLMClient
from .role import Role
from .agent_memory import Memory


class AgentAction(pydantic.BaseModel):
    target: str
    input: dict


class AgentResponse(pydantic.BaseModel):
    actions: list[AgentAction]


class Agent:
    def __init__(self, llm: LLMClient, name: str, role: Role):
        self.id = id(self)
        self.llm: LLMClient = llm
        self.name: str = name
        self.role: Role = role
        self.memory: Memory = Memory(agent=self, llm=llm)

        # gets set by the system
        self.system_goal: str = None
        self.tools: list[Tool] = None
        self.agents: list[Agent] = None

        # internal state
        self.current_context: str = None
        self.inbox: list = []
        self.inbox_archive: list = []
        self.tool_invoke_history: list[tuple[str, str]] = []

    def __str__(self):
        return f"Agent(name={self.name}, role={self.role})"

    def __repr__(self):
        return str(self)

    @property
    def team(self):
        return [agent for agent in self.agents if agent.id != self.id]

    def find_tool(self, tool_name: str, find_in: list[Tool]) -> Tool:
        tool = [t for t in find_in if t.name == tool_name]
        if not tool:
            return None
        return tool[0]

    def prepare(self, system_state):
        self.current_context = f"""
Current time: {system_state.time}.
You are {self.name}, a {self.role} agent.
Your team includes: {self.team}.
The team tries to achieve: {self.system_goal}.

You have the following impressions of your team members:
{self.memory.agent_summaries}
"""

    def process_phase(
        self,
        prompt: str,
        time: datetime.datetime,
        time_delta_per_cycle: int,
        cycles: int,
        tools: list[Tool],
        retry_count: int = 3,
    ):
        prompt = f"""{self.current_context}
You can use the following tools:
{[tool.get_json_schema() for tool in tools]}

You must strictly respond in JSON format by providing this information:
"target": "<tool_name>.<function_name>"
"input": <function_input>

You will have the opportunity to cycle {cycles} time(s) through this process step.
After each cycle, your chosen action(s) will be performed and the result(s) will be presented to you.
Keep your answers short and concise.
"""
        for cycle in range(cycles):
            prompt = self.process_phase_cycle(prompt, time, cycle, cycles, tools, retry_count)
            time += datetime.timedelta(minutes=time_delta_per_cycle)

        return prompt, time

    def process_phase_cycle(
        self,
        prompt: str,
        time: datetime.datetime,
        cycle: int,
        cycles: int,
        tools: list[Tool],
        retry_count: int,
    ):
        prompt += f"""
---------------------------------
Starting Cycle {cycle+1} of {cycles}.
Current time: {time}.

What are your next actions, {self.name}?
If you are done and do not need all cycles, respond with empty action list to finish.
"""

        for attempt in range(retry_count):
            # print(f'Input: "{prompt}"')
            response = self.llm.generate(prompt, format=AgentResponse)
            prompt += f"""
---------------------------------
Processing your request...
---------------------------------
System response:
"""

            if len(response.actions) == 0:
                prompt += f"Agent responded with empty action list, assuming done.\n"
                return prompt

            fail_count = 0

            def _handle_failure(message: str):
                nonlocal fail_count, prompt
                fail_count += 1
                message = f"[FAILED] {message}"
                prompt += f"- {message}\n"
                print(f" - {message}")

            for action in response.actions:
                target = action.target
                print(f"Agent {self.name} function: {target}", end="")

                parts = target.split(".")
                if len(parts) != 2:
                    _handle_failure(f"Invalid target: {target}")
                    continue

                tool_name, function_name = parts
                tool = self.find_tool(tool_name, find_in=tools)
                if not tool:
                    _handle_failure(f"Invalid tool: {tool_name}")
                    continue
                if function_name not in tool.functions:
                    _handle_failure(f"Invalid function: {function_name}")
                    continue

                function: Function = tool.functions[function_name]
                try:
                    fin = function.func_TIn(**action.input)
                except Exception as e:
                    _handle_failure(f"Invalid input: {e}")
                    continue

                result = function.func(self, fin)
                if result.status.code != 200:
                    _handle_failure(result.status.message)
                    continue

                prompt += f"- [OK] {action.target}, Input: {fin}, Output: {result}\n"
                print(f" - OK")
                self.memory.add_to_history(time, f"Performed: {action.target}")

            if fail_count == 0:
                break

            prompt += f"""
---------------------------------
It appears that {fail_count} actions failed to process successfully.
Please try again (attempt {attempt+1} of {retry_count}) or respond with empty action list if you're done anyways.
"""
        return prompt

    def plan(self, system_state: dict):
        print(f'{f" {self.name} PLANING ":=^50} ')

        planner = self.find_tool("Planner", self.tools)
        assert planner, "Planner tool not found"

        prompt = f"""{self.current_context}
---------------------------------
Process:
- 1. Planning
- 2.1. Acting
- 2.2. Message Updates to Team
- 2.3. Update Planner
- 3. Reflecting
Current stage: 1. Planning

Your goal is to plan out your day.
In the next system tick, you will have the opportunity to act upon your plan.

Yesterday, you performed the following actions:
{self.memory.history_log}

You currently have the following new messages:
{self.inbox}

Your current tasklist is:
{planner.get_tasks(self, {}).tasks}

Maybe you need to break down a task into smaller tasks.
You can also delete tasks that are no longer relevant.
It's important to be specific and clear about your intentions.
It's a good idea to keep the overview by deleting completed items from the list.
If you don't want to change anything, respond with empty actions list.

Is there anything you would like to add/update/delete based on your current information, {self.name}?
"""
        self.inbox_archive.extend(self.inbox)
        _ = [self.memory.add_agent_memory(msg.sender, msg) for msg in self.inbox]
        self.inbox = []
        self.memory.history_log = []

        _ = self.process_phase(
            prompt=prompt,
            time=system_state.time,
            time_delta_per_cycle=30,
            cycles=3,
            tools=[planner],
        )

    def act(self, system_state: dict):
        print(f'{f" {self.name} ACTING ":=^50} ')

        planner = self.find_tool("Planner", self.tools)
        assert planner, "Planner tool not found"
        storage = self.find_tool("Storage", self.tools)
        assert storage, "Storage tool not found"

        prompt = f"""{self.current_context}
---------------------------------
Process:
- 1. Planning
- 2.1. Acting
- 2.2. Message Updates to Team
- 2.3. Update Planner
- 3. Reflecting
Current stage: 2.1. Acting

Your are currently working on your tasks.
The goal in this stage is to perform the actions you planned out in the previous stage (i.e. work on your tasks).

Your current tasklist is:
{planner.get_tasks(self, {}).tasks}

You now have the opportunity to perform one or more actions to get closer to your goal.
"""
        prompt, time = self.process_phase(
            prompt=prompt,
            time=system_state.time,
            time_delta_per_cycle=30,
            cycles=5,
            tools=[storage],
            retry_count=3,
        )

        ######  Stage 2.2 - Message Updates to Team
        time += datetime.timedelta(minutes=30)
        messenger = self.find_tool("Messenger", self.tools)
        assert messenger, "Messenger tool not found"

        prompt += f"""
---------------------------------
Current stage: 2.2. Message Updates to Team

You have the opportunity to update and respond to your team about what you've done.
"""

        prompt, time = self.process_phase(
            prompt=prompt,
            time=time,
            time_delta_per_cycle=30,
            cycles=1,
            tools=[messenger],
            retry_count=3,
        )

        #######  Stage 2.3 - Update Planner
        time += datetime.timedelta(minutes=30)
        prompt += f"""
---------------------------------
Current stage: 2.3. Update Planner

You have the opportunity to update your planner to reflect the actions you performed.
You can add new tasks, delete completed tasks, or update existing tasks.
Make sure they are specific and clear about your intentions such that the future you can understand them.
If you don't want to change anything, respond with empty actions list.
"""

        _ = self.process_phase(
            prompt=prompt,
            time=time,
            time_delta_per_cycle=30,
            cycles=1,
            tools=[planner],
        )

    def reflect(self, system_state: dict):
        self.memory.summarize_memories(system_state.time)
