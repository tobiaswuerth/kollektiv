"""Isolated browser-test server using real graph/tools/checkpoints and scripted models."""

import argparse
import json
import time
from tempfile import TemporaryDirectory
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from test_collective import ScriptedModels, task

from kollektiv.models import Plan
from kollektiv.webapp import ControlRoomServer, RunManager


class DemoModels(ScriptedModels):
    """Use a visible pace for README capture without changing graph or tool execution."""

    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    def structured(self, *args, **kwargs):
        time.sleep(self.delay)
        agent, schema, prompt = args[:3]
        if schema is Plan and "DELEGATION_BROWSER" in prompt:
            plan = Plan(
                summary="Hire a specialist family, supervise it, then write the project summary",
                success_criteria=["Complete project summary"],
                tasks=[task("summary", "summary.md", owner="builder")],
            )
            validate = args[3] if len(args) > 3 else kwargs.get("validate")
            if validate:
                validate(plan)
            return plan
        return super().structured(*args, **kwargs)

    def act(self, *args, **kwargs):
        time.sleep(self.delay)
        agent, messages, tools = args[:3]
        prompt = messages[1].content
        if prompt.startswith("SHARED GOAL:"):
            return AIMessage(content="Discuss the approach first; hire only during artifact work.")
        if "DELEGATION_BROWSER" in prompt:
            return self.delegation_act(agent, messages, tools)
        return super().act(*args, **kwargs)

    @staticmethod
    def call(name, arguments):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": arguments,
                    "id": uuid4().hex,
                    "type": "tool_call",
                }
            ],
        )

    def delegation_act(self, agent, messages, tools):
        """Real hires and scoped writes with time for browser operator intervention."""
        prompt = messages[1].content
        data = json.JSONDecoder().raw_decode(prompt.split("YOUR TASK:\n", 1)[1])[0]
        history = messages[2:]
        results = [
            message
            for message in history
            if isinstance(message, ToolMessage) and message.status != "error"
        ]
        written = any(message.name == "write_file" for message in results)
        hired = any(message.name == "hire" for message in results)
        tools_available = {tool.name for tool in tools}
        if agent in ("builder", "browser_scout") and not hired:
            child_name = "browser_scout" if agent == "builder" else "browser_leaf"
            child_tools = [
                "read_file",
                "write_file",
                "check_file",
                "send_message",
                "monitor_agents",
            ]
            if child_name == "browser_scout":
                child_tools += ["hire", "fire", "instruct_agent"]
            assert "hire" in tools_available
            return self.call(
                "hire",
                {
                    "name": child_name,
                    "role": "Research specialist"
                    if child_name == "browser_scout"
                    else "Evidence specialist",
                    "instructions": "Write complete scoped evidence, report to your parent, and follow operator corrections.",
                    "skills": ["research", "writing"],
                    "tools": child_tools,
                    "assignment": "Write a complete evidence note and remain available for parent supervision.",
                    "output": "evidence.md",
                    "acceptance_criteria": ["Contains COMPLETE evidence"],
                },
            )
        if agent == "builder":
            monitors = [
                json.loads(message.content)
                for message in results
                if message.name == "monitor_agents"
            ]
            retired = (
                monitors
                and monitors[-1].get("children", {}).get("browser_scout", {}).get("status")
                == "retired"
            )
            if not retired:
                time.sleep(0.1)
                return self.call("monitor_agents", {"wait_seconds": 0.25})
        if not written:
            return self.call(
                "write_file", {"content": f"# {data['title']}\nCOMPLETE evidence from {agent}."}
            )
        if agent == "builder":
            return AIMessage(
                content="Project summary completed after supervising the specialist family."
            )
        if agent == "browser_leaf" and "BROWSER_OPERATOR" in prompt:
            acknowledged = any(message.name == "send_message" for message in results)
            if not acknowledged:
                return self.call(
                    "send_message",
                    {
                        "recipient": "browser_scout",
                        "content": "BROWSER_OPERATOR instruction applied at the next work step.",
                    },
                )
        time.sleep(0.2)
        return self.call("monitor_agents", {"wait_seconds": 0.25})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--delay", type=float, default=0)
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("delay must not be negative")
    with TemporaryDirectory(prefix="kollektiv-ui-") as directory:
        server = ControlRoomServer(
            ("127.0.0.1", args.port), RunManager(directory, models=DemoModels(args.delay))
        )
        try:
            server.serve_forever()
        finally:
            server.server_close()
