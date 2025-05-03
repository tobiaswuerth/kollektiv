from langchain_core.tools import tool
from pydantic import BaseModel, Field
from .messages import ToolMessage


class ToolCall(BaseModel):
    name: str = Field(description="The name of the tool to call.")
    arguments: dict = Field(description="The arguments to pass to the tool.")


class ToolHandler:

    def __init__(self, tools: list[callable], retry_attempts: int = 3):
        self.tools = tools
        self.retry_attempts = retry_attempts
        self.attempt = 0
        self.tool_mapping = {t.__name__: t for t in tools}
        self.instructions = self._prepare_instructions(tools)

    def _prepare_instructions(
        self, tools: list[callable]
    ) -> tuple[str, dict[str, callable]]:
        instructions = "You currently have access to the following tools:\n\n"
        for i, (name, t) in enumerate(self.tool_mapping.items()):
            instructions += (
                f"** Tool #{i} **:\n"
                f"Name: {name}\n"
                f"Description: {t.__doc__}\n"
                f"Function schema:\n"
                f"```json\n{tool(t).args_schema.model_json_schema()}\n```\n\n"
            )

        instructions += (
            "** Instructions on how to use the tools **\n"
            "If you want to use a tool, you must respond with `INVOKE_TOOL` followed by this schema:\n"
            f"```json\n{ToolCall.model_json_schema()}\n```\n"
            "\n"
            "For example:\n"
            "INVOKE_TOOL```json\n"
            '{"name": "foo", "arguments": {"query": "bar"}}\n'
            "```"
            "\n"
            "You will receive a response in the next response.\n"
            "You are free to use the tools again if you need to do so."
        )
        return instructions

    def resolve(self, response: str) -> tuple[bool, ToolMessage]:
        try:
            if not response.startswith("INVOKE_TOOL"):
                raise ValueError("Response does not start with 'INVOKE_TOOL'.")
            response = response[11:].strip()

            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()

            response = ToolCall.model_validate_json(response)

            tool_name = response.name
            tool_args = response.arguments
            tool = self.tool_mapping[tool_name]
            result = tool(**tool_args)

            return True, ToolMessage(
                f"Tool '{tool_name}' executed successfully:\n\n{result}"
            )
        except Exception as e:
            self.attempt += 1
            if self.attempt >= self.retry_attempts:
                raise e

            return False, ToolMessage(
                (
                    f"!! [ERROR] {e}\n"
                    f"!! If you see this message, it means that your output did not adhere to the requested format.\n"
                    f"!! Retry attempt {self.attempt+1} of {self.retry_attempts}."
                )
            )
