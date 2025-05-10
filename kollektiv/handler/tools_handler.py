from langchain_core.tools import tool
from pydantic import BaseModel, Field


from kollektiv.handler.handle import Handler
from kollektiv.core import ToolMessage


class ToolCall(BaseModel):
    name: str = Field(description="The name of the tool to call.")
    arguments: dict = Field(description="The arguments to pass to the tool.")


class ToolHandler(Handler):

    def __init__(self, tools: list[callable], retry_attempts: int = 3):
        self.tools = tools
        self.tool_mapping = {t.__name__: t for t in tools}
        super().__init__(retry_attempts)

    def _prepare_instructions(self) -> str:
        instructions = "You currently have access to this tool(s):\n\n"
        for i, (name, t) in enumerate(self.tool_mapping.items()):
            instructions += (
                f"** Tool #{i} **:\n"
                f"Name: {name}\n"
                f"Description: {t.__doc__}\n"
                f"Function schema:\n"
                f"```json\n{tool(t).args_schema.model_json_schema()}\n```\n\n"
            )

        instructions += (
            "** Instructions on how to use the tool(s) **\n"
            "IF you want to invoke the tool, you MUST respond with the `INVOKE_TOOL` prefix followed by this schema:\n"
            f"```json\n{ToolCall.model_json_schema()}\n```\n"
            "\n"
            "For example:\n"
            "INVOKE_TOOL```json\n"
            '{"name": "foo", "arguments": {"query": "bar"}}\n'
            "```"
            "\n"
            "Note: You can only use one tool at a time!\n"
            "You will receive the result in the next response."
        )
        return instructions

    def consider(self, response: str) -> bool:
        return response.startswith("INVOKE_TOOL")

    def _invoke(self, response: str) -> ToolMessage:
        response = response[11:].strip()

        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()

        response = ToolCall.model_validate_json(response)

        tool_name = response.name
        tool_args = response.arguments
        tool = self.tool_mapping[tool_name]
        result = tool(**tool_args)

        return ToolMessage(
            f"Tool '{tool_name}' executed successfully.\n\n{result.content}"
        )
