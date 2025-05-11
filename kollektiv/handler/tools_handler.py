from pydantic import BaseModel
from typing import Callable
import logging
import re
from ollama._utils import convert_function_to_tool


from kollektiv.handler.handle import Handler
from kollektiv.core import ToolMessage


def dump_function(func: Callable) -> dict:
    tool = convert_function_to_tool(func)
    tool = tool.model_dump(exclude_none=True)

    if (
        "function" in tool
        and "parameters" in tool["function"]
        and "defs" in tool["function"]["parameters"]
    ):
        tool["function"]["parameters"]["$defs"] = tool["function"]["parameters"].pop(
            "defs"
        )

    return tool


class ToolCall(BaseModel):
    name: str
    parameters: dict


class ToolHandler(Handler):
    logger = logging.getLogger(__name__)

    def __init__(self, tools: list[callable], retry_attempts: int = 3):
        self.tools = tools
        self.tool_mapping = {t.__name__: t for t in tools}
        self.tool_functions = {t.__name__: convert_function_to_tool(t) for t in tools}
        super().__init__(retry_attempts)

    def _prepare_instructions(self) -> str:
        func_models = [str(dump_function(t)) for t in set(self.tools)]
        func_models = "\n\n".join(func_models)

        instructions = (
            "# Tools\n"
            "\n"
            "You may call one or more functions to assist with the user query.\n"
            "\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{func_models}\n"
            "</tools>\n"
            "\n"
            "For each function call, return a json object with function name and parameters within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "parameters": <args-json-object>}\n'
            "</tool_call>"
        )
        return instructions

    def extract_tool_calls(self, response: str) -> list[ToolCall]:
        tool_calls = []
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, response, re.DOTALL)

        for call_text in matches:
            try:
                call_text = call_text.strip()
                tool_calls.append(ToolCall.model_validate_json(call_text))
            except Exception as e:
                ToolHandler.logger.warning(f"Failed to parse tool call: {e}")

        return tool_calls

    def consider(self, response: str) -> bool:
        return "<tool_call>" in response and "</tool_call>" in response

    def _invoke(self, response: str) -> ToolMessage:
        ToolHandler.logger.debug(f"Handling: {response}...")
        tool_calls = self.extract_tool_calls(response)

        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.name
            if tool_name not in self.tool_mapping:
                ToolHandler.logger.error(f"Tool '{tool_name}' not found. Skipping...")
                results.append(
                    f"!! [ERROR] Tool '{tool_name}' not found in the mapping. Skipping..."
                )
                continue

            ToolHandler.logger.debug(
                f"Invoking '{tool_name}' with params: {tool_call.parameters}"
            )
            tool = self.tool_mapping[tool_name]
            result = tool(**tool_call.parameters)
            results.append(
                f"Tool '{tool_name}' executed successfully.\n{result.content}"
            )

        result = ToolMessage(
            content="\n\n".join(results),
        )

        ToolHandler.logger.debug(f"Tool result: {result}")
        return ToolMessage(
            content=result.content,
        )
