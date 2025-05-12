import os
import ollama
import pydantic
import random
import logging
from typing import List, Callable, Tuple, Optional
import re
from pydantic import BaseModel
from ollama._utils import convert_function_to_tool

from kollektiv.core import Message, ToolMessage
from kollektiv.task.result import Result
from kollektiv.task.step import Step


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


class ToolStep(Step):
    def __init__(self, name: str, tool: Callable):
        super().__init__(name, "")
        self.tool = tool

    def instructions(self) -> str:
        return (
            "# Tools\n"
            "You may call one or more functions to assist with the user query.\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{str(dump_function(self.tool))}\n"
            "</tools>\n"
            "\n"
            "For each function call, return a json object with function name and parameters within <tool_call></tool_call> XML tags, like:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "parameters": <args-json-object>}\n'
            "</tool_call>"
        )

    def extract_tool_calls(self, request: str) -> list[ToolCall]:
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, request, re.DOTALL)

        tool_calls = []
        failures = []
        for call_text in matches:
            try:
                call_text = call_text.strip()
                tool_calls.append(ToolCall.model_validate_json(call_text))
            except Exception as e:
                failures.append(f'"{call_text}" caused "{e}"')
                logging.warning(f"Failed to parse tool call: {e}")

        return tool_calls, failures

    def on_validate_request(self, request: Message) -> Result:
        request = request.content
        has_tags = "<tool_call>" in request and "</tool_call>" in request
        if not has_tags:
            return Result(False, "Request must contain <tool_call> tags")

        tool_calls, failures = self.extract_tool_calls(request)
        if failures:
            failures = "\n".join(failures)
            return Result(False, f"Failed to parse tool calls: {failures}")

        failures = []
        for call in tool_calls:
            if call.name != self.tool.__name__:
                failures.append(
                    f"Tool name '{call.name}' does not match expected '{self.tool.__name__}'"
                )
            if not isinstance(call.parameters, dict):
                failures.append("Tool parameters must be a JSON object")
        if failures:
            failures = "\n".join(failures)
            return Result(False, f"Tool call validation failed: {failures}")

        return Result(True, None)

    def execute(self, request: Message, history: List[Message]) -> Message:
        tool_calls, failures = self.extract_tool_calls(request.content)
        results = [self.tool(**call.parameters) for call in tool_calls]
        response = "\n\n".join([result.content for result in results])
        self.logger.debug(f"Tool results: {results}")
        return ToolMessage(response)
