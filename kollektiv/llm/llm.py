import ollama
import pydantic
import random
from typing import List, Callable, Tuple, Optional

from .messages import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
)
from .handler import Handler, ToolHandler, FormatHandler


def _clean_thinking(response: str) -> str:
    if response.startswith("<think>"):
        response = response.split("</think>")[-1].strip()
    return response


class LLMClient:
    def __init__(self, model_name: str = "mistral-nemo:latest") -> None:
        self.model_name = model_name
        self.context_window = 2048
        self.context_window_dynamic = False
        self.debug = False

    def _get_response(self, messages: list[Message], verbose: bool) -> str:
        if self.debug:
            _ = input("Enter to clear the screen and continue...")
            import os

            os.system("cls" if os.name == "nt" else "clear")
            for message in messages:
                message.print()

        total_word_count = sum(len(m.content.split()) for m in messages)
        context_windows = (
            self.context_window
            if not self.context_window_dynamic
            else int(total_word_count * 1.5)
        )
        print(
            f"[DEBUG] Input word count: {total_word_count} / Context window: {context_windows}"
        )

        response = ollama.chat(
            self.model_name,
            messages=[m.__dict__ for m in messages],
            stream=verbose,
            options={
                "temperature": 0.5,
                "top_p": 0.9,
                "num_ctx": context_windows,
                "seed": random.randint(0, 2**30 - 1),
            },
        )

        if not verbose:
            response = _clean_thinking(response.message.content)
            return AssistantMessage(response)

        AssistantMessage("")._print_title()
        chunks = []
        for chunk in response:
            chunk = chunk.message.content
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print()
        response = "".join(chunks)
        response = _clean_thinking(response)
        return AssistantMessage(response)

    def _force_handler(
        self, history: List[Message], handler: Handler, verbose: bool
    ) -> Tuple[Message, List[Message]]:
        model_input = history.copy()
        model_input.append(SystemMessage(handler.instructions).print(verbose))

        while True:
            ai_message = self._get_response(model_input, verbose)
            ok, response = handler.invoke(ai_message.content)
            if not ok:
                model_input.append(ai_message)
                model_input.append(response.print(verbose))
                continue

            history.append(ai_message)
            if issubclass(response.__class__, Message):
                history.append(response.print(verbose))
            return response, history

    def chat(
        self,
        message: str,
        history: List[Message] = [],
        format: Optional[pydantic.BaseModel] = None,
        verbose: bool = True,
        tools: Optional[List[Callable]] = None,
        tools_forced_sequence: bool = False,
    ) -> Tuple[Message, List[Message]]:
        history = history.copy()
        history.append(UserMessage(message).print(verbose))

        if not tools and not format:
            ai_message = self._get_response(history, verbose)
            history.append(ai_message)
            return ai_message.content, history

        if tools_forced_sequence:
            for tool in tools:
                handler = ToolHandler([tool])
                response, history = self._force_handler(history, handler, verbose)

            if format:
                handler = FormatHandler(format)
                response, history = self._force_handler(history, handler, verbose)
            elif tools:
                # if tools were used but no format is provided, we need to invoke the LLm once more to get the final response
                ai_message = self._get_response(history, verbose)
                history.append(ai_message)
                response = ai_message.content

            return response, history

        model_input = history.copy()

        instructions = ""
        if tools:
            handler_tools = ToolHandler(tools)
            instructions += handler_tools.instructions
        if tools and format:
            instructions += (
                "\n\n---\n"
                "You can either use the tool(s) outlined above OR format your final according to the following instructions.\n"
                "---\n\n"
            )
        if format:
            handler_format = FormatHandler(format)
            instructions += handler_format.instructions

        if tools or format:
            model_input.append(SystemMessage(instructions).print(verbose))

        while True:
            ai_message = self._get_response(model_input, verbose)

            if tools and handler_tools.consider(ai_message.content):
                ok, response = handler_tools.invoke(ai_message.content)
                model_input.append(ai_message)
                model_input.append(response.print(verbose))
                if not ok:
                    continue
                history.append(ai_message)
                history.append(response)
                continue
            if format:
                ok, response = handler_format.invoke(ai_message.content)
                if not ok:
                    model_input.append(ai_message)
                    model_input.append(response.print(verbose))
                    continue
                history.append(ai_message)
                return response, history

            return ai_message.content, history
