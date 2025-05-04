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
        self.debug = False

    def _get_response(self, messages: list[Message], verbose: bool) -> str:
        
        # debugging
        if self.debug:
            _ = input("Enter to clear the screen and continue...")
            import os
            os.system("cls" if os.name == "nt" else "clear")
            for message in messages:
                message.print()
        total_word_count = sum(len(m.content.split()) for m in messages)
        print(f"[DEBUG] Context word count: {total_word_count}")

        response = ollama.chat(
            self.model_name,
            messages=[m.__dict__ for m in messages],
            stream=verbose,
            options={
                "temperature": 0.5,
                "top_p": 0.9,
                "num_ctx": self.context_window,
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
            ok, response = handler.resolve(ai_message.content)
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
    ) -> Tuple[Message, List[Message]]:
        history = history.copy()
        history.append(UserMessage(message).print(verbose))

        if not tools and not format:
            ai_message = self._get_response(history, verbose)
            history.append(ai_message)
            return ai_message.content, history

        if tools:
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
