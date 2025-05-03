import ollama
import pydantic
import random

from .messages import (
    Message,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    SystemMessage,
)
from .format_handler import FormatHandler
from .tools_handler import ToolHandler


class LLMClient:
    def __init__(self, model_name: str = "mistral-nemo:latest") -> None:
        self.model_name = model_name
        self.context_window = 2048

    def chat(
        self,
        message: str,
        message_history: list[Message] = [],
        format: pydantic.BaseModel = None,
        verbose: bool = True,
        tools: list = None,
    ) -> tuple[Message, list[Message]]:
        if message:
            message_history.append(UserMessage(message).print(verbose))

        model_input = message_history.copy()

        if tools or format:
            instructions = ""
            if tools:
                tools_handler = ToolHandler(tools)
                instructions += tools_handler.instructions

            if tools and format:
                instructions += "\n\n---\n\n"

            if format:
                format_handler = FormatHandler(format)
                instructions += format_handler.instructions

            instructions = instructions.strip()
            model_input.append(SystemMessage(instructions).print(verbose))

        for _ in range(10):
            response = ollama.chat(
                self.model_name,
                messages=[m.__dict__ for m in model_input],
                stream=verbose,
                options={
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "num_ctx": self.context_window,
                    "seed": random.randint(0, 2**30 - 1),
                },
            )

            if verbose:
                AssistantMessage("")._print_title()
                chunks = []
                for chunk in response:
                    chunk = chunk.message.content
                    print(chunk, end="", flush=True)
                    chunks.append(chunk)
                print()
                response = "".join(chunks)
                ai_message = AssistantMessage(response)
            else:
                response = response.message.content
                ai_message = AssistantMessage(response).print(verbose)

            if not format and not tools:
                message_history.append(ai_message)
                return response, message_history

            if response.startswith("<think>"):
                response = response.split("</think>")[-1].strip()

            if tools and response.startswith("INVOKE_TOOL"):
                ok, msg = tools_handler.resolve(response)
                model_input.append(ai_message)
                model_input.append(msg.print(verbose))
                if not ok:
                    continue

                message_history.append(ai_message)
                message_history.append(msg)
                continue

            if format:
                ok, msg = format_handler.resolve(response)
                if not ok:
                    model_input.append(ai_message)
                    model_input.append(msg.print(verbose))
                    continue

            message_history.append(ai_message)
            return msg, message_history
