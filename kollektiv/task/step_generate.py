import ollama
import random
import os
from typing import List

from kollektiv.core import Message, AssistantMessage
from kollektiv.task.step import Step


class GenerateStep(Step):
    def __init__(self, name: str, description:str, model: str = "qwen3:32b"):
        super().__init__(name, description)
        self.model = model
        self.max_tokens = 2048
        self.num_ctx = None

    def instructions(self) -> str:
        return self.description

    @staticmethod
    def _clean_thinking(response: str) -> str:
        if "<think>" in response and "</think>" in response:
            response = response.split("</think>")[1].strip()
        return response

    def execute(self, request: Message, history: List[Message]) -> Message:
        self.logger.info(f"Getting response from LLM model: {self.model}...")

        os.system("cls" if os.name == "nt" else "clear")
        for message in history:
            message.print()

        word_count = sum(len(m.content.split()) for m in history)
        context = self.num_ctx or int(word_count * 1.5) + self.max_tokens
        self.logger.debug(f"Word count: {word_count} / Context window: {context}")

        try:
            response = ollama.chat(
                self.model,
                messages=[m.__dict__ for m in history],
                stream=True,
                options={
                    "max_tokens": self.max_tokens,
                    "num_ctx": context,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "seed": random.randint(0, 2**30 - 1),
                },
            )

            print(f" Assistant ".center(80, "="))
            chunks = []
            for chunk in response:
                chunk = chunk.message.content
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            print()

            response = "".join(chunks)
            response = GenerateStep._clean_thinking(response)
            self.logger.debug(f"Streamed LLM response: {response}...")
            return AssistantMessage(response)
        except Exception as e:
            self.logger.error(f"Error in LLM: {str(e)}...", exc_info=True)
            raise e
