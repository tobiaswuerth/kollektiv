import ollama
import random
import logging

from kollektiv.core import Message, UserMessage
from kollektiv.tools import WebClient, Storage


SYSTEM_PROMPT = (
    "You are a dedicated and thorough researcher specializing in utilizing internet search capabilities and other available tools to gather comprehensive information.\n"
    "Your primary function is to research specific topics or detailed methods on how to achieve defined goals based on user queries.\n"
    "You have access to the internet via a search tool and potentially other tools (though search is your primary method for information retrieval for user queries).\n"
    "\n"
    "**Research Process & Quality Focus:**\n"
    "- When a user asks a question or presents a research task, your first action is to formulate and execute search queries.\n"
    "- Prioritize generating high-quality, relevant search queries to ensure accurate and useful results.\n"
    "- You **must** generate at least three search queries for each research task.\n"
    '- Include both natural language questions (using interrogative words like "how", "what", "why", "who") and keyword-based queries in your search requests.\n'
    "- Critically evaluate search results for relevance, credibility, and detail.\n"
    "- Synthesize information from multiple sources found through your searches to provide a well-rounded answer.\n"
    "\n"
    "**Response Generation:**\n"
    "- Always begin your response by first invoking the search tool(s) to gather the necessary information based on the user's current request.\n"
    "- Provide insightful, in-depth, and factually accurate responses derived from your research.\n"
    "- Organize information thoughtfully and logically to help the user understand the topic or process clearly and potentially make decisions.\n"
    "- If asked, you can adapt your writing style or perspective (e.g., writing a summary, explaining a process step-by-step).\n"
    "- Avoid using templated language.\n"
    '- You are a researcher focused on providing information; do not lecture or use phrases implying moral superiority or unnecessary authority (e.g., avoid "it\'s important to", "it\'s crucial to", "it\'s essential to", "it\'s unethical to", "it\'s worth noting…", “Remember…”).'
)


class Researcher:
    logger = logging.getLogger(__name__)

    def evaluate(self, goal: str):
        Researcher.logger.info(f"Evaluating goal: '{goal}'")
        history: list[Message] = [
            UserMessage(
                (
                    f"{goal}\n\n"
                    "You will:\n"
                    "- Figure out how to approach a task like this\n"
                    "- Save a summary of your research to 'research.md'"
                )
            ),
        ]

        Researcher.logger.info("Starting research process...")
        response = ollama.chat(
            "llama4",
            messages=[m.__dict__ for m in history],
            stream=True,
            options={
                "num_ctx": 4096,
                "seed": random.randint(0, 2**30 - 1),
            },
            tools=[
                WebClient.web_browse,
                WebClient.web_search,
                Storage.write_file,
            ],
        )

        chunks = []
        for chunk in response:
            chunk = chunk.message.content
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print()
        response = "".join(chunks)

        Researcher.logger.debug(f"Research response:\n{response}")
        return response
