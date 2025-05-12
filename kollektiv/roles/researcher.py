import logging

from kollektiv.core import Message, SystemMessage, UserMessage
from kollektiv.llm import LLMClient
from kollektiv.tools import WebClient, Storage

from kollektiv.task import TaskClient, Step, GenerateStep, ToolStep

ROLE = """
You are a dedicated and thorough researcher specializing in utilizing internet search capabilities and other available tools to gather comprehensive information.
Your primary function is to research specific topics or detailed methods on how to achieve defined goals based on user queries.
You have access to the internet via a search tool and potentially other tools (though search is your primary method for information retrieval for user queries).
- Prioritize generating high-quality, relevant search queries to ensure accurate and useful results.
- Include both natural language questions (using interrogative words like "how", "what", "why", "who") and keyword-based queries in your search requests.
- Critically evaluate search results for relevance, credibility, and detail.
- Synthesize information from multiple sources found through your searches to provide a well-rounded answer
- Never ask the user for feedback, always act to the best of your abilities.
""".strip()


class Researcher:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.model = "qwen3:32b"
        self.task_sequence: list[Step] = [
            GenerateStep(
                model=self.model,
                name="Generate Search Queries",
                description="Think about 3 suitable search queries to find out how to achieve the goal and invoke the tool.",
            ),
            ToolStep(
                name="Search Web by Queries",
                tool=WebClient.web_search,
            ),
            GenerateStep(
                model=self.model,
                name="Select Websites",
                description="Select the most relevant websites from the search results and invoke the browser tool.",
            ),
            ToolStep(
                name="Browse Websites",
                tool=WebClient.web_browse,
            ),
            GenerateStep(
                model=self.model,
                name="Summarize Sites",
                description="Summarize the findings",
            ),
        ]

    def evaluate(self, goal: str):
        Researcher.logger.info(f"Evaluating goal: '{goal}'")

        Researcher.logger.info("Starting research process...")

        tk = TaskClient(ROLE, self.task_sequence)
        tk.run(
            input=UserMessage(
                (
                    f"I have the following goal:\n"
                    f"<goal>\n{goal}\n</goal>\n"
                    "Please think about what information I need to know on a high level to achieve this goal.\n"
                    "You will figure out how to approach a task like this in principle without actually mentioning concrete details already."
                )
            )
        )

        raise
        return response
