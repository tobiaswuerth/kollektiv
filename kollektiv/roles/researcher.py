import logging

from kollektiv.core import Message, SystemMessage
from kollektiv.llm import LLMClient
from kollektiv.tools import WebClient, Storage


SYSTEM_PROMPT = """
# Role

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

    def evaluate(self, goal: str):
        Researcher.logger.info(f"Evaluating goal: '{goal}'")

        Researcher.logger.info("Starting research process...")
        llm = LLMClient("qwen3:32b")
        llm.debug = True
        llm.context_window_dynamic = True

        history: list[Message] = [
            SystemMessage(SYSTEM_PROMPT),
        ]

        response, _ = llm.chat(
            message=(
                f"I have the following goal:\n"
                f"<goal>\n{goal}\n</goal>\n"
                "Please provide a detailed plan on how to achieve this goal.\n"
                "You will figure out how to approach a task like this in principle without actually mentioning concrete details, by\n"
                "- think about what you need to know\n"
                "- web_search 3 different queries, wait for response\n"
                "- web_browse 3 different URLs, wait for response\n"
                "Summarize your findings on how to structure a project like this step by step."
            ),
            history=history,
            tools=[
                WebClient.web_searches,
                WebClient.web_browses,
            ],
            tools_forced_sequence=True,
        )
        Storage.write_file("research.md", response)

        Researcher.logger.debug(f"Research response:\n{response}")
        raise
        return response
