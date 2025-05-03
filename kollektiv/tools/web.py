from duckduckgo_search import DDGS
import trafilatura
import json

from ..llm.messages import ToolMessage


class WebClient:
    def __init__(self):
        self.ddgs: DDGS = DDGS()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0"
        }
        self.timeout = 10

    def search(self, query: str) -> ToolMessage:
        """
        Perform a search query using the DuckDuckGo search engine and return the results.
        Args:
            query (str): The search query string.
        Returns:
            ToolMessage: A message containing the search results in JSON format. If no results
            are found, the content of the message will indicate that no results were found.
        """
        results = self.ddgs.text(keywords=query, max_results=5)
        if not results:
            return ToolMessage(content="<No results found>")

        return ToolMessage(
            content=(
                f"Search results for '{query}':\n"
                f"```json\n{json.dumps(results, indent=2)}\n```"
            )
        )

    def browse(self, url: str) -> ToolMessage:
        """
        Fetches and extracts the content of a web page from the given URL.
        Args:
            url (str): The URL of the web page to fetch and extract content from.
        Returns:
            ToolMessage: A message object containing the extracted content of the web page
                         or an error message if the operation fails. If the content is
                         successfully extracted, it will be formatted as a string. If the
                         operation fails, the content will indicate the error encountered.
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                return ToolMessage(content=f"Failed to fetch {url}")
            content = trafilatura.extract(downloaded)
            if content is None:
                return ToolMessage(content=f"Failed to extract content from {url}")

            return ToolMessage(content=(f"Page content from {url}:\n\n" f"{content}"))
        except Exception as e:
            return ToolMessage(content=(f"Error fetching {url}:\n\n" f"{str(e)}"))
