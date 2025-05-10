from duckduckgo_search import DDGS
import trafilatura
import logging

from kollektiv.core import ToolMessage


class WebClient:
    logger = logging.getLogger(__name__)

    @staticmethod
    def web_search(query: str) -> ToolMessage:
        """
        Perform a search query using the DuckDuckGo search engine and return the results.
        Args:
            query (str): The search query string.
        Returns:
            ToolMessage: A message containing the search results in JSON format. If no results
            are found, the content of the message will indicate that no results were found.
        """
        WebClient.logger.info(f"Performing web search for query: '{query}'...")
        ddgs: DDGS = DDGS()
        results = ddgs.text(keywords=query, max_results=5)
        if not results:
            WebClient.logger.warning(f"No results found for query: '{query}'")
            return ToolMessage(f"!! [WARNING] No results found for query '{query}'")

        WebClient.logger.info(f"Found {len(results)} results for query: '{query}'")
        WebClient.logger.debug(f"Search results: {results}")
        return ToolMessage(
            content=(f"Search results for '{query}':\n" f"<results>{results}</results>")
        )

    @staticmethod
    def web_browse(url: str) -> ToolMessage:
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
        WebClient.logger.info(f"Browsing web content at URL: '{url}'...")
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                WebClient.logger.warning(
                    f"trafilatura.fetch_url returned None from URL: '{url}'"
                )
                return ToolMessage(
                    f"!! [WARNING] trafilatura.fetch_url returned None from URL '{url}'"
                )
            content = trafilatura.extract(downloaded)
            if content is None:
                WebClient.logger.warning(
                    f"trafilatura.extract returned None for downloaded URL: '{url}'"
                )
                return ToolMessage(
                    f"!! [WARNING] trafilatura.extract returned None for downloaded URL '{url}'"
                )

            max_words = 3000
            words = content.split()
            if len(words) > max_words:
                WebClient.logger.info(
                    f"Content from URL '{url}' truncated to {max_words} words"
                )
                content = " ".join(words[:max_words]) + "... [truncated]"
            else:
                WebClient.logger.info(
                    f"Successfully extracted content from URL '{url}' ({len(words)} words)"
                )

            WebClient.logger.debug(f"Extracted content: {content}")
            return ToolMessage(
                (f"Page content from '{url}':\n" f"<content>{content}</content>")
            )
        except Exception as e:
            WebClient.logger.error(
                f"Exception during processing of URL '{url}': {e}", exc_info=True
            )
            return ToolMessage(
                f"!! [ERROR] Exception during processing of URL '{url}': {e}"
            )
