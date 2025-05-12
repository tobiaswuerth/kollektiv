from langchain_community.utilities.brave_search import BraveSearchWrapper
import trafilatura
import logging

from kollektiv.core import ToolMessage
from kollektiv.config import config

import os

os.environ["BRAVE_SEARCH_API_KEY"] = config.brave_search_api_key


class WebClient:
    logger = logging.getLogger(__name__)

    @staticmethod
    def web_search(query: str) -> ToolMessage:
        """
        Search the web using DuckDuckGo and return top results.

        Args:
            query (str): Search query string

        Returns:
            ToolMessage: Message containing search results or warning if none found
        """
        WebClient.logger.info(f"Performing web search: '{query}'")
        bs = BraveSearchWrapper(
            search_kwargs={
                "count": 5,
                "extra_snippets": False,
            }
        )
        results = bs.run(query)
        WebClient.logger.debug(f"Search results: {results}")
        return ToolMessage(
            f"Search results for '{query}':\n<results>{results}</results>"
        )

    @staticmethod
    def web_searches(queries: list[str]) -> ToolMessage:
        """
        Perform multiple web searches (max 3) using DuckDuckGo.

        Args:
            queries (list[str]): List of search queries

        Returns:
            ToolMessage: Combined results from all searches or error if too many queries
        """
        if len(queries) > 3:
            WebClient.logger.error(f"Too many queries: {len(queries)}, maximum is 3")
            return ToolMessage(
                f"!! [ERROR] Too many queries: {len(queries)}, maximum is 3"
            )

        results = [WebClient.web_search(query) for query in queries]
        WebClient.logger.info(
            f"Completed {len(results)} searches for queries: {queries}"
        )
        WebClient.logger.debug(f"Search results: {results}")
        contents = [result.content for result in results]
        return ToolMessage(content="\n\n".join(contents))

    @staticmethod
    def web_browse(url: str) -> ToolMessage:
        """
        Fetch and extract content from a web page.

        Args:
            url (str): URL to retrieve content from

        Returns:
            ToolMessage: Extracted page content or error/warning message
        """
        WebClient.logger.info(f"Browsing URL: '{url}'")
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                WebClient.logger.warning(f"Failed to fetch URL: '{url}'")
                return ToolMessage(f"!! [WARNING] Failed to fetch URL: '{url}'")
            content = trafilatura.extract(downloaded)
            if content is None:
                WebClient.logger.warning(f"Failed to extract content from URL: '{url}'")
                return ToolMessage(
                    f"!! [WARNING] Failed to extract content from URL: '{url}'"
                )

            max_words = 3000
            words = content.split()
            if len(words) > max_words:
                WebClient.logger.info(
                    f"Content truncated to {max_words} words for URL: '{url}'"
                )
                content = " ".join(words[:max_words]) + "... [truncated]"
            else:
                WebClient.logger.info(
                    f"Successfully extracted {len(words)} words from URL: '{url}'"
                )

            WebClient.logger.debug(f"Extracted content: {content}")
            return ToolMessage(
                (f"Page content from '{url}':\n" f"<content>{content}</content>")
            )
        except Exception as e:
            WebClient.logger.error(f"Error processing URL '{url}': {e}", exc_info=True)
            return ToolMessage(
                f"!! [ERROR] Error processing URL: '{url}', details: {e}"
            )

    @staticmethod
    def web_browses(urls: list[str]) -> ToolMessage:
        """
        Fetch and extract content from multiple web pages (max 3).

        Args:
            urls (list[str]): List of URLs to process

        Returns:
            ToolMessage: Combined content from all URLs or error if too many URLs
        """
        if len(urls) > 3:
            WebClient.logger.error(f"Too many URLs: {len(urls)}, maximum is 3")
            return ToolMessage(f"!! [ERROR] Too many URLs: {len(urls)}, maximum is 3")

        results = [WebClient.web_browse(url) for url in urls]
        WebClient.logger.info(f"Processed {len(results)} URLs: {urls}")
        WebClient.logger.debug(f"Browse results: {results}")
        contents = [result.content for result in results]
        return ToolMessage(content="\n\n".join(contents))
