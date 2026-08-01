"""
Web search tool factory, backed by Tavily's search API — with a local
cache layer in front of it (brick 4).

make_web_search_tool() returns a fresh web_search function bound to
its own sources list and status_events list, rather than shared
module-level state. This matters once multiple conversations can run
concurrently (brick 7's API): shared globals would leak one user's
search citations and status updates into another user's session.
"""

import logging
from collections.abc import Callable

from rich.console import Console
from tavily import TavilyClient

import config
from src.storage import cache

console = Console()
logger = logging.getLogger(__name__)


def make_web_search_tool(
    sources: list[dict],
    status_events: list[str] | None = None,
) -> Callable[[str], str]:
    """Build a web_search tool function bound to the given sources list.

    Each Conversation gets its own sources list passed in here, so
    tracking which citations belong to which conversation stays
    correct even when many conversations run at once.

    status_events, if provided, gets a short human-readable message
    appended every time the tool runs (e.g. "searching: ..." or "using
    cached results: ..."). Brick 8's streaming endpoint reads this list
    to surface "the agent is searching now" to the client in real time,
    interleaved with the model's streamed answer text.
    """

    def _record_status(message: str) -> None:
        console.print(f"[dim]  {message}[/dim]")
        if status_events is not None:
            status_events.append(message)

    def web_search(query: str) -> str:
        """Search the live web for current, factual information.

        Use this whenever the user's question needs up-to-date information,
        specific facts, or anything you're not confident about from memory
        alone — news, current events, prices, people, recent releases, etc.

        Args:
            query: A short, specific search query (a few words, like you'd
                type into a search engine — not a full sentence).

        Returns:
            A text summary of the top search results, including titles,
            URLs, and relevant snippets.
        """
        cached_matches = cache.find_similar(query)
        if cached_matches:
            _record_status(f"💾 using cached results for: {query}")
            logger.info("cache_hit query=%r matches=%d", query, len(cached_matches))
            formatted_chunks = []
            for _score, title, url, content in cached_matches:
                sources.append({"title": title, "url": url})
                formatted_chunks.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            return "\n\n---\n\n".join(formatted_chunks)

        _record_status(f"🔍 searching: {query}")
        logger.info("live_search query=%r", query)

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(query=query, max_results=5)

        results = response.get("results", [])
        if not results:
            logger.warning("live_search_no_results query=%r", query)
            return f"No search results found for '{query}'."

        formatted_chunks = []
        for r in results:
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "")

            cache.store_result(query=query, title=title, url=url, content=content)

            sources.append({"title": title, "url": url})
            formatted_chunks.append(f"Title: {title}\nURL: {url}\nContent: {content}")

        return "\n\n---\n\n".join(formatted_chunks)

    return web_search
