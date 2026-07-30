"""
Web search tool, backed by Tavily's search API.

This function is handed directly to Gemini as a "tool" — the model reads
its name, docstring, and type hints to decide when it's relevant, and
google-genai's automatic function calling handles invoking it and feeding
the result back into the model's reasoning. We don't manually manage that
loop; the SDK does it.
"""

from tavily import TavilyClient
from rich.console import Console

import config

console = Console()


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
    console.print(f"[dim]  🔍 searching: {query}[/dim]")

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(query=query, max_results=5)

    results = response.get("results", [])
    if not results:
        return f"No search results found for '{query}'."

    formatted_chunks = []
    for r in results:
        formatted_chunks.append(
            f"Title: {r.get('title', 'Untitled')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content: {r.get('content', '')}"
        )

    return "\n\n---\n\n".join(formatted_chunks)