"""
Web search tool, backed by Tavily's search API — with a local cache
layer in front of it (brick 4).

Every result Delve fetches gets embedded and stored locally. Before
spending a real Tavily API call, we check whether something
semantically similar has already been fetched, and reuse it if so.
This means repeated or related questions get faster and don't burn
through the free-tier search quota, and the tool's own knowledge base
grows the more it's used.
"""

from tavily import TavilyClient
from rich.console import Console

import config
from src.storage import cache

console = Console()

# Every source seen this session (cache hit or live search), for later
# export as a citations appendix. Deduplicated by URL at export time.
SESSION_SOURCES: list[dict] = []


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
        console.print(f"[dim]  💾 using cached results for: {query}[/dim]")
        formatted_chunks = []
        for _score, title, url, content in cached_matches:
            SESSION_SOURCES.append({"title": title, "url": url})
            formatted_chunks.append(f"Title: {title}\nURL: {url}\nContent: {content}")
        return "\n\n---\n\n".join(formatted_chunks)

    console.print(f"[dim]  🔍 searching: {query}[/dim]")

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(query=query, max_results=5)

    results = response.get("results", [])
    if not results:
        return f"No search results found for '{query}'."

    formatted_chunks = []
    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")

        cache.store_result(query=query, title=title, url=url, content=content)

        SESSION_SOURCES.append({"title": title, "url": url})
        formatted_chunks.append(f"Title: {title}\nURL: {url}\nContent: {content}")

    return "\n\n---\n\n".join(formatted_chunks)