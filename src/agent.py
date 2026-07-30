"""
The core agent loop.

Brick 1 (today): a bare single-turn call to Claude — no tools, no memory.
This exists purely to prove the environment and API key work end to end.

Brick 2 (next): add the web_search tool and let Claude decide when to call it.
Brick 3: multi-turn conversation history.
Brick 4: local caching + retrieval.
"""

import anthropic
import config


def ask(user_message: str) -> str:
    """Send a single message to Claude and return the text response."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=config.MODEL_NAME,
        max_tokens=config.MAX_TOKENS,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    # response.content is a list of blocks (text, tool_use, etc).
    # For brick 1 we only expect plain text blocks.
    text_parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_parts)
