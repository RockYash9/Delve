"""
Export the current conversation into a shareable markdown research
report — the transcript plus a deduplicated citations/sources
appendix drawn from every search performed during the session.

This is what turns a chat session into an actual deliverable document,
rather than something that only exists inside the terminal.
"""

from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports"


def build_report(transcript: list[tuple[str, str]], sources: list[dict]) -> str:
    """Render a transcript + sources into a markdown document."""
    lines = [
        "# Delve Research Report",
        f"_Generated {datetime.now():%B %d, %Y at %I:%M %p}_",
        "",
    ]

    for role, text in transcript:
        speaker = "You" if role == "user" else "Delve"
        lines.append(f"### {speaker}")
        lines.append(text)
        lines.append("")

    if sources:
        lines.append("---")
        lines.append("## Sources")
        seen_urls = set()
        for source in sources:
            url = source["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            lines.append(f"- [{source['title']}]({url})")

    return "\n".join(lines)


def save_report(content: str) -> Path:
    """Write the report to reports/, timestamped, and return its path."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = f"delve_report_{datetime.now():%Y%m%d_%H%M%S}.md"
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path
