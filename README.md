# Delve

An agentic, conversational research/search engine. Gemini decides when and
how to search the web, iterates on its own queries when results aren't
enough, and remembers context across a session.

Built brick by brick — this repo will grow in stages:

- [x] **Brick 1** — environment + a single working call to Gemini
- [x] **Brick 2** — web search tool + agent decides when to call it
- [x] **Brick 3** — multi-turn conversation memory (type 'reset' to start fresh)
- [x] **Brick 4** — local caching + semantic retrieval (the agent reuses past searches)
- [x] **Brick 5** — CLI polish + exportable research reports

## Commands (inside the running app)

- `help` — list available commands
- `reset` — start a fresh conversation (clears memory, keeps the knowledge cache)
- `export` — save the current conversation as a markdown report with a sources appendix, into `reports/`
- `exit` / `quit` — leave

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API keys
cp .env.example .env
# then edit .env and paste in:
#   GEMINI_API_KEY  — free, no card, from https://aistudio.google.com
#   TAVILY_API_KEY  — free (1,000 searches/month), no card, from https://tavily.com

# 4. Run it
python main.py
```

**First run only:** the very first search will download the local
embedding model (~90MB, one-time, cached afterward) — this happens
automatically but takes a moment, and needs an internet connection
even though the model runs locally after that.

A `delve_cache.db` file will appear in the project folder as you use
it — that's your growing local knowledge base. It's already gitignored.

## Experimenting

`playground.ipynb` is a scratch notebook for testing new ideas (a new tool,
a raw API response, etc.) before writing them into `src/`. It's not part of
the actual product — that's `main.py`. Launch it with:

```bash
jupyter notebook playground.ipynb
```

## Project structure

```
delve/
├── main.py              # entry point
├── config.py            # env vars & settings, loaded once
├── src/
│   ├── cli.py           # interactive REPL
│   ├── agent.py         # core Claude call / agent loop
│   ├── tools/           # search & other tools Claude can call
│   ├── memory/          # conversation history management
│   └── storage/         # local cache / vector index (later brick)
```