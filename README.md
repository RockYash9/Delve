# Delve

An agentic, conversational research/search engine. Gemini decides when and
how to search the web, iterates on its own queries when results aren't
enough, and remembers context across a session.

Built brick by brick — this repo will grow in stages:

- [x] **Brick 1** — environment + a single working call to Gemini
- [x] **Brick 2** — web search tool + agent decides when to call it
- [x] **Brick 3** — multi-turn conversation memory (type 'reset' to start fresh)
- [ ] **Brick 4** — local caching + embedding-based retrieval
- [ ] **Brick 5** — CLI polish + exportable research reports

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