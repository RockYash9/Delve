# Delve

An agentic, conversational research/search engine. Claude decides when and
how to search the web, iterates on its own queries when results aren't
enough, and remembers context across a session.

Built brick by brick — this repo will grow in stages:

- [x] **Brick 1** — environment + a single working call to Claude
- [ ] **Brick 2** — web search tool + agent decides when to call it
- [ ] **Brick 3** — multi-turn conversation memory
- [ ] **Brick 4** — local caching + embedding-based retrieval
- [ ] **Brick 5** — CLI polish + exportable research reports

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
cp .env.example .env
# then edit .env and paste in your ANTHROPIC_API_KEY

# 4. Run it
python main.py
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
