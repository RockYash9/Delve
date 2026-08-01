"""
Manual dev utility to test /chat/stream without fighting shell quoting
(PowerShell in particular mangles curl's -d JSON argument).

Usage:
    python scripts/test_stream.py "your question here"

Make sure `uvicorn api:app` is running in another terminal first.
"""

import json
import sys

import requests

message = sys.argv[1] if len(sys.argv) > 1 else "what are electric vehicle tax credits in 2026?"

response = requests.post(
    "http://127.0.0.1:8000/chat/stream",
    json={"message": message},
    stream=True,
)

print(f"Status: {response.status_code}\n")

for line in response.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data: "):
        continue
    event = json.loads(line.removeprefix("data: "))

    if event["type"] == "session":
        print(f"[session started: {event['session_id']}]")
    elif event["type"] == "status":
        print(f"\n[{event['text']}]")
    elif event["type"] == "token":
        print(event["text"], end="", flush=True)
    elif event["type"] == "error":
        print(f"\n[ERROR: {event['text']}]")
    elif event["type"] == "done":
        print("\n\n[stream complete]")
