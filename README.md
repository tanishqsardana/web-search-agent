# web-search-agent

A minimal LangChain agent that answers a question by searching the web. Built
to be a small, predictable test subject for an agent monitoring platform,
not a feature-complete research assistant.

## `testing.py` (PRISM diagnostic)

`testing.py` is a separate, minimal interactive LangGraph agent created to
check whether PRISM tracing was configured and working correctly. It wraps its
graph with PRISM and sends an interactive chat session, so it needs the
OpenRouter, Tavily, and PRISM environment variables configured. It is a
troubleshooting script rather than part of the main agent workflow; at present,
it demonstrates that the PRISM integration is **not working correctly** and
should not be treated as a successful tracing check.

## Architecture

```
question --> create_agent graph --> [web_search tool] --> LLM --> answer
```

One tool (`web_search`, backed by Tavily),
one model, one tool-calling loop, built with LangChain's `create_agent`
(the standard agent builder in LangChain 1.x, itself a thin layer over
LangGraph). No memory, no sub-agents, no vector store. That's the whole
system, on purpose, so a monitoring layer has as little surface to account
for as possible while you get the integration working.

Files:
- `agent.py` — the agent itself (`WebSearchAgent` class + `answer()` helper)
- `main.py` — CLI for running it standalone
- `.env.example` — copy to `.env` and fill in a key
- `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your model key and TAVILY_API_KEY
```

## Usage

```bash
python main.py "What is the current federal funds rate?"
python main.py --json "Who won the last F1 race?"   # full structured output
python main.py                                       # interactive loop
python prism_tool_smoke_test.py                       # deterministic 2 + 2 tracing check
```

Or from code:

```python
from agent import WebSearchAgent

agent = WebSearchAgent()
result = agent.run("What is LangChain?")
# {"question": ..., "answer": ..., "tool_calls": [...], "latency_s": ...}
```

## Integrating with a monitoring platform

`WebSearchAgent.run()` returns a plain dict, not LangChain message objects,
so a monitoring layer can log or ship it without importing LangChain itself:

```python
{
  "question": "...",
  "answer": "...",
  "tool_calls": [{"tool": "web_search", "input": {"query": "..."}, "output": "..."}],
  "latency_s": 1.42,
}
```

A few ways to hook a monitor in, roughly least to most invasive:

1. **Wrap `run()`.** Time it, catch exceptions, ship the returned dict to
   your platform. Nothing inside the agent needs to change.
2. **LangChain callbacks.** Pass a `callbacks=[...]` list into
   `agent.invoke(..., config={"callbacks": [...]})` in `agent.py` for
   step-level events (model calls, tool calls, token usage) as they happen,
   rather than only the final summary.
3. **OpenTelemetry / LangSmith.** Both integrate with LangChain's tracing
   hooks with a few lines of setup and need no changes to `agent.py`.

## Optional: PRISM tracing

`agent.py` can send traces to PRISM, but stays fully untraced until you set
`PRISMTRACE_API_KEY` in your environment or `.env`. Nothing here ever holds
the key value in code, only reads it from the environment.

1. Put your real key, project id, and host in `.env` (not `.env.example`):
   ```
   PRISMTRACE_API_KEY=pt-sk-...
   PRISMTRACE_PROJECT_ID=...
   PRISMTRACE_HOST=https://prism.blockconvey.com
   ```
   Environment variables already exported by your shell take precedence over
   `.env`. Do not point `PRISMTRACE_HOST` at any other domain; rotate the API
   key immediately if it was sent to one.
2. Verify the credential yourself, from your own shell (this repo's code
   never calls this endpoint):
   ```bash
   curl -sS -X POST "$PRISMTRACE_HOST/api/setup-doctor/handshake" \
     -H "Content-Type: application/json" \
     -H "X-PRISMtrace-Key: $PRISMTRACE_API_KEY" \
     -d "{\"project_id\": \"$PRISMTRACE_PROJECT_ID\", \"send_test_trace\": true, \"client\": \"manual\"}"
   ```
3. Run a real question through the agent, e.g. `python main.py "test question"`.
   Because `PRISMTRACE_API_KEY` is set, `agent.py` builds a
   `PRISMtraceLangGraphHandler` once and wraps the compiled LangGraph agent
   with `wrap_langgraph(...)`. Each invocation supplies a LangGraph
   `thread_id`, which PRISM uses to group one conversation. That's a live
   trace, not a handshake.
4. Confirm it landed:
   ```bash
   curl -sS "$PRISMTRACE_HOST/api/setup-doctor?project_id=$PRISMTRACE_PROJECT_ID" \
     -H "X-PRISMtrace-Key: $PRISMTRACE_API_KEY"
   ```
   Look for both `"live_connected": true` and `"app_connected": true`.
5. Call `close_tracing()` from `agent.py` once at process shutdown (not per
   question) to flush and close the handler cleanly.

If you're wiring this into a bigger app rather than running it standalone,
the same lifecycle applies: build the handler once and use a LangGraph
`thread_id` per conversation. LangGraph agents must be wrapped with
`wrap_langgraph(...)`; other LangChain integrations may use their appropriate
callback path.

## Swapping the search tool or model

- Search: `web_search` uses Tavily's fast `basic` search mode. Set
  `TAVILY_API_KEY` in `.env` or your process environment before running the
  agent. Results are compacted to a 3,000-character total so tool output stays
  useful to the model and trace payloads retain the final answer.
- Model: set `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=openai` in `.env`,
  plus the matching model name and API key.
