"""
Minimal web-search Q&A agent, built with LangChain (v1 `create_agent` API).

Architecture (deliberately flat, one path in and out):

    question --> create_agent graph --> [web_search tool] --> LLM --> answer

There is exactly one tool (web search) and one model in a single tool-calling
loop. No memory, no multi-agent handoff, no vector store. The point is to have
a small, predictable unit that an external monitoring platform can wrap,
trace, and evaluate without fighting a complex agent graph.

run() returns a plain dict rather than LangChain message objects, so a
monitoring layer can log/serialize it without knowing LangChain internals.
"""

import os
import time
import uuid
from typing import Any, Dict, Optional

import prismtrace
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from tavily import TavilyClient
from prismtrace import PRISMtraceLangGraphHandler

# Keep the graph-state trace compact: LangGraph serializes tool output inside
# several parent spans, and oversized results can push the final answer beyond
# a tracing backend's span-output limit.
MAX_TAVILY_SNIPPET_CHARS = 500
MAX_TAVILY_TOOL_OUTPUT_CHARS = 3000


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with Tavily and return the top results (title, snippet, url)."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Search is unavailable: set TAVILY_API_KEY to enable Tavily search."

    response = TavilyClient(api_key=api_key).search(
        query,
        search_depth="basic",
        max_results=max_results,
    )
    results = response.get("results", [])
    if not results:
        return "No results found."

    formatted_results = []
    total_length = 0
    for result in results:
        title = " ".join(str(result.get("title", "Untitled")).split())[:200]
        snippet = " ".join(str(result.get("content", "")).split())[
            :MAX_TAVILY_SNIPPET_CHARS
        ]
        url = str(result.get("url", "")).strip()
        entry = f"- {title}: {snippet} ({url})"
        separator_length = 1 if formatted_results else 0

        # Preserve whole, source-attributed results rather than cutting a URL
        # halfway through to fit the tracing payload budget.
        if total_length + separator_length + len(entry) > MAX_TAVILY_TOOL_OUTPUT_CHARS:
            break
        formatted_results.append(entry)
        total_length += separator_length + len(entry)

    return "\n".join(formatted_results) or "No results found."


SYSTEM_PROMPT = """You are a research assistant that answers questions using web search.

Rules:
- Use the search tool at least once before answering, unless the question needs no external facts (e.g. simple math).
- Base your answer only on what the search results actually say. Do not invent facts.
- End your answer with the source URL(s) you relied on.
- If the results are inconclusive or conflicting, say so plainly instead of guessing.
- Keep answers concise and directly responsive to the question.
"""


def _get_model():
    """Resolves the model for create_agent from env vars.

    Returns either a 'provider:model' string (for providers init_chat_model
    knows natively) or a constructed chat model instance (for an
    OpenAI-compatible endpoint like OpenRouter that it doesn't).
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        return f"anthropic:{os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-6')}"
    elif provider == "openai":
        return f"openai:{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}"
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.2:free"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider!r}. Use 'anthropic', 'openai', or 'openrouter'."
        )


_prism_handler = None
_PRISMTRACE_HOST = "https://prism.blockconvey.com"


def _get_prism_handler():
    global _prism_handler
    if _prism_handler is not None:
        return _prism_handler

    api_key = os.getenv("PRISMTRACE_API_KEY")
    project_id = os.getenv("PRISMTRACE_PROJECT_ID")
    # Tracing is optional: an incomplete PRISM configuration must never stop
    # the search agent itself from running.
    if not api_key or not project_id:
        return None

    host = os.getenv("PRISMTRACE_HOST", _PRISMTRACE_HOST).strip().rstrip("/")
    if host != _PRISMTRACE_HOST:
        raise ValueError(
            "Refusing to send PRISMTRACE_API_KEY to an untrusted host. "
            f"Set PRISMTRACE_HOST={_PRISMTRACE_HOST}."
        )

    
    print('here')
    _prism_handler = PRISMtraceLangGraphHandler(
        api_key=api_key,
        project_id=project_id,
        host=host,
        agent_name=os.getenv("PRISMTRACE_AGENT_NAME", "web-search-agent"),
    )
    return _prism_handler


def close_tracing() -> None:
    """Flush and close the PRISM handler once at process shutdown."""
    global _prism_handler
    if _prism_handler is not None:
        _prism_handler.close()
        _prism_handler = None


class WebSearchAgent:
    """Thin LangGraph wrapper with one explicit chat node."""

    def __init__(self):
        # This is the inner LangChain agent: it owns the model <-> tool loop.
        # The outer graph deliberately has one named ``chat`` node so the
        # LangGraph structure is explicit in both the application and traces.
        chat_agent = create_agent(
            model=_get_model(),
            tools=[web_search],
            system_prompt=SYSTEM_PROMPT,
        )

        def chat_node(state: MessagesState, config: RunnableConfig) -> dict:
            """Run the model/tool loop and append only its new messages."""
            messages = state["messages"]
            result = chat_agent.invoke({"messages": messages}, config=config)
            return {"messages": result["messages"][len(messages) :]}

        builder = StateGraph(MessagesState)
        builder.add_node("chat", chat_node)
        builder.add_edge(START, "chat")
        builder.add_edge("chat", END)
        graph = builder.compile()

        handler = _get_prism_handler()
        if handler is not None:
            from prismtrace import wrap_langgraph

            graph = wrap_langgraph(graph, handler)

        # ``graph`` is the compiled graph from the documented PRISM example.
        # ``agent`` is retained as a compatibility alias for existing callers.
        self.graph = graph
        self.agent = graph

    def run(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer one question.

        session_id groups this call into a PRISM trajectory when tracing is
        configured. Pass the same id for turns in one conversation; omit it
        and a fresh id is generated per call (fine for one-off questions).

        Returns a dict shaped for a monitoring platform to ingest directly:
            {
              "question": str,
              "answer": str,
              "tool_calls": [{"tool": str, "input": dict, "output": str}, ...],
              "latency_s": float,
            }
        Raises whatever the underlying LangChain call raises; callers/monitoring
        wrappers decide how to log and handle failures.
        """
        start = time.time()
        trace_session_id = session_id or str(uuid.uuid4())
        input_state = {"messages": [("user", question)]}

        # This is the same integration shape as the PRISM LangGraph example:
        # create one handler, wrap one compiled graph, then invoke inside a
        # session context. The ambient session groups all callback spans.
        if _prism_handler is not None:
            with prismtrace.session(trace_session_id):
                result = self.graph.invoke(input_state)
        else:
            result = self.graph.invoke(input_state)
        latency = time.time() - start

        # Walk the full message trajectory and pair each tool call with its result
        # by tool_call_id, so this stays correct even with multiple/parallel calls.
        calls_by_id: Dict[str, Dict[str, Any]] = {}
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls_by_id[tc["id"]] = {"tool": tc["name"], "input": tc["args"], "output": None}
            elif isinstance(msg, ToolMessage) and msg.tool_call_id in calls_by_id:
                calls_by_id[msg.tool_call_id]["output"] = str(msg.content)
        final_content = result["messages"][-1].content
        answer = final_content.strip() if isinstance(final_content, str) else str(final_content).strip()
        rounded_latency = round(latency, 3)
        return {
            "question": question,
            "answer": answer,
            "tool_calls": list(calls_by_id.values()),
            "latency_s": rounded_latency,
        }


def answer(question: str) -> str:
    """Convenience one-liner for scripts that just want the text back."""
    return WebSearchAgent().run(question)["answer"]
