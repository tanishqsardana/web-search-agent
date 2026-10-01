"""Deterministic LangGraph + PRISM smoke test.

Run:
    python prism_tool_smoke_test.py

It forces one add_numbers(2, 2) tool call, so it verifies LangGraph tool
callbacks without relying on an LLM provider or an external search service.
"""

import json
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from agent import _get_prism_handler, close_tracing


class CalculatorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def add_numbers(left: int, right: int) -> int:
    """Add two integer values."""
    return left + right


def request_addition(_: CalculatorState) -> dict:
    """Create a deterministic tool call equivalent to asking for 2 + 2."""
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "add_numbers",
                        "args": {"left": 2, "right": 2},
                        "id": "add-2-and-2",
                    }
                ],
            )
        ]
    }


def respond(state: CalculatorState) -> dict:
    """Turn the tool result into a deterministic final graph message."""
    tool_result = next(
        message for message in reversed(state["messages"]) if isinstance(message, ToolMessage)
    )
    return {"messages": [AIMessage(content=f"2 + 2 = {tool_result.content}")]}


def build_graph():
    graph = StateGraph(CalculatorState)
    graph.add_node("request_addition", request_addition)
    graph.add_node("tools", ToolNode([add_numbers]))
    graph.add_node("respond", respond)
    graph.add_edge(START, "request_addition")
    graph.add_edge("request_addition", "tools")
    graph.add_edge("tools", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def main() -> None:
    load_dotenv()
    graph = build_graph()
    handler = _get_prism_handler()
    if handler is not None:
        from prismtrace import wrap_langgraph

        graph = wrap_langgraph(graph, handler)

    thread_id = f"prism-tool-smoke-{uuid.uuid4()}"
    try:
        result = graph.invoke(
            {"messages": []},
            config={"configurable": {"thread_id": thread_id}},
        )
    finally:
        close_tracing()

    print(
        json.dumps(
            {
                "session_id": thread_id,
                "answer": result["messages"][-1].content,
                "tool_calls": [
                    {
                        "tool": "add_numbers",
                        "input": {"left": 2, "right": 2},
                        "output": 4,
                    }
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
