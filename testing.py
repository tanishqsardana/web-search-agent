"""Smallest runnable LangGraph agent with an LLM and PRISM tracing."""

import os

import prismtrace
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from prismtrace import PRISMtraceLangGraphHandler, wrap_langgraph
from tavily import TavilyClient

load_dotenv()


def get_model():
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.2:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


@tool
def web_search(query: str) -> str:
    """Search the web with Tavily for the supplied text query."""
    response = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(
        query=query,
        search_depth="basic",
        max_results=3,
    )
    results = response.get("results", [])
    if not results:
        return "No results found."

    return "\n".join(
        f"{result['title']}: {result['content']} ({result['url']})"
        for result in results
    )


tools = [web_search]
model = get_model().bind_tools(tools)


def chat(state: MessagesState) -> dict:
    """Reply using all messages from this interactive conversation."""
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chat")
builder.add_conditional_edges("chat", tools_condition)
builder.add_edge("tools", "chat")
agent = builder.compile()

handler = PRISMtraceLangGraphHandler(
    api_key=os.environ["PRISMTRACE_API_KEY"],
    project_id=os.environ["PRISMTRACE_PROJECT_ID"],
    host=os.environ["PRISMTRACE_HOST"],
    agent_name="minimal-llm-agent",
)
graph = wrap_langgraph(agent, handler)

try:
    messages = []
    print("Chat ready. Type 'quit' or 'exit' to stop.")
    with prismtrace.session("minimal-chat-session"):
        while True:
            prompt = input("You > ").strip()
            if prompt.lower() in {"quit", "exit"}:
                break
            if not prompt:
                continue

            result = graph.invoke({"messages": [*messages, ("user", prompt)]})
            messages = result["messages"]
            print(f"Assistant > {messages[-1].content}")
finally:
    handler.close()
