"""
CLI for the web-search agent.

Usage:
    python main.py "What is the current federal funds rate?"
    python main.py --json "Who won the last F1 race?"
    python main.py               # interactive loop
"""

import argparse
import json
import uuid

from dotenv import load_dotenv


from agent import WebSearchAgent, close_tracing
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Ask the web-search agent a question.")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit for an interactive loop.")
    parser.add_argument("--json", action="store_true", help="Print the full structured result, not just the answer.")
    parser.add_argument(
        "--session-id",
        help="Optional PRISM conversation ID; generated automatically when omitted.",
    )
    args = parser.parse_args()

    agent = WebSearchAgent()

    def emit(result: dict):
        print(json.dumps(result, indent=2) if args.json else result["answer"])

    if args.question:
        # WebSearchAgent.run passes this to LangGraph as configurable.thread_id,
        # which is the session identifier the PRISM LangGraph wrapper uses.
        emit(
            agent.run(
                " ".join(args.question),
                session_id=args.session_id or str(uuid.uuid4()),
            )
        )
        close_tracing()
        return

    print("Web search agent ready. Type a question (or 'quit').")
    # Keep all turns in this interactive conversation under one trajectory.
    session_id = args.session_id or str(uuid.uuid4())
    while True:
        q = input("> ").strip()
        if q.lower() in ("quit", "exit"):
            close_tracing()
            break
        if not q:
            continue
        emit(agent.run(q, session_id=session_id))


if __name__ == "__main__":
    main()
