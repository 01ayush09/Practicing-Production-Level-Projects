import asyncio
import json
import sys
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage

from utils import get_message_text


async def run(query: str) -> dict:
    from master_graph import agent

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    if result.get("final_report"):
        return {"final_report": result["final_report"]}

    messages = result.get("messages", [])
    if messages:
        clarifying_text = get_message_text(messages[-1].content)
    else:
        clarifying_text = "The agent did not produce a report or a clarifying question. Please try again."

    return {
        "needs_clarification": True,
        "clarifying_question": clarifying_text,
    }


def main():
    query = sys.stdin.read().strip()
    if not query:
        print(json.dumps({"error": "Empty query received."}))
        sys.exit(1)

    output = asyncio.run(run(query))
    print(json.dumps(output))


if __name__ == "__main__":
    main()
