from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages

from config import compress_model
from model_bindings import model_with_tools, tools_by_name
from prompts import compress_research_human_message, compress_research_system_prompt, research_agent_prompt
from state import ResearcherState
from utils import get_message_text, get_today_str, invoke_with_retry


def llm_call(state: ResearcherState):
    """The 'brain' of the researcher: analyzes the current state and decides on the next action (call a tool or finish)."""
    try:
        response = invoke_with_retry(
            model_with_tools.invoke,
            [SystemMessage(content=research_agent_prompt.format(date=get_today_str()))] + state["researcher_messages"],
            max_retries=3,
        )
    except Exception as e:
        print(f"[llm_call] Giving up after retries due to a provider error, ending this research step early: {e}")
        response = AIMessage(content="Unable to continue searching due to a repeated provider error. Finishing with information gathered so far.")

    return {"researcher_messages": [response]}


def tool_node(state: ResearcherState):
    """The 'hands' of the researcher: executes all tool calls from the previous LLM response."""
    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    tool_outputs = [
        ToolMessage(
            content=str(observation),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """A conditional edge that determines whether to continue the ReAct loop or finish the research sub-task."""
    messages = state["researcher_messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return "compress_research"


def compress_research(state: ResearcherState) -> dict:
    """The final node in the research sub-graph: it compresses all findings from the ReAct loop into a clean, cited summary."""
    system_message = compress_research_system_prompt.format(date=get_today_str())

    messages = (
        [SystemMessage(content=system_message)]
        + state.get("researcher_messages", [])
        + [HumanMessage(content=compress_research_human_message.format(research_topic=state["research_topic"]))]
    )

    response = invoke_with_retry(compress_model.invoke, messages)

    raw_notes = [
        get_message_text(m.content) for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"],
        )
    ]

    return {
        "compressed_research": get_message_text(response.content),
        "raw_notes": ["\n".join(raw_notes)],
    }
