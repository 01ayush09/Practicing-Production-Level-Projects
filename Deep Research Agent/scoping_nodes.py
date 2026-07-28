from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.graph import END
from langgraph.types import Command

from config import RESEARCH_MODE, creative_model, model
from prompts import (
    clarify_with_user_instructions,
    draft_report_generation_prompt,
    transform_messages_into_research_topic_human_msg_prompt,
)
from schemas import ClarifyWithUser
from state import AgentState
from utils import get_message_text, get_today_str, invoke_with_retry


def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", END]]:
    """
    This node acts as a gatekeeper. It determines if the user's request has enough detail to proceed.
    If not, it HALTS the graph and asks a clarifying question. If yes, it proceeds to the next step.
    """
    messages_text = get_buffer_string(state["messages"])
    current_date = get_today_str()

    structured_output_model = model.with_structured_output(ClarifyWithUser)
    response = invoke_with_retry(
        structured_output_model.invoke,
        [HumanMessage(content=clarify_with_user_instructions.format(
            messages=messages_text,
            date=current_date,
        ))],
    )

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]},
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


def write_research_brief(state: AgentState) -> Command[Literal["write_draft_report"]]:
    """
    This node transforms the confirmed conversation history into a single, comprehensive research brief.
    Uses plain text generation (not JSON structured output) since the schema is a single free-text
    field, and wrapping long free text in JSON only adds failure risk for no structural benefit.
    """
    response = invoke_with_retry(
        model.invoke,
        [HumanMessage(content=transform_messages_into_research_topic_human_msg_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str(),
        ))],
    )
    research_brief = get_message_text(response.content).strip()

    return Command(
        goto="write_draft_report",
        update={"research_brief": research_brief},
    )


def write_draft_report(state: AgentState) -> dict:
    """
    This node takes the research brief and generates an initial, unresearched draft.
    This serves as the "noisy" starting point for our diffusion process. Uses plain text
    generation (not JSON structured output) for the same reason as write_research_brief -
    a single free-text field gains nothing from JSON wrapping but adds truncation/escaping risk.
    """
    research_brief = state.get("research_brief", "")

    draft_report_prompt_formatted = draft_report_generation_prompt.format(
        research_brief=research_brief,
        date=get_today_str(),
    )

    if RESEARCH_MODE == "speed":
        draft_report_prompt_formatted += "\n\nIMPORTANT: Keep this initial draft concise - aim for under 300 words. It will be expanded later with research findings."
    else:
        draft_report_prompt_formatted += "\n\nIMPORTANT: This is only the initial rough draft, not the final report. Keep it to roughly 400-600 words covering the main sections at a high level - it will be substantially expanded and refined with research findings in later steps."

    response = invoke_with_retry(
        creative_model.invoke,
        [HumanMessage(content=draft_report_prompt_formatted)],
    )
    draft_report = get_message_text(response.content).strip()

    return {
        "research_brief": research_brief,
        "draft_report": draft_report,
        "supervisor_messages": ["Here is the draft report: " + draft_report, research_brief],
    }
