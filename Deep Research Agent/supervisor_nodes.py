import asyncio
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command

from config import MAX_CONCURRENT_RESEARCHERS, MAX_RESEARCHER_ITERATIONS
from model_bindings import supervisor_model_with_tools
from prompts import lead_researcher_with_multiple_steps_diffusion_double_check_prompt
from researcher_graph import researcher_agent
from self_correction_nodes import evaluate_draft_quality
from state import QualityMetric, SupervisorState
from tools import refine_draft_report, think_tool
from utils import ainvoke_with_retry, get_notes_from_tool_calls, get_today_str, invoke_with_retry


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """
    The 'Brain' of the diffusion process. This node analyzes the current state,
    including any critical feedback, and decides on the next set of actions (tool calls).
    """
    supervisor_messages = state.get("supervisor_messages", [])

    system_message = lead_researcher_with_multiple_steps_diffusion_double_check_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCHERS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    critiques = state.get("active_critiques", [])
    unaddressed = [c for c in critiques if not c.addressed]
    if unaddressed:
        critique_text = "\n".join([f"- {c.author} says: {c.concern}" for c in unaddressed])
        intervention = SystemMessage(content=f"""
        CRITICAL INTERVENTION REQUIRED.
        The following issues were detected by the Adversarial Team in your draft:
        {critique_text}

        You MUST address these issues in your next step.
        If the critique says citations are missing, call 'ConductResearch' to find them.
        If the critique says logic is flawed, call 'think_tool' to plan a fix.
        """)
        messages.append(intervention)

    if state.get("needs_quality_repair"):
        messages.append(SystemMessage(content="PREVIOUS DRAFT QUALITY WAS LOW (Score < 7/10). Focus on finding new sources and citing them."))

    try:
        response = await ainvoke_with_retry(supervisor_model_with_tools.ainvoke, messages, max_retries=3)
    except Exception as e:
        print(f"[supervisor] Giving up after retries due to a provider error, finalizing with current draft: {e}")
        response = AIMessage(content="Unable to plan further research due to a repeated provider error. Finalizing with the current draft.")

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
            "needs_quality_repair": False,
        },
    )


async def supervisor_tools(state: SupervisorState) -> Command[Literal["red_team", "context_pruner", "__end__"]]:
    """
    The 'Hands' of the Supervisor. This node executes the planned tool calls, including
    fanning out to parallel research sub-graphs and running the denoising step.
    """
    most_recent_message = state.get("supervisor_messages", [])[-1]

    exceeded_iterations = state.get("research_iterations", 0) >= MAX_RESEARCHER_ITERATIONS
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls)

    if exceeded_iterations or no_tool_calls or research_complete:
        kb_notes = [f"{f.content} (Confidence: {f.confidence_score})" for f in state.get("knowledge_base", [])]
        if not kb_notes:
            kb_notes = get_notes_from_tool_calls(state.get("supervisor_messages", []))

        return Command(goto=END, update={"notes": kb_notes, "research_brief": state.get("research_brief", "")})

    conduct_research_calls = [t for t in most_recent_message.tool_calls if t["name"] == "ConductResearch"]
    refine_report_calls = [t for t in most_recent_message.tool_calls if t["name"] == "refine_draft_report"]
    think_calls = [t for t in most_recent_message.tool_calls if t["name"] == "think_tool"]

    tool_messages = []
    all_raw_notes = []
    draft_report = state.get("draft_report", "")
    updates = {}

    for tool_call in think_calls:
        observation = think_tool.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=observation, name="think_tool", tool_call_id=tool_call["id"]))

    if conduct_research_calls:
        coros = [
            researcher_agent.ainvoke({
                "researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])],
                "research_topic": tc["args"]["research_topic"],
            }) for tc in conduct_research_calls
        ]

        results = await asyncio.gather(*coros)
        for result, tool_call in zip(results, conduct_research_calls):
            tool_messages.append(ToolMessage(content=result.get("compressed_research", ""), name=tool_call["name"], tool_call_id=tool_call["id"]))
            all_raw_notes.extend(result.get("raw_notes", []))

    for tool_call in refine_report_calls:
        kb = state.get("knowledge_base", [])
        kb_str = "CONFIRMED FACTS:\n" + "\n".join([f"- {f.content}" for f in kb]) if kb else "\n".join(get_notes_from_tool_calls(state.get("supervisor_messages", [])))
        new_draft = invoke_with_retry(refine_draft_report.invoke, {"research_brief": state.get("research_brief", ""), "findings": kb_str, "draft_report": state.get("draft_report", "")})

        eval_result = invoke_with_retry(evaluate_draft_quality, research_brief=state.get("research_brief", ""), draft_report=new_draft)
        avg_score = (eval_result.comprehensiveness_score + eval_result.accuracy_score) / 2

        tool_messages.append(ToolMessage(content=f"Draft Updated.\nQuality Score: {avg_score}/10.\nJudge Feedback: {eval_result.specific_critique}", name=tool_call["name"], tool_call_id=tool_call["id"]))
        draft_report = new_draft

        updates["quality_history"] = [QualityMetric(score=avg_score, feedback=eval_result.specific_critique, iteration=state.get("research_iterations", 0))]
        if avg_score < 7.0:
            updates["needs_quality_repair"] = True

    updates["supervisor_messages"] = tool_messages
    updates["raw_notes"] = all_raw_notes
    updates["draft_report"] = draft_report

    return Command(goto=["red_team", "context_pruner"], update=updates)
