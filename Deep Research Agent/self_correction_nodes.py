from langchain_core.messages import HumanMessage, SystemMessage

from config import RUN_CONTEXT_PRUNER, RUN_RED_TEAM, compressor_model, critic_model, judge_model
from schemas import EvaluationResult, FactExtraction
from state import Critique, SupervisorState
from utils import ainvoke_with_retry, get_message_text


async def red_team_node(state: SupervisorState) -> dict:
    """
    This node represents the 'Red Team' agent. It runs in parallel to other steps,
    critiquing the current draft to find logical flaws and biases. Skipped entirely
    in Speed Mode to keep latency low.
    """
    if not RUN_RED_TEAM:
        return {}

    draft = state.get("draft_report", "")

    if not draft or len(draft) < 50:
        return {}

    prompt = f"""
    You are the 'Red Team' Adversary.
    The researcher has written the following draft report.

    <Draft>
    {draft}
    </Draft>

    Your goal is NOT to be helpful. Your goal is to find:
    1. Claims that lack citations or are not supported by the evidence.
    2. Logical leaps where the conclusion does not follow from the premises.
    3. Significant bias or a failure to consider alternative viewpoints.

    If the draft is solid and has no major logical or factual issues, output exactly "PASS".
    If there are issues, output a specific, harsh, and actionable critique describing the errors.
    """

    response = await ainvoke_with_retry(critic_model.ainvoke, [HumanMessage(content=prompt)])
    content = get_message_text(response.content)

    if "PASS" in content and len(content) < 20:
        return {}

    critique = Critique(
        author="Red Team Adversary",
        concern=content,
        severity=8,
        addressed=False,
    )

    return {
        "active_critiques": [critique],
        "supervisor_messages": [
            SystemMessage(content=f"ADVERSARIAL FEEDBACK DETECTED: {content}")
        ],
    }


def evaluate_draft_quality(research_brief: str, draft_report: str) -> EvaluationResult:
    """
    This function implements the 'Self-Evolution' scoring mechanism. It acts as an
    LLM-as-a-judge, programmatically evaluating the quality of a draft against the original brief.
    """
    eval_prompt = f"""
    You are a Senior Research Editor. Your standards are exceptionally high. Evaluate this draft report against the research brief.

    <Research Brief>
    {research_brief}
    </Research Brief>

    <Draft Report>
    {draft_report}
    </Draft Report>

    Be extremely critical. High scores (8+) should be reserved for truly excellent, comprehensive, and well-cited work.
    Focus your evaluation on these key areas:
    1. **Comprehensiveness:** Does the draft fully address all parts of the research brief? Are there significant gaps?
    2. **Accuracy & Grounding:** Are the claims specific and well-supported? Look for vague statements that need citations.
    3. **Coherence & Structure:** Is the report well-organized and easy to follow? Is the language clear and professional?

    Provide specific, actionable critique for the researcher.
    """

    structured_judge = judge_model.with_structured_output(EvaluationResult)
    return structured_judge.invoke([HumanMessage(content=eval_prompt)])


async def context_pruning_node(state: SupervisorState) -> dict:
    """
    This node implements 'Context Engineering'. It takes the temporary buffer of raw notes,
    extracts structured facts, adds them to the permanent knowledge base, and then clears the buffer.
    Skipped entirely in Speed Mode to keep latency low.
    """
    if not RUN_CONTEXT_PRUNER:
        return {}

    raw_notes = state.get("raw_notes", [])

    if not raw_notes:
        return {}

    text_block = "\n".join(raw_notes)

    prompt = f"""
    You are a Knowledge Graph Engineer.

    New Raw Notes from a research agent:
    {text_block[:20000]}

    Your task is to:
    1. Extract all atomic, verifiable facts from the New Raw Notes.
    2. For each fact, identify its source URL.
    3. Assign a confidence score (1-100) based on the credibility of the source.
    4. Ignore any information that is the agent's internal "thinking" or planning.

    Return ONLY a valid JSON object with a single key 'new_facts' containing a list of these structured facts.
    """

    try:
        structured_llm = compressor_model.with_structured_output(FactExtraction)
        result = await ainvoke_with_retry(structured_llm.ainvoke, [HumanMessage(content=prompt)])
        new_facts = result.new_facts
        message = f"[SYSTEM] Context Pruned. {len(new_facts)} new facts added to Knowledge Base. Raw notes buffer cleared."
    except Exception as e:
        new_facts = []
        message = f"[SYSTEM] Context Pruning failed: {str(e)}"

    return {
        "raw_notes": [],
        "knowledge_base": new_facts,
        "supervisor_messages": [
            SystemMessage(content=message)
        ],
    }
