from langchain_core.messages import HumanMessage

from config import writer_model
from prompts import final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt
from state import AgentState
from utils import ainvoke_with_retry, get_message_text, get_today_str


async def final_report_generation(state: AgentState):
    """
    The final node in our master graph. It takes all the curated artifacts from the
    Supervisor loop and generates the final, polished report. Falls back to the best
    available draft if the provider call fails persistently (e.g. an exhausted daily
    quota), rather than crashing the whole run.
    """
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
    )

    try:
        final_report = await ainvoke_with_retry(writer_model.ainvoke, [HumanMessage(content=final_report_prompt)])
        report_text = get_message_text(final_report.content)
    except Exception as e:
        print(f"[final_report_generation] Giving up after retries due to a provider error: {e}")
        draft = state.get("draft_report", "")
        report_text = (
            (draft or "No report could be generated due to a repeated provider error.")
            + "\n\n---\n*Note: the final polishing pass could not be completed because of a "
              "provider error (e.g. a rate limit or exhausted quota). This is the best draft "
              "report available; try again later or switch provider.*"
        )

    return {
        "final_report": report_text,
        "messages": ["Here is the final report: " + report_text],
    }
