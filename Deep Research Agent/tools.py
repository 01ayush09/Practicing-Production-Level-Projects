from typing import Annotated, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from config import TAVILY_MAX_RESULTS, writer_model
from prompts import report_generation_with_draft_insight_prompt
from search_utils import (
    deduplicate_search_results,
    format_search_output,
    process_search_results,
    tavily_search_multiple,
)
from utils import get_message_text, get_today_str


@tool
class ConductResearch(BaseModel):
    """A tool for delegating a specific research task to a specialized sub-agent. The Supervisor uses this to fan out work."""
    research_topic: str = Field(
        description="The topic to research. Should be a single, self-contained topic described in high detail.",
    )


@tool
class ResearchComplete(BaseModel):
    """A tool for the Supervisor to signal that the research process is complete and the final report can be generated."""
    pass


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps.

    Returns:
        Confirmation that reflection was recorded for decision-making.
    """
    return f"Reflection recorded: {reflection}"


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = TAVILY_MAX_RESULTS,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """A tool that fetches results from the Tavily search API and performs content summarization.

    Args:
        query: A single, specific search query to execute.
        max_results: The maximum number of results to return.
        topic: The topic to filter results by ('general', 'news', 'finance').

    Returns:
        A formatted string of the deduplicated and summarized search results.
    """
    search_results = tavily_search_multiple([query], max_results=max_results, topic=topic, include_raw_content=True)
    unique_results = deduplicate_search_results(search_results)
    summarized_results = process_search_results(unique_results)
    return format_search_output(summarized_results)


@tool(parse_docstring=True)
def refine_draft_report(
    research_brief: Annotated[str, InjectedToolArg],
    findings: Annotated[str, InjectedToolArg],
    draft_report: Annotated[str, InjectedToolArg],
):
    """Refine draft report

    Synthesizes all research findings into a comprehensive draft report

    Args:
        research_brief: user's research request
        findings: collected research findings for the user request
        draft_report: draft report based on the findings and user request

    Returns:
        refined draft report
    """
    draft_report_prompt = report_generation_with_draft_insight_prompt.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str(),
    )
    draft_report_response = writer_model.invoke([HumanMessage(content=draft_report_prompt)])
    return get_message_text(draft_report_response.content)
