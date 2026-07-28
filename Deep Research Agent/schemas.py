from typing import List

from pydantic import BaseModel, Field

from state import Fact


class ClarifyWithUser(BaseModel):
    """The output of the user clarification decision agent."""
    need_clarification: bool = Field(description="Whether the user needs to be asked a clarifying question.")
    question: str = Field(description="A question to ask the user to clarify the report scope")
    verification: str = Field(description="A verification message to confirm that research will begin.")


class ResearchQuestion(BaseModel):
    """The output of the research brief generation agent."""
    research_brief: str = Field(description="A research question that will be used to guide the research.")


class DraftReport(BaseModel):
    """The output of the initial draft generation agent."""
    draft_report: str = Field(description="A draft report that will be used as the starting point for the diffusion process.")


class Summary(BaseModel):
    """The output of the webpage summarization model."""
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")


class EvaluationResult(BaseModel):
    """The structured output of the programmatic quality evaluator."""
    comprehensiveness_score: int = Field(description="0-10 score on coverage")
    accuracy_score: int = Field(description="0-10 score on factual grounding")
    coherence_score: int = Field(description="0-10 score on flow")
    specific_critique: str = Field(description="Actionable feedback for the researcher")


class FactExtraction(BaseModel):
    """The output of the context pruning / fact extraction agent."""
    new_facts: List[Fact]
