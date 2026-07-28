import operator
from typing import Annotated, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Fact(BaseModel):
    """An atomic unit of knowledge, extracted from raw research notes and stored in the structured knowledge base."""
    content: str = Field(description="The factual statement")
    source_url: str = Field(description="Where this fact came from")
    confidence_score: int = Field(description="1-100 confidence score based on source credibility")
    is_disputed: bool = Field(default=False, description="If this fact conflicts with others")


class Critique(BaseModel):
    """A structured model for adversarial feedback from the Red Team or other quality control agents."""
    author: str
    concern: str
    severity: int
    addressed: bool = Field(default=False, description="Has the supervisor fixed this?")


class QualityMetric(TypedDict):
    """A snapshot of the draft's quality at a specific iteration."""
    score: float
    feedback: str
    iteration: int


class SupervisorState(TypedDict):
    """The hierarchical state for the main Supervisor agent, the central workbench for the diffusion process."""
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    draft_report: str
    raw_notes: Annotated[List[str], operator.add]
    knowledge_base: Annotated[List[Fact], operator.add]
    research_iterations: int
    active_critiques: Annotated[List[Critique], operator.add]
    quality_history: Annotated[List[QualityMetric], operator.add]
    needs_quality_repair: bool


class ResearcherState(TypedDict):
    """The state for a single 'worker' research agent sub-graph."""
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]


class ResearcherOutputState(TypedDict):
    """The output schema exposed by the researcher sub-graph."""
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]


class AgentInputState(MessagesState):
    """The initial input state, which only contains the user's messages."""
    pass


class AgentState(MessagesState):
    """The main state for the full multi-agent system, accumulating all final artifacts."""
    research_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add] = []
    notes: Annotated[list[str], operator.add] = []
    draft_report: str
    final_report: str
