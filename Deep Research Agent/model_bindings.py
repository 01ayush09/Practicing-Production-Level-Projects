from config import LLM_PROVIDER, model
from tools import ConductResearch, ResearchComplete, refine_draft_report, tavily_search, think_tool

_bind_kwargs = {"parallel_tool_calls": False} if LLM_PROVIDER == "groq" else {}

model_with_tools = model.bind_tools([tavily_search, think_tool], **_bind_kwargs)
tools_by_name = {"tavily_search": tavily_search, "think_tool": think_tool}

supervisor_tools_list = [ConductResearch, ResearchComplete, refine_draft_report, think_tool]
supervisor_model_with_tools = model.bind_tools(supervisor_tools_list, **_bind_kwargs)
