from langgraph.graph import START, StateGraph

from self_correction_nodes import context_pruning_node, red_team_node
from state import SupervisorState
from supervisor_nodes import supervisor, supervisor_tools

supervisor_builder = StateGraph(SupervisorState)

supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("red_team", red_team_node)
supervisor_builder.add_node("context_pruner", context_pruning_node)

supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_edge("supervisor", "supervisor_tools")
supervisor_builder.add_edge("red_team", "supervisor")
supervisor_builder.add_edge("context_pruner", "supervisor")

supervisor_agent = supervisor_builder.compile()
