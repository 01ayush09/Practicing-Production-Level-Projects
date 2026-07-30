import asyncio
import csv
import importlib
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.messages import HumanMessage

# Note: judge_model, invoke_with_retry, and get_message_text are deliberately
# NOT imported at the top level here. If they were, they'd freeze to whichever
# LLM_PROVIDER happened to be active when this file was first imported, and
# every later load_stack() call in the loop below would silently keep using
# that original provider's throttle/backoff settings even after switching.
# See load_stack() and judge_report() below - both fetch these fresh from
# sys.modules so they always reflect whichever provider is currently loaded.

EVAL_QUESTIONS = [
    "Suggest me some of major studies and techniques analyzing European energy systems investigate the economic viability of exporting alternative green molecules (e-fuels, e-methanol, and hydrogen) rather than just electricity, while examining cross-border trade models?",
]

# Run each question this many times per config. >1 lets you measure
# self-consistency (how much a single run's scores wobble run-to-run, on
# the SAME config/question) instead of over-reading a single n=1 result.
REPEATS_PER_QUESTION = 1

JUDGE_PROMPT = """You are a strict Senior Research Editor. Score this report on a 0-10 scale for each dimension.

<Question>
{question}
</Question>

<Report>
{report}
</Report>

Score:
- comprehensiveness_score (0-10): does it fully address the question with real depth?
- accuracy_score (0-10): are claims specific, plausible, and free of obvious factual errors?
- coherence_score (0-10): is it well-organized and easy to follow?

Be strict. A generic, shallow, or unsupported-claims-heavy answer should score low (2-4), not high.
"""

# The judge model's provider is fixed and explicit here, rather than being
# whatever LLM_PROVIDER happens to be set in .env when this script starts.
# This judge only ever scores YOUR agent's own report - it is not a second
# model being compared against your agent, just a fixed grading instrument
# so scores from different sessions stay comparable.
JUDGE_PROVIDER = "mistral"

# Each entry is one agent configuration to evaluate - NOT a different model
# to compare against. Both entries run through the exact same agent
# architecture (master_graph); only the LLM_PROVIDER/RESEARCH_MODE knobs
# differ, matching how you actually run this in production.
AGENT_RUNS = [
    {"label": "groq_speed", "provider": "groq", "mode": "speed"},
    {"label": "mistral_depth", "provider": "mistral", "mode": "depth"},
]

# config.py bakes LLM_PROVIDER/RESEARCH_MODE-derived constants (which models
# get built, iteration limits, model max_tokens, RUN_RED_TEAM /
# RUN_CONTEXT_PRUNER flags, ...) into module-level values, and every other
# module in the project imports those values by name at import time (e.g.
# `from config import RUN_RED_TEAM`). That means just changing the env vars
# mid-process does nothing on its own - the modules already hold their old,
# imported copies. To actually switch provider/mode within a single script
# run, we drop the whole dependency chain from sys.modules and re-import it
# fresh so it picks up the new env vars.
AGENT_MODULES_TO_RELOAD = [
    "config",
    "utils",
    "prompts",
    "schemas",
    "state",
    "search_utils",
    "tools",
    "model_bindings",
    "researcher_nodes",
    "researcher_graph",
    "self_correction_nodes",
    "supervisor_nodes",
    "supervisor_graph",
    "scoping_nodes",
    "scoping_graph",
    "final_report",
    "master_graph",
]


def load_stack(provider: str, mode: Optional[str]):
    """Set LLM_PROVIDER (and RESEARCH_MODE, if given) then force a fresh
    import of config, and - if a mode was given - the whole agent module
    chain, so everything actually gets built for this provider/mode combo.

    Returns (config_module, agent_or_None, utils_module).
    """
    os.environ["LLM_PROVIDER"] = provider
    if mode is not None:
        os.environ["RESEARCH_MODE"] = mode
    else:
        os.environ.pop("RESEARCH_MODE", None)

    for mod_name in AGENT_MODULES_TO_RELOAD:
        sys.modules.pop(mod_name, None)

    config_module = importlib.import_module("config")
    utils_module = importlib.import_module("utils")

    agent = None
    if mode is not None:
        master_graph = importlib.import_module("master_graph")
        agent = master_graph.agent

    return config_module, agent, utils_module


@dataclass
class RunResult:
    question: str
    mode: str
    repeat_index: int
    success: bool
    latency_seconds: float = 0.0
    report_text: str = ""
    error: str = ""

    # LLM-judge scores (grading YOUR agent's own report, not vs. another model)
    comprehensiveness: Optional[int] = None
    accuracy: Optional[int] = None
    coherence: Optional[int] = None

    # Iteration efficiency - pulled straight from SupervisorState
    research_iterations: Optional[int] = None
    stopped_early: Optional[bool] = None  # True if it finished before hitting the max

    # Quality trend across the supervisor's own refine loop (quality_history)
    quality_score_first: Optional[float] = None
    quality_score_last: Optional[float] = None
    quality_score_delta: Optional[float] = None  # last - first; did the draft actually improve?

    # Knowledge base / groundedness proxy (knowledge_base: List[Fact])
    num_facts: Optional[int] = None
    avg_fact_confidence: Optional[float] = None
    num_disputed_facts: Optional[int] = None

    # Red-team catch rate (active_critiques: List[Critique])
    num_critiques_raised: Optional[int] = None
    num_critiques_addressed: Optional[int] = None


def judge_report(question: str, report: str, judge_model, utils_module) -> dict:
    from schemas import EvaluationResult

    structured_judge = judge_model.with_structured_output(EvaluationResult)
    prompt = JUDGE_PROMPT.format(question=question, report=report)
    result = utils_module.invoke_with_retry(structured_judge.invoke, [HumanMessage(content=prompt)])
    return {
        "comprehensiveness": result.comprehensiveness_score,
        "accuracy": result.accuracy_score,
        "coherence": result.coherence_score,
    }


def extract_agent_metrics(result: dict, max_iterations: int) -> dict:
    """Pull the agent's own process/quality signals out of the final graph
    state - no extra LLM calls needed, this data is already computed by the
    agent's existing nodes (self_correction_nodes.py, supervisor_nodes.py).
    """
    quality_history = result.get("quality_history", []) or []
    knowledge_base = result.get("knowledge_base", []) or []
    critiques = result.get("active_critiques", []) or []
    research_iterations = result.get("research_iterations")

    metrics = {
        "research_iterations": research_iterations,
        "stopped_early": (research_iterations is not None and research_iterations < max_iterations),
        "quality_score_first": None,
        "quality_score_last": None,
        "quality_score_delta": None,
        "num_facts": len(knowledge_base),
        "avg_fact_confidence": None,
        "num_disputed_facts": sum(1 for f in knowledge_base if getattr(f, "is_disputed", False)),
        "num_critiques_raised": len(critiques),
        "num_critiques_addressed": sum(1 for c in critiques if getattr(c, "addressed", False)),
    }

    if quality_history:
        scores = [qm["score"] for qm in quality_history]
        metrics["quality_score_first"] = scores[0]
        metrics["quality_score_last"] = scores[-1]
        metrics["quality_score_delta"] = round(scores[-1] - scores[0], 2)

    if knowledge_base:
        confidences = [getattr(f, "confidence_score", None) for f in knowledge_base]
        confidences = [c for c in confidences if c is not None]
        if confidences:
            metrics["avg_fact_confidence"] = round(statistics.mean(confidences), 1)

    return metrics


async def run_agent(question: str, agent, mode: str, repeat_index: int, judge_model, utils_module, max_iterations: int) -> RunResult:
    start = time.time()
    try:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # ainvoke() only returns fields declared on the top-level AgentState
        # schema. research_iterations/knowledge_base/quality_history/
        # active_critiques all live on SupervisorState instead - the state of
        # the "supervisor_subgraph" node (see supervisor_graph.py) - and never
        # bubble up through a plain ainvoke() call. astream(..., subgraphs=True)
        # yields (namespace, state) for every subgraph too, so we grab the
        # supervisor subgraph's own last state (its namespace is non-empty)
        # separately from the top-level state (namespace == ()).
        final_agent_state = None
        final_supervisor_state = None
        async for namespace, chunk in agent.astream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="values",
            subgraphs=True,
        ):
            if namespace == ():
                final_agent_state = chunk
            else:
                final_supervisor_state = chunk

        elapsed = time.time() - start
        result = final_agent_state or {}
        supervisor_state = final_supervisor_state or {}

        agent_metrics = extract_agent_metrics(supervisor_state, max_iterations)

        if not result.get("final_report"):
            return RunResult(
                question=question, mode=mode, repeat_index=repeat_index, success=False,
                latency_seconds=elapsed, error="No final_report (agent may have asked for clarification)",
                **agent_metrics,
            )

        report = result["final_report"]
        scores = judge_report(question, report, judge_model, utils_module)
        return RunResult(
            question=question, mode=mode, repeat_index=repeat_index, success=True, latency_seconds=elapsed,
            report_text=report, comprehensiveness=scores["comprehensiveness"],
            accuracy=scores["accuracy"], coherence=scores["coherence"],
            **agent_metrics,
        )
    except Exception as e:
        elapsed = time.time() - start
        return RunResult(question=question, mode=mode, repeat_index=repeat_index, success=False, latency_seconds=elapsed, error=str(e))


def summarize(results: List[RunResult], mode: str) -> dict:
    subset = [r for r in results if r.mode == mode]
    successes = [r for r in subset if r.success]
    n = len(subset)
    success_rate = (len(successes) / n * 100) if n else 0.0

    def avg(field_name):
        vals = [getattr(r, field_name) for r in successes if getattr(r, field_name) is not None]
        return round(statistics.mean(vals), 2) if vals else None

    def stdev(field_name):
        vals = [getattr(r, field_name) for r in successes if getattr(r, field_name) is not None]
        return round(statistics.stdev(vals), 2) if len(vals) > 1 else None

    latencies = [r.latency_seconds for r in successes]
    early_stop_count = sum(1 for r in successes if r.stopped_early)

    return {
        "mode": mode,
        "runs": n,
        "success_rate_pct": round(success_rate, 1),
        "avg_latency_sec": round(statistics.mean(latencies), 2) if latencies else None,
        "median_latency_sec": round(statistics.median(latencies), 2) if latencies else None,

        # Quality (judge) - mean AND stdev, since stdev tells you how much a
        # single run's score can be trusted (only meaningful when
        # REPEATS_PER_QUESTION > 1)
        "avg_comprehensiveness": avg("comprehensiveness"),
        "avg_accuracy": avg("accuracy"),
        "avg_coherence": avg("coherence"),
        "stdev_comprehensiveness": stdev("comprehensiveness"),
        "stdev_accuracy": stdev("accuracy"),
        "stdev_coherence": stdev("coherence"),

        # Iteration efficiency
        "avg_research_iterations": avg("research_iterations"),
        "stopped_early_count": early_stop_count,
        "stopped_early_pct": round(early_stop_count / len(successes) * 100, 1) if successes else None,

        # Quality trend across the refine loop
        "avg_quality_score_delta": avg("quality_score_delta"),

        # Groundedness proxy
        "avg_num_facts": avg("num_facts"),
        "avg_fact_confidence": avg("avg_fact_confidence"),
        "avg_num_disputed_facts": avg("num_disputed_facts"),

        # Red-team catch rate
        "avg_critiques_raised": avg("num_critiques_raised"),
        "avg_critiques_addressed": avg("num_critiques_addressed"),
    }


async def main():
    all_results: List[RunResult] = []

    print(f"Loading judge model (fixed at provider='{JUDGE_PROVIDER}' so scores stay comparable across sessions)...")
    judge_config_module, _, _ = load_stack(JUDGE_PROVIDER, None)
    judge_model = judge_config_module.judge_model

    for entry in AGENT_RUNS:
        label, provider, mode = entry["label"], entry["provider"], entry["mode"]
        print(f"\nLoading agent stack for '{label}' (provider={provider}, mode={mode})...")
        config_module, agent, utils_module = load_stack(provider, mode)
        max_iterations = config_module.MAX_RESEARCHER_ITERATIONS

        print(f"Running {len(EVAL_QUESTIONS)} question(s) x {REPEATS_PER_QUESTION} repeat(s) through '{label}'...")
        for q in EVAL_QUESTIONS:
            for rep in range(REPEATS_PER_QUESTION):
                tag = f"{label}" + (f" (repeat {rep+1}/{REPEATS_PER_QUESTION})" if REPEATS_PER_QUESTION > 1 else "")
                print(f"  {tag}: {q[:60]}...")
                r = await run_agent(q, agent, label, rep, judge_model, utils_module, max_iterations)
                all_results.append(r)
                status = "OK" if r.success else f"FAILED ({r.error[:80]})"
                print(f"    -> {status} in {r.latency_seconds:.1f}s"
                      + (f" | iterations={r.research_iterations} facts={r.num_facts} critiques={r.num_critiques_raised}" if r.success else ""))

    mode_order = [entry["label"] for entry in AGENT_RUNS]
    summaries = {mode: summarize(all_results, mode) for mode in mode_order}

    print("\n" + "=" * 60)
    print("SUMMARY (agent-only metrics - no external model comparison)")
    print("=" * 60)
    for s in summaries.values():
        print(json.dumps(s, indent=2))

    with open("eval_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question", "mode", "repeat_index", "success", "latency_seconds",
            "comprehensiveness", "accuracy", "coherence",
            "research_iterations", "stopped_early",
            "quality_score_first", "quality_score_last", "quality_score_delta",
            "num_facts", "avg_fact_confidence", "num_disputed_facts",
            "num_critiques_raised", "num_critiques_addressed",
            "error",
        ])
        for r in all_results:
            writer.writerow([
                r.question, r.mode, r.repeat_index, r.success, round(r.latency_seconds, 2),
                r.comprehensiveness, r.accuracy, r.coherence,
                r.research_iterations, r.stopped_early,
                r.quality_score_first, r.quality_score_last, r.quality_score_delta,
                r.num_facts, r.avg_fact_confidence, r.num_disputed_facts,
                r.num_critiques_raised, r.num_critiques_addressed,
                r.error,
            ])

    with open("eval_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("\nWrote eval_results.csv and eval_summary.json")


if __name__ == "__main__":
    asyncio.run(main())
