import asyncio
import getpass
import os
import uuid

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

complex_query = """
Conduct a deep analysis of the 'Splinternet' phenomenon's impact on global semiconductor supply chains by 2028.
Specifically contrast TSMC's diversification strategy against Intel's IDM 2.0 model under 2024-2025 US export controls,
and predict the resulting shift in insurance liability models for cross-border wafer shipments.
"""


def ensure_env_var(var_name: str, prompt_label: str) -> str:
    value = os.environ.get(var_name)
    if value:
        return value
    value = getpass.getpass(f"{prompt_label}: ").strip()
    if not value:
        raise SystemExit(f"{var_name} is required to run this project.")
    os.environ[var_name] = value
    return value


def collect_user_keys():
    provider = os.environ.get("LLM_PROVIDER")
    if not provider:
        provider = input("Choose your LLM provider - 'groq' or 'mistral' [groq]: ").strip() or "groq"
    os.environ["LLM_PROVIDER"] = provider

    if provider == "groq":
        ensure_env_var("GROQ_API_KEY", "Enter your Groq API key (get one free at https://console.groq.com/keys)")
        default_mode = "speed"
    elif provider == "mistral":
        ensure_env_var("MISTRAL_API_KEY", "Enter your Mistral API key (get one free at https://console.mistral.ai/api-keys)")
        default_mode = "depth"
    else:
        raise SystemExit(f"Unsupported LLM_PROVIDER '{provider}'. Use 'groq' or 'mistral'.")

    ensure_env_var("TAVILY_API_KEY", "Enter your Tavily API key (get one free at https://app.tavily.com)")

    mode = os.environ.get("RESEARCH_MODE")
    if not mode:
        mode = input(f"Choose research mode - 'speed' or 'depth' [{default_mode}]: ").strip() or default_mode
    os.environ["RESEARCH_MODE"] = mode

    print(f"\nUsing provider '{provider}' in '{mode}' mode. All requests run against your own personal free-tier quota.\n")


async def run_research(agent, human_message_cls, query: str):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = await agent.ainvoke(
        {"messages": [human_message_cls(content=query)]},
        config=config,
    )
    return result


def main():
    collect_user_keys()

    from langchain_core.messages import HumanMessage
    from rich.console import Console
    from rich.markdown import Markdown
    from master_graph import agent
    from utils import get_message_text

    console = Console()

    user_query = input("\nEnter your research question (press Enter to use the built-in demo question): ").strip()
    query = user_query or complex_query

    result = asyncio.run(run_research(agent, HumanMessage, query))

    if result.get("final_report"):
        console.print(Markdown(result["final_report"]))
    else:
        messages = result.get("messages", [])
        console.print("\nThe agent needs more detail before it can research this. It asked:\n")
        if messages:
            console.print(get_message_text(messages[-1].content))
        else:
            console.print("(No report or clarifying question was produced. Please try again.)")
        console.print("\nTip: include the extra detail directly in your question and rerun.")


if __name__ == "__main__":
    main()
