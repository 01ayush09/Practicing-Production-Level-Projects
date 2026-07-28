import time
from typing import List, Literal

from langchain_core.messages import HumanMessage
from tavily import TavilyClient

from config import MAX_SUMMARIZE_INPUT_CHARS, SUMMARIZE_THROTTLE_SECONDS, summarization_model
from prompts import summarize_webpage_prompt
from schemas import Summary
from utils import get_today_str

tavily_client = TavilyClient()


def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> List[dict]:
    print(f"--- [TOOL] Executing Tavily search for queries: {search_queries} ---")
    search_docs = []
    for query in search_queries:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        search_docs.append(result)
    return search_docs


def summarize_webpage_content(webpage_content: str) -> str:
    try:
        structured_model = summarization_model.with_structured_output(Summary)
        summary_result = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content,
                date=get_today_str(),
            ))
        ])
        formatted_summary = (
            f"<summary>\n{summary_result.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary_result.key_excerpts}\n</key_excerpts>"
        )
        return formatted_summary
    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content


def deduplicate_search_results(search_results: List[dict]) -> dict:
    unique_results = {}
    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = result
    return unique_results


def process_search_results(unique_results: dict) -> dict:
    summarized_results = {}
    for url, result in unique_results.items():
        if result.get("raw_content"):
            content = summarize_webpage_content(result["raw_content"][:MAX_SUMMARIZE_INPUT_CHARS])
            if SUMMARIZE_THROTTLE_SECONDS:
                time.sleep(SUMMARIZE_THROTTLE_SECONDS)
        else:
            content = result["content"]
        summarized_results[url] = {"title": result["title"], "content": content}
    return summarized_results


def format_search_output(summarized_results: dict) -> str:
    if not summarized_results:
        return "No valid search results found."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"
    return formatted_output
