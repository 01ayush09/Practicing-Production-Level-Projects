import json
import os
import subprocess
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Deep Research Agent API")


class ResearchRequest(BaseModel):
    query: str
    llm_provider: str = "groq"
    research_mode: Optional[str] = None
    groq_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    tavily_api_key: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research")
def research(payload: ResearchRequest):
    if not payload.query.strip():
        raise HTTPException(400, "query cannot be empty")

    env = os.environ.copy()
    env["LLM_PROVIDER"] = payload.llm_provider
    env["TAVILY_API_KEY"] = payload.tavily_api_key
    if payload.research_mode:
        env["RESEARCH_MODE"] = payload.research_mode

    if payload.llm_provider == "groq":
        if not payload.groq_api_key:
            raise HTTPException(400, "groq_api_key is required for provider 'groq'")
        env["GROQ_API_KEY"] = payload.groq_api_key
    elif payload.llm_provider == "mistral":
        if not payload.mistral_api_key:
            raise HTTPException(400, "mistral_api_key is required for provider 'mistral'")
        env["MISTRAL_API_KEY"] = payload.mistral_api_key
    else:
        raise HTTPException(400, f"Unsupported llm_provider '{payload.llm_provider}'")

    try:
        result = subprocess.run(
            [sys.executable, "worker.py"],
            input=payload.query,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Research run timed out after 10 minutes")

    if result.returncode != 0:
        raise HTTPException(500, f"Research run failed: {result.stderr[-2000:]}")

    output_line = result.stdout.strip().splitlines()[-1]
    return json.loads(output_line)
