# Deployment Guide

## Speed Mode vs Depth Mode
This project runs in one of two tuned configurations, controlled by `RESEARCH_MODE`:

- **Speed Mode** (`RESEARCH_MODE=speed`, pairs with `LLM_PROVIDER=groq`) - target latency ~15-20s.
  Single researcher, single search, no Red Team / Context Pruner, short final report.
  Best for live demos and quick sanity checks.
- **Depth Mode** (`RESEARCH_MODE=depth`, pairs with `LLM_PROVIDER=mistral`) - target latency ~60-90s.
  Multiple parallel researchers, multiple supervisor iterations, full Red Team + Evaluator +
  Context Pruner self-correction loop, longer final report.
  Best for genuinely thorough research output.

Ready-made presets: copy `.env.speed-mode.example` or `.env.depth-mode.example` to `.env` and
fill in your keys. If `RESEARCH_MODE` isn't set, it defaults automatically based on `LLM_PROVIDER`
(groq -> speed, mistral -> depth), but you can mix and match if you want.

Note: these are tuned *targets* based on each provider's per-call throughput, not a hard guarantee -
actual latency still depends on your network, the live API's response time, and how long a real
web search takes. `main.py`/`api.py`/`streamlit_app.py` all have generous timeouts as a safety net
in case a run runs long.

Three ways to run this, from simplest to most "production":

## 1. Local CLI (already covered)
```
pip install -r requirements.txt
python main.py
```

## 2. Local web UI (Streamlit) - recommended for sharing with non-technical people
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Opens a browser page where each visitor pastes in their own free API keys and question.
Nothing is stored server-side; keys only live for the duration of one subprocess run.

## 3. Hosted deployment

### UI (Streamlit Community Cloud - free)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io, connect your GitHub account.
3. Point it at this repo, set the main file to `streamlit_app.py`.
4. Deploy. No secrets needed in the platform's secrets manager, since users type in their own keys at runtime.

### API (FastAPI) - Render/Railway (both have free tiers)
1. Push this repo to GitHub.
2. Create a new "Web Service" on Render (or Railway).
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Do NOT set GROQ_API_KEY/MISTRAL_API_KEY/TAVILY_API_KEY as platform secrets - callers send
   their own keys in the request body, so the server never holds anyone's key.

Example request once deployed:
```
curl -X POST https://your-app.onrender.com/research \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Compare TSMC and Intel chip strategy for 2026-2028",
        "llm_provider": "mistral",
        "mistral_api_key": "YOUR_OWN_KEY",
        "tavily_api_key": "YOUR_OWN_KEY"
      }'
```

## Why isolated subprocesses?
Each request/run spawns a fresh `worker.py` process with only that caller's keys in its
environment. This guarantees one user's API key and quota can never leak into or be reused by
another user's request, even if many people use the same hosted UI/API at the same time.

## Notes
- Free-tier rate limits still apply per key - if a user's own key is rate-limited, only their
  run fails, not anyone else's.
- Long research runs (multiple iterations) can take minutes; the FastAPI endpoint has a
  10-minute timeout and the Streamlit UI a 15-minute timeout - adjust if needed.
