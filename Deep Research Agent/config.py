import os

from langchain.chat_models import init_chat_model

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

DEFAULT_MODE = "speed" if LLM_PROVIDER == "groq" else "depth"
RESEARCH_MODE = os.environ.get("RESEARCH_MODE", DEFAULT_MODE)

if RESEARCH_MODE == "speed":
    MAX_CONCURRENT_RESEARCHERS = 1
    MAX_RESEARCHER_ITERATIONS = 2
    MAX_SUMMARIZE_INPUT_CHARS = 1500
    TAVILY_MAX_RESULTS = 1
    WRITER_MAX_TOKENS = 1500
    RUN_RED_TEAM = False
    RUN_CONTEXT_PRUNER = False
elif RESEARCH_MODE == "depth":
    MAX_CONCURRENT_RESEARCHERS = 1
    MAX_RESEARCHER_ITERATIONS = 2      # was 3
    MAX_SUMMARIZE_INPUT_CHARS = 3500   # was 6000
    TAVILY_MAX_RESULTS = 2             # was 3
    WRITER_MAX_TOKENS = 2000           # was 3000
    RUN_RED_TEAM = True
    RUN_CONTEXT_PRUNER = True
else:
    raise ValueError(f"Unsupported RESEARCH_MODE '{RESEARCH_MODE}'. Use 'speed' or 'depth'.")

# Both free-tier providers have tight per-second/per-minute limits, so every LLM call
# (not just summarization) gets a small pre-call pace-limiter to avoid bursts - see
# LLM_CALL_THROTTLE_SECONDS usage in utils.py's retry helpers.
SUMMARIZE_THROTTLE_SECONDS = 1.5 if LLM_PROVIDER == "groq" else (1.3 if LLM_PROVIDER == "mistral" else 0)
LLM_CALL_THROTTLE_SECONDS = 1.2 if LLM_PROVIDER == "groq" else (1.0 if LLM_PROVIDER == "mistral" else 0)

if LLM_PROVIDER == "groq":
    SMART_MODEL = "groq:llama-3.3-70b-versatile"
    FAST_MODEL = "groq:llama-3.3-70b-versatile"
elif LLM_PROVIDER == "mistral":
    SMART_MODEL = "mistralai:mistral-large-latest"
    FAST_MODEL = "mistralai:mistral-small-latest"
else:
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' (needs GROQ_API_KEY) "
        f"or 'mistral' (needs MISTRAL_API_KEY)."
    )

model = init_chat_model(model=SMART_MODEL)
creative_model = init_chat_model(model=SMART_MODEL, max_tokens=max(WRITER_MAX_TOKENS, 2500))
summarization_model = init_chat_model(model=FAST_MODEL)
compress_model = init_chat_model(model=SMART_MODEL, max_tokens=max(WRITER_MAX_TOKENS, 3000))
critic_model = init_chat_model(model=SMART_MODEL)
judge_model = init_chat_model(model=SMART_MODEL)
compressor_model = init_chat_model(model=FAST_MODEL)
writer_model = init_chat_model(model=SMART_MODEL, max_tokens=WRITER_MAX_TOKENS)
baseline_model = init_chat_model(model=SMART_MODEL)
