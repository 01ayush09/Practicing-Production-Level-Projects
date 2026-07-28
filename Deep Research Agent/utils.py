import asyncio
import re
import time
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import BaseMessage, filter_messages

from config import LLM_CALL_THROTTLE_SECONDS


def get_today_str() -> str:
    now = datetime.now()
    return f"{now.strftime('%a %b')} {now.day}, {now.strftime('%Y')}"


def get_message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def get_notes_from_tool_calls(messages: List[BaseMessage]) -> List[str]:
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


def _parse_wait_seconds(msg: str) -> Optional[float]:
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg, re.IGNORECASE)
    if not match:
        return None
    try:
        minutes = float(match.group(1)) if match.group(1) else 0.0
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    except ValueError:
        return None


def _is_rate_limit_error(msg: str) -> bool:
    lowered = msg.lower()
    return "rate_limit" in lowered or "rate limit" in lowered or "429" in msg or "too many requests" in lowered


def _get_backoff_seconds(exc: Exception, attempt: int = 0, default: float = 5.0, max_wait: float = 30.0) -> Optional[float]:
    msg = str(exc)
    wait = _parse_wait_seconds(msg)
    if wait is not None:
        if wait > max_wait:
            return None
        return wait + 0.5
    if _is_rate_limit_error(msg):
        # No explicit wait time given (e.g. Mistral's generic 429) - back off progressively
        # longer on each attempt instead of repeating the same short wait into the same wall.
        return default * (attempt + 1)
    return 0.0


def invoke_with_retry(fn, *args, max_retries: int = 3, **kwargs):
    last_exc = None
    for attempt in range(max_retries + 1):
        if LLM_CALL_THROTTLE_SECONDS:
            time.sleep(LLM_CALL_THROTTLE_SECONDS)
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            print(f"[retry] LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                wait = _get_backoff_seconds(e, attempt=attempt)
                if wait is None:
                    print("[retry] Required wait time is too long to retry automatically (likely a daily quota limit) - giving up.")
                    break
                if wait > 0:
                    print(f"[retry] Rate limited - waiting {wait:.1f}s before retrying...")
                    time.sleep(wait)
    raise last_exc


async def ainvoke_with_retry(fn, *args, max_retries: int = 3, **kwargs):
    last_exc = None
    for attempt in range(max_retries + 1):
        if LLM_CALL_THROTTLE_SECONDS:
            await asyncio.sleep(LLM_CALL_THROTTLE_SECONDS)
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            print(f"[retry] LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                wait = _get_backoff_seconds(e, attempt=attempt)
                if wait is None:
                    print("[retry] Required wait time is too long to retry automatically (likely a daily quota limit) - giving up.")
                    break
                if wait > 0:
                    print(f"[retry] Rate limited - waiting {wait:.1f}s before retrying...")
                    await asyncio.sleep(wait)
    raise last_exc
