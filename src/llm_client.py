"""Azure OpenAI routing for judge / pair-gen calls. See docs/AZURE.md.

Never used for the subject model (Qwen) — that runs in-process via
transformers on the GPU VM (src/agent_loop.py), not through this client.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
)

logger = logging.getLogger(__name__)

def _find_env_path() -> Path:
    """Search upward from this file for the nearest .env, instead of a
    hardcoded parents[N] hop count. Laptop deployment nests this repo at
    ai_notes/app/shortcut_forensics/src/ (.env 3 levels up); the GPU box
    only gets shortcut_forensics/ rsynced (bootstrap.sh copies the minimal
    secrets .env to REMOTE_DIR/.env, one level up from src/) -- a fixed
    hop count silently resolves to the wrong path (e.g. /home/.env) on
    whichever layout it wasn't written for."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        p = candidate / ".env"
        if p.exists():
            return p
    return here.parent / ".env"  # fallback: repo root guess, load_dotenv no-ops if missing


_ENV_PATH = _find_env_path()

_client: OpenAI | None = None
_model: str | None = None


def _load_env() -> None:
    load_dotenv(_ENV_PATH)


def get_client() -> tuple[OpenAI, str]:
    """Return (client, model_name) per docs/AZURE.md routing rule.

    Prefer Azure OpenAI when OPENAI_PREFER_AZURE + the three AZURE_OPENAI_*
    vars are set; else fall back to direct OpenAI; else raise.
    """
    global _client, _model
    if _client is not None and _model is not None:
        return _client, _model

    _load_env()
    prefer_azure = os.getenv("OPENAI_PREFER_AZURE", "true").lower() in ("1", "true", "yes")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_ep = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    azure_deploy = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if prefer_azure and azure_key and azure_ep and azure_deploy:
        _client = OpenAI(api_key=azure_key, base_url=f"{azure_ep}/openai/v1/")
        _model = azure_deploy
        logger.info("llm_client: routed to Azure OpenAI deployment=%s", azure_deploy)
    elif os.getenv("OPENAI_API_KEY"):
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        _model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.warning("llm_client: Azure unset/incomplete, falling back to direct OpenAI")
    else:
        raise RuntimeError(
            "No usable chat credentials: set AZURE_OPENAI_{API_KEY,ENDPOINT,DEPLOYMENT} "
            f"or OPENAI_API_KEY in {_ENV_PATH}"
        )
    return _client, _model


class TransientLLMError(Exception):
    """429 / timeout / connection error — safe to retry."""


# Retry window for 429/connection errors. Default 10 min per the SPEC failure table. Rollout workers
# set SCFX_LLM_RETRY_S low (their thread is a generation slot; a lost verdict is recovered by
# scripts/rejudge_phase.py); the rejudge pass keeps the long window.
_RETRY_S = int(os.environ.get("SCFX_LLM_RETRY_S", "600"))


@retry(
    retry=retry_if_exception_type(TransientLLMError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_delay(_RETRY_S),
    reraise=True,
)
def chat(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    response_format: dict | None = None,
) -> str:
    """One chat call against the routed client. Raises TransientLLMError on
    429/connection errors so callers get tenacity backoff; other errors
    propagate immediately (caller decides retry-once-then-skip per SPEC)."""
    client, model = get_client()
    kwargs: dict = dict(model=model, messages=messages, temperature=temperature, max_completion_tokens=max_tokens)
    if response_format is not None:
        kwargs["response_format"] = response_format
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # openai SDK raises typed errors; route 429/conn to retry
        cls_name = type(e).__name__
        if "RateLimit" in cls_name or "APIConnection" in cls_name or "Timeout" in cls_name:
            raise TransientLLMError(str(e)) from e
        # newer Azure deployments (reasoning models) reject max_tokens/temperature;
        # retry once with the minimal, widely-compatible kwarg set before giving up.
        if "unsupported_parameter" in str(e) or "Unsupported parameter" in str(e):
            fallback_kwargs = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
            resp = client.chat.completions.create(**fallback_kwargs)
            return resp.choices[0].message.content or ""
        raise
    return resp.choices[0].message.content or ""


def ping() -> str:
    return chat([{"role": "user", "content": "ping"}], max_tokens=10)
