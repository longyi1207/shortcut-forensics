"""Azure OpenAI judge calls: rollout labeling, plant checks, contrast-pair
gen/filter. SPEC.md §4.2. Templates in prompts/. JSON parse: retry once,
then skip + log (never silently swallow — caller decides what "skip" means).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src import llm_client
from src.concepts import CONCEPTS

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"

_TEMPLATE_CACHE: dict[str, str] = {}


def _load_template(name: str) -> str:
    if name not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[name] = (PROMPTS_DIR / name).read_text()
    return _TEMPLATE_CACHE[name]


def _fill(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class JudgeParseError(Exception):
    pass


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from a possibly-fenced / chatty LLM response."""
    candidates = []
    m = _JSON_FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    # Also try the first {...} span, in case of leading/trailing prose.
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
    raise JudgeParseError(f"could not parse JSON from response: {text[:300]!r}")


def _call_judge_json(
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 500,
    log_path: Path | None = None,
    log_context: dict | None = None,
) -> dict | None:
    """Call the judge, parse JSON, retry once on parse failure, else skip+log.
    Returns None on double failure (caller must handle: skip this item)."""
    for attempt in range(2):
        try:
            raw = llm_client.chat([{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
            return parse_json_response(raw)
        except JudgeParseError as e:
            logger.warning("judge JSON parse failed (attempt %d/2): %s", attempt + 1, e)
            if attempt == 1:
                if log_path is not None:
                    from src.clock import append_jsonl

                    append_jsonl(
                        log_path,
                        {"error": "judge_parse_fail", "detail": str(e), "context": log_context or {}},
                    )
                return None
        except Exception as e:
            logger.error("judge call failed: %s", e)
            if log_path is not None:
                from src.clock import append_jsonl

                append_jsonl(
                    log_path,
                    {"error": "judge_call_fail", "detail": str(e), "context": log_context or {}},
                )
            return None
    return None


def judge_rollout(transcript_text: str, diffs_text: str, *, errors_log: Path | None = None, rollout_id: str = "") -> dict | None:
    template = _load_template("judge_rollout.md")
    prompt = _fill(template, transcript=transcript_text, diffs=diffs_text or "(no diff captured)")
    return _call_judge_json(prompt, max_tokens=500, log_path=errors_log, log_context={"rollout_id": rollout_id, "kind": "judge_rollout"})


def judge_plant(transcript_text: str, concept: str, *, errors_log: Path | None = None, rollout_id: str = "") -> dict | None:
    c = CONCEPTS[concept]
    template = _load_template("judge_plant.md")
    prompt = _fill(template, transcript=transcript_text, concept=concept, concept_definition=c.definition)
    return _call_judge_json(prompt, max_tokens=200, log_path=errors_log, log_context={"rollout_id": rollout_id, "kind": "judge_plant", "concept": concept})


_PAIR_OBJ_RE = re.compile(r'\{\s*"plus"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"minus"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}')


def _salvage_pairs(text: str) -> list[dict]:
    """A too-large `n` (or a terse max_tokens) can truncate the JSON array
    mid-stream. Rather than discard the whole (expensive) response, pull out
    whichever {"plus":...,"minus":...} objects are individually complete."""
    out = []
    for m in _PAIR_OBJ_RE.finditer(text):
        try:
            out.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return out


def gen_contrast_pairs(concept: str, n: int, *, errors_log: Path | None = None, max_tokens: int = 3000) -> list[dict]:
    """Return up to n {"plus":..., "minus":...} dicts. Azure OpenAI writes
    a JSON list; on parse failure, salvage whichever pairs are individually
    complete (common failure mode: response truncated at max_tokens) before
    giving up. Keep `n` modest (SPEC default calls this with n=25) — asking
    for 100 pairs in one completion reliably truncates even with a large
    max_tokens budget on reasoning-tier deployments."""
    c = CONCEPTS[concept]
    template = _load_template("gen_contrast_pairs.md")
    prompt = _fill(template, concept=concept, concept_definition=c.definition, n=str(n), notes=c.notes)
    for attempt in range(2):
        raw = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=max_tokens)
        try:
            data = parse_json_response(raw)
            if isinstance(data, dict) and "pairs" in data:
                data = data["pairs"]
            if not isinstance(data, list):
                raise JudgeParseError("gen_contrast_pairs: expected a JSON list")
            pairs = [p for p in data if isinstance(p, dict) and "plus" in p and "minus" in p]
            return pairs
        except JudgeParseError as e:
            salvaged = _salvage_pairs(raw)
            if salvaged:
                logger.warning("gen_contrast_pairs concept=%s: full parse failed (%s), salvaged %d/%d pairs from a truncated response", concept, e, len(salvaged), n)
                return salvaged
            logger.warning("gen_contrast_pairs parse failed (attempt %d/2) concept=%s: %s", attempt + 1, concept, e)
            if attempt == 1 and errors_log is not None:
                from src.clock import append_jsonl

                append_jsonl(errors_log, {"error": "gen_pairs_parse_fail", "concept": concept, "detail": str(e)})
    return []


def filter_contrast_pair(concept: str, plus: str, minus: str, *, errors_log: Path | None = None) -> dict | None:
    c = CONCEPTS[concept]
    template = _load_template("filter_contrast_pairs.md")
    prompt = _fill(template, concept=concept, concept_definition=c.definition, plus=plus, minus=minus)
    return _call_judge_json(prompt, max_tokens=200, log_path=errors_log, log_context={"kind": "filter_pair", "concept": concept})
