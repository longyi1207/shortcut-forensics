"""Re-judge rollouts of a given phase whose judge verdict is missing (Azure 429
casualties), appending to the sidecar outputs/<run>/rejudge.jsonl (idempotent:
skips ids already in the sidecar). Usage: python scripts/rejudge_phase.py <phase> [<phase> ...]"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from scripts.run_phase import read_jsonl
from src import judge as judge_mod
from src.agent_loop import transcript_to_text
from src.clock import append_jsonl

run = Path("outputs/20260821-launch")
phases = set(sys.argv[1:]) or {"b5_prompt_decay"}
side = run / "rejudge.jsonl"
done = {r["id"] for r in read_jsonl(side)} if side.exists() else set()
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") in phases and r.get("status") == "ok" and r.get("transcript_path")]
todo = [r for r in rows if not (isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None) and r["id"] not in done]
# File order (oldest first): the judge deployment is the bottleneck tonight, so rows from the N=60 stage
# are labelled before the N=90 extras. (Set SCFX_REJUDGE_NEWEST_FIRST=1 to clear a fresh backlog first.)
if os.environ.get("SCFX_REJUDGE_NEWEST_FIRST") == "1":
    todo.reverse()
# Optional sharding so several passes can run on different judge quotas: SCFX_REJUDGE_SHARD="i/n"
# keeps every n-th row starting at i (rows are in a fixed order, so shards never overlap).
shard = os.environ.get("SCFX_REJUDGE_SHARD")
if shard:
    i, n = (int(x) for x in shard.split("/"))
    todo = todo[i::n]
print(f"phases={sorted(phases)} rows={len(rows)} to re-judge={len(todo)} shard={shard or '-'} route={'openai' if os.environ.get('OPENAI_PREFER_AZURE', 'true').lower() in ('0', 'false', 'no') else 'azure'}", flush=True)

# The judge deployment allows 150k tokens per minute, and a single request above that can never be
# served. Transcripts (at ~4 chars per token) that would exceed TOKEN_CAP are compacted: assistant
# turns keep their tail, tool results are shortened. The sidecar records when this happened so the
# analyses can count labels that came from a compacted input (<1% of rollouts).
TOKEN_CAP = 130_000


def compact(t: list[dict], content_tail: int, tool_head: int) -> list[dict]:
    out = []
    for e in t:
        e = dict(e)
        if e.get("role") == "assistant" and e.get("content") and len(e["content"]) > content_tail:
            e["content"] = "[... earlier part of this turn omitted ...]\n" + e["content"][-content_tail:]
        if e.get("role") == "tool" and e.get("output") and len(e["output"]) > tool_head:
            e["output"] = e["output"][:tool_head] + "\n[... truncated ...]"
        out.append(e)
    return out


for r in todo:
    t = json.loads((run / r["transcript_path"]).read_text())
    dpath = run / "diffs" / r["id"] / "diff.txt"
    diff = dpath.read_text() if dpath.exists() else ""
    text = transcript_to_text(t)
    compacted = None
    for tail, head in ((3000, 800), (1500, 500), (800, 300)):
        if len(text) // 4 + len(diff) // 4 <= TOKEN_CAP:
            break
        compacted = (tail, head)
        text = transcript_to_text(compact(t, tail, head))
    j = None
    for attempt in range(4):
        j = judge_mod.judge_rollout(text, diff, errors_log=run / "incidents" / "judge_errors.jsonl", rollout_id=r["id"] + "_rejudge")
        if isinstance(j, dict) and j.get("is_shortcut") is not None:
            break
        time.sleep(30)
    append_jsonl(side, {"id": r["id"], "phase": r["phase"], "condition": r["condition"], "judge": j,
                        "approx_tokens": len(text) // 4, "compacted": compacted})
    print(r["id"], r["condition"], "->", None if not j else j.get("is_shortcut"), f"~{len(text) // 4} tok", "compacted" if compacted else "", flush=True)
    time.sleep(8)
print("done", flush=True)
