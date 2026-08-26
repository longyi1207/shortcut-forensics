# Agent notes — shortcut_forensics

You are implementing or running the experiment in this directory. Read `SPEC.md` first. It beats chat history.

## Run model

Claude Code **owns phases 0–8**. Laptop session may die; `scripts/controller.py` on the A100 spot VM must keep going. Cold resume = `outputs/<run_id>/STATUS.md` + `clock.json`.

## Azure / OpenAI

- Source of truth: repo-root `docs/AZURE.md`.
- Secrets: `/Users/apple/Desktop/ai_notes/.env` only. Never llm-vault for Azure/OpenAI here. Never print keys.
- Chat/judge: OpenAI SDK, `base_url=$AZURE_OPENAI_ENDPOINT/openai/v1/`, `model=$AZURE_OPENAI_DEPLOYMENT` when `OPENAI_PREFER_AZURE=true`.
- Subject models (Qwen) are **not** Azure OpenAI. They run on an **Azure GPU VM**. See `infra/azure.md`.
- Confirm with LY before spinning **A100 PAYG**, a **3rd GPU**, or any run that would exceed remaining budget. Total envelope **$600** (approved 2026-08-13). **A100 spot**, never T4 for the subject model. Autokill 8h. Never 6×(ablate+±α).
- **Credits vs card:** GPU bills the subscription. At **$30** estimated spend, you must have confirmed remaining **Azure credits** cover this (quotaId / Consumption lots / credit summary). If you cannot confirm, **deallocate and wait for LY**. Do not assume Azure OpenAI credits pay for A100s.

## Cloud vs laptop

If a job will take **> 30 minutes** (model download, n=40+ agent rollouts, activation dumps, steer sweeps): run on the Azure GPU VM in tmux, checkpoint every 5 rollouts (`SPEC.md` §4.4). Do not start 27B on the Mac.

## Judges

Use Azure OpenAI for semantic labels (shortcut score, workaround type, verbalization, pair quality, capability). Regex only as a prefilter the judge confirms. Templates in `prompts/`.

## MATS

- Track hours in `TIMELOG.md`. 20h + 2h writeup. See spec §8 for what counts.
- Do **not** submit the MATS application. Draft writeup only.
- Form Qs and exec summary must be human-voiced (LY rewrites). No LLM slop in the submitted pack.

## Engineering

- Resume-friendly JSONL **fsync per rollout**. Frozen config per `run_id`. `decisions.jsonl` + `ledger.jsonl`.
- Logging + tqdm + failure isolation (one bad rollout must not kill the job). Bounded retries then SPEC fallback; no infinite loops.
- Inspectable intermediates: transcripts, diffs, judge raw, activations on disk. `infra/pull.sh` every 15 min.
- Do not expand scope (SAEs, extra envs, extra models, extra steer strengths) unless recon kill-rules in `SPEC.md` §3 fire. Signed intervention pack is in `configs/default.yaml` / SPEC §5.4 — follow it.
- Parallelize: P3 with P1/P2; 2nd A100 spot only for P6 if budget/ETA says so. Never a 3rd GPU.

## Related local code (read, don’t copy blindly)

- `code/emotion_vectors/` — residual hooks, mean-diff, steer
- `code/typebits/cloud/` — job env files, autokill (AWS; we use Azure instead)
