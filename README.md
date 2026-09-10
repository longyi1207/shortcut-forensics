# Shortcut Forensics

Prompting and contrast-direction steering do not move the same thing: a per-factor test on an agentic shortcut task (Qwen3.5-9B in the Singh et al. pre-commit environment). MATS 12.0 application, Neel Nanda stream.

**Read first:** [`writeup/MATS_SUBMISSION_v2.md`](writeup/MATS_SUBMISSION_v2.md) (the submission; executive summary on top). Result tables regenerated from the data: [`writeup/RESULTS_2026-09-10.md`](writeup/RESULTS_2026-09-10.md). What ran and what broke on the final night: [`writeup/NIGHT_LOG_2026-09-10.md`](writeup/NIGHT_LOG_2026-09-10.md). Figures: [`writeup/figs/`](writeup/figs/).

## TL;DR

For five candidate motives (tedium, desperation, temptation, wanting to be done, fear of the user's disapproval): a one-line instruction in the prompt cuts the shortcut rate from 31% to 0-11% for every factor (n = 90 each; a same-length neutral line does nothing, 150 vs 150). Steering on the matching contrast directions moves behaviour for tedium only (ablation 9%, addition 53%, random direction 29%). Even for tedium the instruction's footprint on the residual stream is not along the direction, while the concept's own sentences do move along it. The instruction reaches the decision through the context (the agent writes it into its own plan), a route a fixed direction does not express, so a direction that steers and reads its own concept still misses instruction-controlled behaviour.

The earlier write-up of this project ([`WRITEUP.md`](WRITEUP.md), [`writeup/MATS_SUBMISSION.md`](writeup/MATS_SUBMISSION.md)) is kept as a record; its component- and sentence-level attributions rested on a decision-point readout later found not to track the judge labels, and are withdrawn in v2 §7.

**Data.** `outputs/20260821-launch/rollouts.jsonl` (3,271 rollout rows with judge labels; 327 MB) is published under the release `data-2026-09-10` as seven 6 MB parts (`cat rollouts.jsonl.gz.part-* > rollouts.jsonl.gz`, md5 in the release notes) rather than tracked in git. Transcripts, diffs and activations (8 GB) are available on request.

## Layout

| Path | What |
|---|---|
| [`WRITEUP.md`](WRITEUP.md) | **Full results writeup** — read this first |
| [`SPEC.md`](SPEC.md) | Source of truth for methodology — context, research questions, method, setup, phases, data schema |
| [`CLAUDE.md`](CLAUDE.md) | Agent rules used to run this (Azure, cloud, 20h clock, no MATS submit) |
| [`configs/default.yaml`](configs/default.yaml) | Experiment knobs |
| [`prompts/`](prompts/) | Azure OpenAI judge / contrast-pair-generation templates |
| [`src/`](src/) | Direction fitting, steering, judging, agent loop, hooks |
| [`scripts/`](scripts/) | Phase runner (`run_phase.py`), VM controller, and same-night repair scripts |
| [`infra/`](infra/) | GPU VM bring-up / Azure config |
| [`outputs/20260821-launch/`](outputs/20260821-launch/) | Run artifacts tracked in git: fitted vectors, phase summaries, judge outputs; the rollout table is a release asset, transcripts and activations are not in git (see Data above). Original text: rollouts, transcripts, activations, fitted vectors, judge outputs, every phase's results. Included in full for reproducibility, not gitignored. |
| [`TIMELOG.md`](TIMELOG.md) | Neel 20h accounting |

## Compute

Ran on a shared 8×A100 GPU allocation (Microsoft for Startups Startup GPU Cluster, Italy North) plus Azure OpenAI for judge/pair-generation calls. Subject model: `Qwen/Qwen3.5-9B`, bf16 — recon cleared the shortcut-rate kill rule comfortably, so the escalation path to a larger model never triggered.

## Reproducing

```bash
cd shortcut-forensics
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_phase.py --phase <0-8> --run-id 20260821-launch --resume
```

Needs a GPU for phases 1, 2, 4, 5, 6 and Azure OpenAI credentials (see `infra/azure.md`) for judge/pair-generation calls. `outputs/20260821-launch/` already contains a complete run if you just want to inspect results without re-running anything.
