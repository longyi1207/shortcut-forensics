# Shortcut Forensics

MATS 12.0 / Neel Nanda Winter 2027 application experiment: **what internal direction (if any) drives naturalistic coding-agent shortcuts, and can you move it on the original prompt with the sign-appropriate intervention?**

**Status: experiment complete.** See [`WRITEUP.md`](WRITEUP.md) for the full account — setup, methodology, every phase's results, findings, and open questions. This README is a quick orientation; the writeup is the real document.

## TL;DR

On `Qwen/Qwen3.5-9B`, in a pre-commit-hook coding environment, none of six pre-registered candidate mechanisms (tedium, eval-awareness, disapproval, desperation, Wu-style shortcut, completion-drive) reached statistical significance as a cause of shortcut-taking in the original confirmatory analysis — including a known-working positive control, confirming the study was underpowered rather than genuinely null throughout. A same-night repair round found and fixed a stale-vector bug affecting two other concepts, but the properly-repaired causal tests, even extended to real power, also came back null. A **post-hoc exploratory phase** then found the project's strongest evidence, in two stages: ablating `tedium` reduces the shortcut rate via two independent methods (p=0.028 coarse direction, p=0.0050 SAE decomposition), and a follow-up sufficiency test found that *amplifying* the same direction sharply increases it (`add_pos_tedium`, n=30, 53.3% vs. 20.0% baseline, **p=0.0009** — the project's strongest result, clearing even a strict multiple-comparisons bar on its own). Necessity and sufficiency both holding, symmetrically, for the same concept at the same layer is hard to explain as generic capability damage. It hasn't been independently replicated from a fresh data split, so it's reported as strong exploratory evidence, not a fully closed discovery. Full findings, numbers, and the honest read on all of it is in [`WRITEUP.md`](WRITEUP.md).

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
| [`outputs/20260821-launch/`](outputs/20260821-launch/) | **Full run artifacts** — rollouts, transcripts, activations, fitted vectors, judge outputs, every phase's results. Included in full for reproducibility, not gitignored. |
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
