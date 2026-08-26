# Limitations

**DRAFT — LY rewrites voice.**

- Single open-weight model family (Qwen), not Kimi K2 Thinking (Singh et al.'s subject) —
  results may not transfer.
- 30-rollout conditions in Phase 6 are small; type-split
  stats in Phase 7 are pre-registered as suggestive, not confirmatory (SPEC.md §5.5).
- Positive-control plants (Phase 5) were run for a shared pool of ~20 rollouts across all
  concepts, not 20 per concept — a concept without enough plant rollouts may have an
  under-powered validity check; see phase5/plant_results.json `n_plants` per concept.
- 20h MATS core budget (+2h writeup) bounded the scope: no thinking-vs-non-thinking sweep
  beyond the Phase 1 kill ladder, no CAFT-style finetuning, no attribution graphs / SAEs.
- Gated DeltaNet hybrid layers (if the recon model uses them) are a known risk for clean
  residual-stream hooks; see decisions.jsonl for whether a fallback model was needed.

Raw phase status JSON is in outputs/20260821-launch/phaseN/status.json for verification.
