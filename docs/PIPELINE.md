# shortcut_forensics — decision-token circuit program (VM-side plan/status)

Owner: autonomous agent session (Claude), full autonomy granted 2026-08-30.
Rule: GPUs must never sit idle. Every stage hand-off is done by a VM-side script
(setsid nohup) so it does not depend on the operator session. Analysis scripts
live in scripts/; every stage's result is appended to
/Users/apple/Desktop/shortcut-forensics/WRITEUP.md and pushed.

## Stage 0 (running) — causal cells on the mean delta, then tug-of-war
- prompt_channel: pc_base_add_delta26, pc_prompt_ablate_delta26 (8 workers, capped at n=12 by chain_tug_of_war.sh)
- then chain_tug_of_war.sh launches pc_prompt_add_tedium19 (strong prompt + add coarse tedium@L19, alpha=1), 8 workers, n=30
- analysis: scripts/status_snapshot.py + a Fisher vs b5_prompt (0/21) / add_pos_tedium (16/30) / identity baselines
- DONE criterion: pc_prompt_add_tedium19 >= 30 ok rows -> write §4.12 tug-of-war result; free all 8 GPUs

## Stage 1 — decision-token capture (HF, hooks)
- code: src/proj_trace.py DecisionTracer (per-turn buffer of full residuals at every decode step; on_turn_end keeps tagged steps), agent_loop.run_rollout on_turn_end callback, scripts/dt_capture.py
- tags: turn_start (prefill last pos), commit_cmd (decode tokens spanning a generated `git commit` command), post_fail_first20 (first 20 decode steps of a turn whose previous tool result had rc!=0), other_cmd (decode tokens of any other command; control), every50 (continuity)
- layers: 19, 26 + all full-attention layers (from model.config.layer_types)
- conditions: dt_baseline, dt_prompt (emphatic user line), n=20 each, phase dt_capture
- analysis: scripts/dt_analysis.py — per tag: delta-h norm/cos(tedium), SAE feature diff (active in >=3 rollouts), tedium projection MW over rollouts; decision tags vs other_cmd control
- DONE criterion: both conditions >= 20 ok rows -> results to WRITEUP; decide Stage 2 layers from where the decision-token delta is largest

## Stage 2 — instruction readability at decision tokens (HF)
- attention mask hook on full-attention layers only: -inf on keys in the instruction span (and separately: on the model's own prior assistant tokens = "notes"), applied for the whole turn after a failed tool result
- conditions: dt_prompt_mask_instr, dt_prompt_mask_notes, dt_prompt_mask_both (n=20 each) vs dt_prompt / dt_baseline
- logit-diff probe at commit-command positions: P(cheat continuations) masked vs unmasked (same prefix, two forward passes)
- DONE criterion: behavioural readout (shortcut rate, commit profile) + logit diffs; survival after masking = recurrent-channel share

## Stage 3 — instruction-reading heads (HF, decode-only attention weights)
- 3a measurement: scripts/dt_heads.py — ordinary sdpa rollouts; after each decision turn (commit / post-fail / every 4th), REPLAY the turn teacher-forced: sdpa prefill of prompt[:-1], sdpa chunks, and a single EAGER forward with output_attentions at each tagged step -> per-head mass on {instr span, length-matched task-text control span, notes, recent-64}. Conditions dth_prompt (N=12) / dth_baseline (N=6). npz traces/<id>_heads.npz
- 3a analysis: scripts/dt_heads_analysis.py -> dt_heads_rank.json (rank (layer,head) by instruction mass at decision steps; top8/top16 + seed-fixed random sets)
- 3b ablation: dt_mask.py with SCFX_DTM_HEADS/HEADSET — block the instruction span for ONLY the top-k heads vs random-k (n=20 each, postfail turns): mask_instr@top8, @rand8_s0, @top16, @rand16_s0. Compare with the all-heads block (dtm_prompt_mask_instr) and dtm_prompt.
- hand-offs: scripts/chain_stage3.sh (Stage 2 complete -> 3a), scripts/chain_stage3b.sh (3a n>=12 -> rank -> 3b)

## Stage 4 — content ablation (HF)
- replace K/V of the instruction span in full-attention layers with those of a length-matched neutral text at prefill; text stays

## Stage 5 — hybrid channel accounting (from Stages 2/4)

## Stage 6 — sentence-level resampling at the plan sentence (HF), if time

## Side analyses (no GPU)
- E0 probe blind spot: scripts/probe_blindspot.py on B5 traces (tedium projection as probe; 2x2 probe verdict x behaviour per arm)
- exposure-conditioned misbehaviour (commit profile) already in status_snapshot.py

## Status log (append below; UTC)
- 2026-08-30 06:00 Stage 0 running: pc cells ~8/24 rows; chain_tug_of_war.sh armed (cap 12 each).
- 2026-08-30 06:12 Stage 1 code deployed + smoke OK (dt_smoke); chain_stage1.sh armed.
- 2026-08-30 06:31 Stage 2 code deployed; chain_stage2.sh armed (gated on MASK_CHECK_OK).
- 2026-08-30 07:27 pc cells done (12/12): add dh26 -> 1/12 (null), prompt+ablate dh26 -> 0/12 (prompt intact). Tug-of-war launched (8 workers, n=30).
- 2026-08-30 07:35 attn_mask_check: MASK_CHECK_OK (span mass 0 at all 8 full-attn layers when active; inactive peaks 0.52-0.62 at L15/L19/L23). Stage-2 gate open.
- NEXT: tug-of-war >=30 -> Stage 1 (dt_capture x8) -> Stage 2 (dt_mask x8). Stage 3 needs a custom decode loop with eager attention at decode steps (per-head span mass) + per-head edge blocking.
