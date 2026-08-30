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
- src/kv_swap.py InstructionKVSwapper: forward hooks on k_proj/v_proj of the 8 full-attention layers overwrite the instruction span's projections at every turn's prefill with a length-matched DONOR's (k_norm + RoPE are position-wise, so this equals a cached-K/V swap; verified vs DynamicCache tensors by scripts/kv_swap_check.py -> KV_SWAP_OK). Text stays, so the GDN/recurrent channel and the residual at the span still see it.
- scripts/dt_kvswap.py 2x2 (n=20 each): dtk_prompt_swapout (instruction text, filler content), dtk_filler (filler text; control), dtk_filler_swapin (filler text, instruction content injected). References: dt_prompt/dtm_prompt, dt_baseline/dtm_baseline. Filler fitted to exactly the instruction's in-context token count (fit_filler).
- analysis: scripts/dt_kvswap_analysis.py (Fisher: swapout vs instr refs = is attention content necessary; swapin vs filler = is it sufficient; filler vs no-line = filler inert)
- hand-off: scripts/chain_stage4.sh (Stage 3b complete -> Stage 4, gated on KV_SWAP_OK)

## Stage 5 — hybrid channel accounting (from Stages 2/4)

## Stage 5-dose — tug-of-war dose-response (queued after Stage 4; no new code)
- motivation: at alpha=1 the prompt gives zero protection (15/30 vs 17/32). Does it shift the threshold at a half dose?
- cells (scripts/prompt_channel.py, env-driven): pc_add_tedium19_a05 (no prompt, add tedium@L19 alpha=0.5, n=20) and pc_prompt_add_tedium19_a05 (prompt on, same steer, n=20); analysis in scripts/pc_analysis.py (direct Fisher between the two)
- hand-off: scripts/chain_stage5_dose.sh (Stage 4 complete -> 4+4 workers)

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
- 2026-08-30 07:48 per-head mask check OK (chosen heads read exactly 0 of the span, others untouched); dt_heads smoke OK (3 turns, replay 1 s). Stage 3 chains armed (chain_stage3.sh, chain_stage3b.sh).
- 2026-08-30 08:20 kv_swap_check: KV_SWAP_OK (|A_swapped - B| = 0.0000 at the span in all 8 full-attn layers, keys and values; |A - B| 5-50 so the donor differs; positions after the span still differ 2-17 -> text/GDN channel intact). dt_kvswap smoke OK (span 70 tokens [470..539], filler fitted exactly, swap on every turn's prefill). Stage 4 chain armed (chain_stage4.sh).
- 2026-08-30 10:00 tug-of-war done: pc_prompt_add_tedium19 = 15/30 (0.50) vs b5_prompt 0/21 (p<0.001), vs add_pos_tedium 17/32 (p=1.0), vs b5_baseline 4/27 (p=0.006). The strong prompt gives zero protection against the tedium direction at alpha=1.
- 2026-08-30 10:00-10:08 INCIDENT: chain_stage1 refused to launch ("dt_capture workers already running") while 0 workers existed -- its `pgrep -f 'dt_captur[e]'` matched an unrelated command line. GPUs idle 8 min; Stage 1 launched manually 10:08. FIX: every chain guard now anchors on the real worker cmdline (`^/mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/<script>.py`); chains re-armed.
- NEXT (all VM-chained, no operator needed): tug-of-war >=30 -> Stage 1 (dt_capture x8) -> Stage 2 (dt_mask x8) -> Stage 3a (dt_heads x8) -> rank -> Stage 3b (head-restricted block x8) -> Stage 4 (dt_kvswap x8). Analyses + WRITEUP after each stage.
- 2026-08-30 11:30 OPS: root disk hit 100% (119G; HF cache 29G incl. 11G of SAEs on /); dt_analysis could not load the L27/L31 SAEs ("No space left"). HF cache moved to /mnt/hf_cache with a symlink at ~/.cache/huggingface (transparent to all processes; workers untouched); pip/apt/journal caches cleared. Re-run dt_analysis at n=20 to get L27/L31 SAE columns.
- 2026-08-30 13:25 Stage-1 midpoint (12+12): nearly all rollouts hit max_turns=80; `git commit` attempted in only 4/13 baseline and 2/12 prompt rollouts (all hook-blocked) -> commit_cmd tokens are scarce by design; post-failure turns (~12/rollout, ~3000 tagged tokens/arm) are the abundant decision events. ||dh||/||h|| at post_fail_first20 > controls at every layer (e.g. L26 0.088 vs 0.054/0.046; L31 0.134 vs 0.084/0.073); cos(dh, tedium) -0.03..-0.07.
- 2026-08-30 13:25 Stage 2 upgraded before launch: (a) KL probe at every post-failure turn + every 4th turn (first 20 tokens replayed, block off vs on: KL, generated-token log-prob, argmax changes); (b) split prefill so the block also covers the FIRST generated token (was unmasked: hook skips prefill). Smoke OK (kl_first=0.031>0). Stage 3b inherits both.
- 2026-08-30 14:40 Stage 1 complete (20/20 both arms); chain_stage2 handed off cleanly (anchored guards): Stage 2 launched, 8 workers (dtm_prompt x1 N=12, dtm_baseline x1 N=12, mask_instr/notes/both x2 N=20 each), with KL probe + split prefill.
- 2026-08-30 15:10 Stage 1 RESULT (20+20): behaviour 4/20 vs 2/20; ||dh||/||h|| at post_fail_first20 = 0.093/0.100/0.106/0.160 (L19/23/26/31) vs other_cmd 0.044/0.041/0.044/0.067 and every50 0.038/0.037/0.041/0.061 (2-2.5x, all layers); commit_cmd same level but 9 vs 4 rollouts. cos(dh,tedium) -0.05..-0.09 at post-fail tokens; L19 tedium proj 20.3->15.9 (p=0.029) but also small shifts at controls; L26 SAE-10 no move (p=0.2-0.4). turn_start has literal instruction-presence SAE features (20/20 vs 0/20, d ~ +885..+3000) that are ABSENT at decision tokens; decision-token features are shared by both arms with d ~ 1-1.8 (mostly baseline-enriched, prompt-suppressed). Written to WRITEUP §4.12. NEXT analysis: feature-context pass (which tokens/turns activate the top decision-token features).
- 2026-08-30 15:50 feature-context pass (scripts/dt_feature_context.py, 11,060 decision-token residuals): prompt-SUPPRESSED features = cost-of-effort framing + change of plan ("Given the time constraints, let me take a different approach", "This is a lot of files to fix") and requirement re-framing ("mypy is set to strict mode which...", "However, the user said..."); prompt-ENHANCED = engagement with the specific error ("I see the issue - I used any instead of Any"), enumerating remaining fixes ("Let me fix them: 1. ..."), and quoting the task requirement ("The task says '...'"). All fire in both arms (balance shift, not a switch); |cos| with tedium <= 0.14. Written to WRITEUP §4.12.
- 2026-08-30 15:10 BUG (mine) + fix + restart: every hand-written prefill (split_prefill, kl_probe, score_continuations, dt_heads replay, kv_swap capture) materialised logits for ALL positions -> 21-24 GB allocations, dozens of allocator OOM warnings, certain hard OOM later in an 80-turn rollout. Added logits_to_keep=1 everywhere; restarted the 8 Stage-2 workers 15:10 (only 1 completed row discarded). Post-restart allocations 1.2-2.8 GB (normal KV growth), zero OOM warnings on the 6 masked workers. generate() does this internally, which is why Stage 1 never hit it.
- 2026-08-30 16:12 Stage-2 machinery verified on live rows: mask_notes rollout masked 9 turns / 25,320 decode steps; mask_both 8 turns / 17,760. Span audit (scripts/check_spans.py, offline, no restart): instruction span exactly 70 tokens; notes span 410-3006 tokens = 9-15% of context, every assistant text found. FIRST NUMBERS: blocking direct decode-time attention to the instruction and/or the notes changes the next-token distribution by KL(off||on) ~ 0.001-0.006 with 0/20 argmax changes -- i.e. at the decision the model is not re-reading the instruction directly. Expected given the mask is decode-only and direct-edge-only (indirect path span -> later prompt tokens -> generated token stays open, as does the GDN recurrent channel); Stage 4's prefill K/V swap is the decisive test. NOTE: notes_span_len is not stored in the rows; recover it offline with check_spans.py.
- 2026-08-30 16:45 REPRIORITISED (agent decision, no operator input): chain is now Stage 2 -> Stage 3a -> **Stage 4** -> dose-response; Stage 3b (head-restricted instruction block, 80 rollouts ~ 10 GPU-h) is NOT armed. Rationale: Stage 3b is a strictly weaker version of the Stage-2 block (subset of heads), and Stage 2's first rows put the full block at KL ~ 0.001-0.006 with 0/20 argmax changes -- it cannot show more than the full block, so it must not sit in front of the decisive experiment. Stage 3a is kept (it MEASURES which heads read the instruction, informative either way, and its ranking file feeds any later 3b run). Re-arm scripts/chain_stage3b.sh after Stage 4 only if Stage 2 at n=20 shows a real masking effect.
- NOTE: workers with long contexts reach ~75 GB/GPU; side checks must pick a GPU with >=24 GB free (nvidia-smi poll) rather than a fixed GPU.
