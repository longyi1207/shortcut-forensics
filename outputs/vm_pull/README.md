# VM pull, 2026-09-04

Pulled from the AML VM (`172.213.142.29:/mnt/scfx_ly_run/outputs/20260821-launch/`) before shutdown.
The repo copy of `rollouts.jsonl` was stale: 621 rows, early phases only.

- `labels.jsonl` — 2497 rows, all phases, judge labels without transcripts. Use this for rate checks.
- `rollouts_full.jsonl` — full 243MB archive including transcripts.
- `rejudge.jsonl`, `decisions.jsonl`, `ledger.jsonl` — as on the VM.

Verified against this data on 2026-09-04: tug-of-war 15/30 vs 17/32 p=1.0000; prompt pooled
2/53 vs 8/59; steering 17/32 vs baseline 17/84; `user_tedium_strong` 2/65 vs `identity` 20/58
p<0.0001; unmodified-prompt workaround types 29 `fake_green` / 4 `disable_hook` of 173 judged.

## Final pull, 2026-09-05

Everything off the VM before shutting it down. 11GB total, all byte-verified
against the source with `rsync -an --itemize-changes` returning zero differences.

- `traces/` (6.0G, 547 files) — .npz activation captures from the decision-point runs
- `interconnect_train_detect/` (5.0G) — the separate SPAR GPU-comm project; it lived on
  the ephemeral /mnt disk and would have been lost on deallocate
- `scfx_ly_data_mirror/` (110M)
- `scfx_core.tgz` (101M) — transcripts (1.2G, 2138 files), diffs, activations, vectors,
  judge_raw, logs, phase outputs
- `scfx_code.tgz` (1.7M) — the VM's scripts/src/configs, including 11 scripts that were
  never mirrored to the repo. Excludes .venv and .env.
- `rollouts_full.jsonl` (232M) — md5 verified identical to the VM copy
- `labels.jsonl` — 2497 rows, all phases, judge labels without transcripts; use this for
  rate checks rather than the 232M file

Compute `scfx-compute` (Standard_ND96amsr_A100_v4) stopped, not deleted. The OS disk
still holds `/home/azureuser/scfx_ly_persistent_outputs`, so restarting recovers it.
