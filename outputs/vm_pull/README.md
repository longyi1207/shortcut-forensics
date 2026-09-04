# VM pull, 2026-09-04

Pulled from the AML VM (`172.213.142.29:/mnt/scfx_ly_run/outputs/20260821-launch/`) before shutdown.
The repo copy of `rollouts.jsonl` was stale: 621 rows, early phases only.

- `labels.jsonl` — 2497 rows, all phases, judge labels without transcripts. Use this for rate checks.
- `rollouts_full.jsonl` — full 243MB archive including transcripts.
- `rejudge.jsonl`, `decisions.jsonl`, `ledger.jsonl` — as on the VM.

Verified against this data on 2026-09-04: tug-of-war 15/30 vs 17/32 p=1.0000; prompt pooled
2/53 vs 8/59; steering 17/32 vs baseline 17/84; `user_tedium_strong` 2/65 vs `identity` 20/58
p<0.0001; unmodified-prompt workaround types 29 `fake_green` / 4 `disable_hook` of 173 judged.
