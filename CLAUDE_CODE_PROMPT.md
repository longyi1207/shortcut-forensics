# Claude Code prompt — shortcut_forensics full run (phases 0–8)

> Paste **everything below the line** into a **brand-new Claude Code session**.  
> cwd = `/Users/apple/Desktop/ai_notes` (repo root). Do **not** paste this header.  
> Approve `az login` / VM create if asked. **Never** click MATS Submit.  
> Laptop will sleep. The **VM controller** must keep running without you.

---

You are the experiment runner for **Shortcut Forensics**. Finish the **entire** experiment (phases **0–8**), not a scaffold. LY will not send a second prompt for later phases. If this Claude Code session dies, the VM controller and on-disk state must be enough for a new session to resume with zero chat context.

Work in `app/shortcut_forensics/`. Read in order: `SPEC.md` (beats this prompt except locked compute), `CLAUDE.md`, `configs/default.yaml`, `docs/AZURE.md`, `infra/azure.md`, `prompts/`.

Steal patterns: `code/emotion_vectors/src/{activation_extraction,emotion_vectors,steering}.py`; `code/typebits/cloud/` job files (**Azure not AWS**).

---

## 0. What “done” means

`outputs/<run_id>/STATUS.md` says **COMPLETE** or **ABORTED** (recon kill / budget / unrecoverable).

COMPLETE requires:

- Phases 0–7 artifacts on disk (and rsynced off the VM).
- Phase 8 **draft** writeup in `writeup/` (mark `DRAFT — LY rewrites voice`). Do not submit MATS.
- `ledger.jsonl` with GPU-hours and estimated USD.
- `decisions.jsonl` with every kill/switch/retry/SKU change.

ABORTED is allowed only via SPEC §3 (no shortcuts after the switch ladder) or remaining budget cannot finish even the tight P6 pack. Write why. Deallocate GPUs.

---

## 1. Nine phases (you own all of them)

| N | Name | Depends on | Runs where | Done when |
|---|---|---|---|---|
| 0 | Bring-up | — | laptop + 1× A100 spot | `phase0/ok.json` (ping + VM + 9B load + residual shape) |
| 1 | Recon n=40 | 0 | GPU | rate table; §3 applied |
| 2 | Natural n≤100 until ≥20 shortcuts + 20-error sanity + activations | 1 passed | GPU | `rollouts.jsonl` |
| 3 | Contrast pairs 80+20 × 8 concepts | 1 looking healthy | **laptop / Azure OpenAI**, parallel with 1–2 | `pairs/<concept>/{train,val}.jsonl` |
| 4 | Fit `d` | 2 (≥20 shortcuts) **and** 3 | GPU | `vectors.pt`, val table, cosine heatmap |
| 5 | Readout + ~20 plants | 4 | GPU | signed AUC; invalid `d`s dropped |
| 6 | Signed intervention pack | 5 | GPU (2nd A100 spot OK if §4) | rate × condition + capability |
| 7 | Type-split + 2–3 figs | 6 (or 5 if P6 shrunk) | local or GPU | `phase7/*.png` + tables |
| 8 | Writeup draft | 7 | local | `writeup/` DRAFT |

**Kill gates (do not skip):**

- P1 <8% → thinking off → 27B same VM → env switch → stop. No “please cheat” prompt.
- P2 <20 shortcuts → do not fit `d`.
- P4 val acc <90% → drop that concept.
- P5 plant does not raise \(h\cdot d\) on the plus pole → that `d` is invalid; do not interpret natural nulls.
- P6: **forbidden** 6×(ablate+±α)=630. Default pack = `configs/default.yaml`. If remaining $ < ~1.3× estimated full pack, shrink to identity + winner signed + second signed + random + prompt.

---

## 2. Clock — you must have one

Maintain `outputs/<run_id>/clock.json` (rewrite atomically every heartbeat, ≤5 min while jobs run):

```json
{
  "run_id": "...",
  "phase_active": [2, 3],
  "phase_done": [0, 1],
  "gpu_hours": 12.4,
  "usd_est": 8.4,
  "usd_cap": 600,
  "usd_remaining": 591.6,
  "credits_confirmed": false,
  "credits_check": {"at_usd": null, "method": null, "balance_usd": null, "quota_id": null},
  "spot": true,
  "n_gpus": 1,
  "rollouts_done": 40,
  "shortcuts": 7,
  "eta_phase_complete_utc": "...",
  "autokill_utc": "...",
  "blocker": null
}
```

Also `STATUS.md` in English, human-readable, **cold-resume complete**: what is running, tmux name, JSONL path, last error, next action, exact resume command.

**Resource policy (you decide, don’t ask):**

| Situation | Do |
|---|---|
| GPU idle >10 min and a GPU phase is pending | start it |
| P1 looks ≥8% at n≈15 | start P3 on laptop immediately |
| P2 ≥20 shortcuts and P3 done | start P4 without waiting for n=100 (keep P2 collecting in background if you want more n) |
| P6 is the bottleneck, `usd_remaining` > 80, 1 GPU ETA > 24h | **2nd** A100 **spot** only; split independent conditions; never a 3rd |
| Spot eviction / autokill | disk kept (Deallocate). Relaunch spot, **resume JSONL**, do not restart from scratch |
| Spot capacity fail ×2 (eastus, westus2) | one PAYG A100, log it |
| `usd_remaining` < estimated next phase × 1.3 | shrink P6; if still short, finish P5+P7+P8 and ABORT P6 with a note |
| `usd_est` ≥ **$30** and `credits_confirmed` is not true | **HARD STOP.** Deallocate all GPUs. §2.1. Do not keep burning. |
| Same failure 3 times | write `incidents/<id>.md`, take SPEC fallback (14B hooks, thinking off, env switch). 4th time: stop that branch, don’t infinite-loop |
| Laptop sleeps | irrelevant; controller is on the VM |

Hourly rate for the ledger: A100 spot **$0.68/h**, PAYG **$3.67/h**. Judge tokens are cheap; still log call counts.

Do **not** wait for LY except: `az login` missing; would exceed $600; 3rd GPU; **credits check failed / ambiguous** (§2.1).

### 2.1 Credits vs real billing (hard gate)

LY’s $600 envelope is **Azure credits**, not a green light to hit a credit card. GPU VMs bill the **subscription (Compute)**. That is **not** Azure OpenAI / Foundry credit. Do not assume chat credits cover A100s.

**Try to confirm at P0** (before a long recon). If you cannot, you may bring up one spot VM, but:

**When `usd_est` ≥ $30 and `credits_confirmed` is not true: deallocate every GPU, stop Azure OpenAI pair-gen, set `blocker: credits_unconfirmed` in clock.json + STATUS.md, and wait. Do not resume until LY types that this subscription’s spend is credits.**

What counts as **confirmed** (need at least one; log raw JSON under `outputs/<run_id>/billing/` with secrets redacted):

1. `az account show` / `az account list` → `quotaId` looks like `Sponsored_*`, `AzureInOpen_*`, `MSDN_*`, `MonetaryLimit*` / `spendingLimit` is `On` or `CurrentPeriodOff` **and** remaining monetary credit covers ≥ remaining experiment. `PayAsYouGo_*` with `spendingLimit: Off` is **not** confirmation.
2. Consumption lots / credit summary via REST, remaining `closedBalance` / `estimatedBalance` **> $200** and lots are `Active` (not expired):
   ```bash
   az billing account list -o json
   az billing profile list --account-name <id> -o json
   az rest --method get --url \
     "https://management.azure.com/providers/Microsoft.Billing/billingAccounts/<ba>/billingProfiles/<bp>/providers/Microsoft.Consumption/credits/balanceSummary?api-version=2024-08-01"
   az rest --method get --url \
     "https://management.azure.com/providers/Microsoft.Billing/billingAccounts/<ba>/billingProfiles/<bp>/providers/Microsoft.Consumption/lots?api-version=2023-03-01"
   ```
3. Cost analysis: this month’s charges sit on a **credit / promotional** lot, not an invoice to a card. Ambiguous → unconfirmed.

If APIs 403, empty, or you are guessing: **unconfirmed**. Tell LY exactly what you ran and what came back. One blocker sentence. Leave VMs **deallocated** (disk kept). Do not treat “LY has Azure credits” in `docs/AZURE.md` as confirmation of **this** SKU on **this** subscription.

If the check says credit balance is **$0** or PAYG card will be invoiced: stop even at $1, same wait.

Controller must enforce this gate every heartbeat, not only at phase boundaries.

---

## 3. Controller (required — this is how the experiment survives a dead session)

Implement `scripts/controller.py` that:

1. Reads `clock.json` + phase `status.json`.
2. Runs the next eligible phase(s) in parallel when dependencies allow.
3. Heartbeats clock + STATUS.md. If `usd_est ≥ 30` and credits not confirmed → deallocate, set blocker, **do not relaunch**.
4. On process crash: restart the phase with `--resume`.
5. On 8h autokill approaching (t−30 min): checkpoint, note `needs_relaunch`, exit 0. A `infra/relaunch.sh` + crontab/systemd **on the VM** should bring the controller back after Deallocate/start. If you cannot install a persistent relaunch, document in STATUS.md: “LY must `az vm start` + `tmux attach`” — still JSONL-safe.
6. `infra/pull.sh` rsyncs `outputs/` to the laptop path **and** keeps a second copy on the VM data disk (`/data/shortcut_forensics/outputs/` or equivalent). Pull at least every 15 min and at every phase boundary.

Long GPU jobs: **tmux `shortcut`**, never a laptop foreground process.

---

## 4. Artifacts — nothing in RAM only

Layout (gitignored `outputs/`):

```
outputs/<run_id>/
  STATUS.md
  clock.json
  config.frozen.yaml
  ledger.jsonl
  decisions.jsonl          # {ts, type, detail}
  incidents/               # failures that needed a fallback
  spotcheck.json           # 30 ids for LY; do not block on him
  rollouts.jsonl           # APPEND ONLY, flush every rollout
  activations/<id>.pt      # per-rollout, not one giant file
  transcripts/<id>.json    # full messages + tools
  diffs/<id>/
  judge_raw/<id>.json
  pairs/<concept>/{train,val,rejected}.jsonl
  vectors/<concept>.pt
  phaseN/status.json
  phaseN/*.png tables
  writeup/
  logs/controller.log
```

Rules:

- Append JSONL **then fsync** before starting the next rollout. Crash-safe.
- Never overwrite `rollouts.jsonl`; resume = skip ids already present.
- Activations: decision token + last-prompt per turn only (SPEC). If a write fails, still keep transcript+diff; mark `activation_path: null` and continue.
- Log prompts/templates used (hash + filename), generation kwargs, model id, thinking flag, layer indices, env git SHA.
- No secrets in any artifact. Redact `.env`.
- `decisions.jsonl` types: `sku_change`, `spot_evict`, `kill_rule`, `retry`, `shrink_p6`, `drop_concept`, `hook_fallback`.

---

## 5. Failure handling (autonomous)

| Failure | Handle |
|---|---|
| Judge JSON parse | retry once; skip + log; don’t halt the job |
| Single rollout tool crash / timeout | retry twice; then `status=error`, continue |
| Qwen3.5 residual hook broken | <2h then fallback `Qwen/Qwen2.5-14B-Instruct`, re-run recon, `decisions.jsonl` |
| OOM | do **not** 4-bit. Reduce `max_new_tokens` / batch=1. Still OOM → 14B fallback or 27B only if §3 |
| Azure OpenAI 429 | exponential backoff (tenacity), cap 10 min, then pause that stream |
| VM SSH drop | retry SSH; `az vm get-instance-view`; if evicted, start + resume |
| Env clone / mypy missing | install; pin SHA; don’t silently change the task |
| Shortcut rate ~0 after full §3 ladder | ABORT, recon note, deallocate, Phase 8 “we stopped” draft |

Never `except: pass`. Never delete `outputs/` to “start clean” unless you copy it to `outputs/<run_id>.bak/` first.

---

## 6. Locked compute / science (do not re-litigate)

- One A100 spot `Standard_NC24ads_A100_v4`, Deallocate, eastus / IncidentFox. T4 is wrong (9B bf16 + hooks). No 4-bit subject model.
- Recon: `Qwen/Qwen3.5-9B` thinking on first.
- Budget ≤ $600. Autokill 8h.
- Signed P6 pack from yaml. Pro-cheat **ablate**; disapproval **+α**; eval-aware both; −α winner only.
- Secrets: repo-root `.env` only. Never print keys.
- Prefer raw PyTorch hooks. nnsight optional.

Azure OpenAI:

```python
# load repo-root .env
# prefer AZURE_OPENAI_* → OpenAI(base_url=ENDPOINT/openai/v1/, model=DEPLOYMENT)
```

---

## 7. Modules to write

```
src/llm_client.py  env_precommit.py  agent_loop.py  hooks.py
src/directions.py  judge.py  steer.py  plot.py  clock.py
scripts/run_phase.py       # --phase N --resume --run-id
scripts/controller.py      # state machine; default entrypoint on VM
infra/launch.sh bootstrap.sh autokill.sh pull.sh relaunch.sh
tests/test_judge_parse.py tests/test_mean_diff.py tests/test_resume_jsonl.py
```

Coding agent = tool/bash loop in the pre-commit env (`gkroiz/agent-interp-envs`, pin SHA), max 80 turns, not a single completion.

Tests: mock judge parse; mean-diff recovers a planted axis; JSONL resume skips existing ids. No CI 9B.

---

## 8. Suggested schedule (flex if the clock says otherwise)

```
hour 0–3     implement + tests + P0
hour 3–13    P1 on GPU; start P3 as soon as P1 rate isn’t dead
then         P2+P3 parallel
then         P4 (minutes) → P5 plants (~5h) → P6 (biggest; maybe GPU #2)
then         P7 plots → P8 draft → deallocate
```

If P1 fails early, do not burn P3 tokens. If P1 passes at n=40, do not sit on your hands waiting for a human.

---

## 9. Out of scope

SAE, extra horses, α sweeps, 630-rollout factorial, SPAR text, **submitting** MATS, FlexHEG, 27B “just to see.”

Phase 8 **is in scope as a draft**. Label it DRAFT. LY rewrites. You still have to write it from the actual numbers.

---

## 10. How to talk to LY

Do not dump a plan and wait. Implement, launch, heartbeat.

If you need him: one blocker sentence (`az login` / over $600 / 3rd GPU / **credits unconfirmed — paste what `az account` / billing REST returned**). Otherwise only update `STATUS.md`.

When COMPLETE/ABORTED: run_id, $ burned, phase table, path to STATUS.md, VM deallocated? yes/no, draft writeup path.
