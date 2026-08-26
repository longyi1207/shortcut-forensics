# Azure GPU — subject-model jobs

Chat/judges stay on **Azure OpenAI** (`docs/AZURE.md`). This file is only for **Qwen + hooks**.

**Preferred path (2026-08-20):** Microsoft for Startups **GPU cluster** — LY applied → capacity ~**2 days** later. Canonical: `docs/AZURE.md` § GPU.

**Active shared box (2026-08-21–23):** AML `scfx-compute` (`scfx-ws-italynorth` / Italy North / `Standard_ND96amsr_A100_v4` 8×A100). This experiment typically uses a GPU slice on that node — see `infra/config.aml.env` / `launch_aml.sh`. Re-request cluster capacity for future windows after this reservation ends.

Legacy fallback: `IncidentFox` / eastus `Standard_NC24ads_A100_v4` spot — only if cluster unavailable.

## Defaults

| | |
|---|---|
| Preferred | Startups **GPU cluster** allocation (see `docs/AZURE.md`) |
| Subscription (fallback VM) | `AZURE_SUBSCRIPTION_ID` in repo-root `.env` (`27ad7138-6e41-4554-9d72-36eb4502b0bb`) |
| RG | `IncidentFox` |
| Location | `eastus` |
| Recon / main SKU | A100 80GB-class (T4 cannot fit 9B bf16 + hooks) |
| Autokill | 8 hours |
| Budget cap | **$600** total (LY 2026-08-13) |
| Confirm with LY | 27B **PAYG**, 3rd GPU, a run that would blow remaining budget, or **credits-vs-card unconfirmed at $30** |

**Credits gate:** GPU is Compute / Startups metered capacity, not Azure OpenAI credit. After ~$30 estimated spend, agent must show `Sponsored_*` / monetary credit lots / balanceSummary **or stop and wait for LY**. `PayAsYouGo` + card is a stop even at $1.

## Rules

1. `az login` / `az account set --subscription ...` before creating anything.
2. Do not leave GPUs up overnight. Checkpoint → deallocate or delete.
3. tmux session `shortcut`. JSONL append so a kill is resumable.
4. rsync this directory + need `HF` cache on the VM disk.
5. Load `.env` on the VM for `HUGGING_FACE_TOKEN` only if required; **judge calls can stay on laptop** hitting Azure OpenAI so the VM is just GPU.

## Suggested bring-up (implementer fills exact image/size after quoting price)

```bash
az vm list-skus --location eastus --size Standard_NC --output table
# then az vm create ... --resource-group IncidentFox ...
```

Write instance id / IP to `infra/.active/<job_id>.env` (gitignored). Mirror `code/typebits/cloud/` job-file pattern.

## Cost sanity

eastus PAYG A100 ~$3.67/h, spot ~$0.68/h. **Always spot first** (`--priority Spot --eviction-policy Deallocate`). T4 is not used: 16GB VRAM cannot do 9B bf16 + residual hooks.

Expected: 9B signed pack on A100 spot ~$80–150; PAYG 27B can approach $400. 6×(ablate+±α)=630 rollouts forbidden. Don't leave A100s up overnight.
