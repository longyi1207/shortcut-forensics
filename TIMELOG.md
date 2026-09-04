# TIMELOG — MATS 12.0, Neel Nanda stream

**This is a reconstruction, not a live track.** I did not run a timer. The rows below were
rebuilt on 2026-09-04 from git history (87 commits, 2026-08-13 to 2026-09-03), the SPEC, and
the run artifacts. I am reporting it as a reconstruction rather than presenting a tidy Toggl
export I do not have.

Counted, per the admissions doc: design and pre-registration, code I wrote or directed and
reviewed, analysing results, re-scoping decisions, and the write-up.

Not counted, per the same doc: cluster bring-up, breaks, form answers, and time waiting on
runs. That last exclusion is most of the wall-clock here. The pipeline ran autonomously on
8 GPUs for roughly ten days behind a work queue, and I was not at the keyboard for most of it.
Elapsed calendar time is three weeks; engaged time is the table.

| dates | phase | h | what |
|---|---|---|---|
| 08-13 to 08-20 | scoping | 4.0 | Read Singh et al. (2606.26071), the task-gaming post, Anthropic emotion concepts, Wu & Tang. Chose the environment and wrote `SPEC.md` with RQ1/RQ3/RQ4 pre-registered before any data. |
| 08-21 to 08-26 | first run | 2.0 | Recon, confirmatory run, reading the all-null result and diagnosing it as underpowered rather than genuinely null. |
| 08-27 to 08-28 | exploratory | 2.5 | SAE decomposition of `tedium`, the sufficiency test, and the call to turn toward prompt versus steering. |
| 08-29 | reframe | 1.5 | B5 decouple read. Dropped the temporal-decay question for "what does the prompt actually move." |
| 08-30 to 09-01 | circuit program | 3.0 | Designed attention masking, K/V content swap, GDN state swap, per-component ablation. Caught the 26x dose mismatch and the correlated-decision-point problem. |
| 09-02 to 09-03 | final results | 1.5 | Sentence-level ablation, the recurrent channel, and the status ledger separating established from refuted. |
| 09-03 | write-up | 1.5 | Main write-up. |
| 09-04 | revision | 1.5 | Reviewed the submission against the full run data, corrected two numbers, added the disapproval dissociation and the concept-geometry check, and tightened the causal caveats. |
| | **counted** | **17.5** | |

**Exec summary allowance (+2):** ~1.5h for the executive summary and the four figures. No new
experiment code was written in that window, per the rule. Form answers are excluded entirely.

**One caveat I would rather state than hide.** A large fraction of the code was written by an
agent under my direction. The hours above are my engaged time deciding, reviewing, debugging
and interpreting, not the agent's execution time. See the LLM-use answer on the form for what
I checked and what I did not.
