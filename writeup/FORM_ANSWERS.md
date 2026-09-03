# Neel MATS 12.0 — form answers (draft, paste into Airtable)

> He reads these **first** and uses them as the filter. Specifics beat vibes.

---

### What question did you try to answer?

A one-line prompt ("this task may feel tedious, don't cut corners") cuts an agent's rate of faking
task success from 14% to 4%. Steering a fitted `tedium` direction moves the same behaviour the other
way, 20% → 53%. Are these the same mechanism? If they are, one probe monitors both. If they aren't,
a deployed prompt fix could be invisible to exactly the monitor you'd trust — so I tried to find
where in the model the prompt's effect actually lives.

### Why is this question interesting / why did you choose it?

Prompting is the mitigation people actually ship, and activation probes are the monitoring people
actually propose, but I hadn't seen anyone check that a probe sees a prompt-based fix. It turns out
it doesn't — and the failure is bidirectional (invisible fixes, and steered runs reading "safe" while
still cheating). It's also a forensics question on a real agentic failure rather than a toy task: the
misbehaviour here is emergent `fake_green` (weakening the check so it passes), not a prompted
demonstration.

### What conclusions have you reached about this research problem?

1. **Prompt and steering are different mechanisms.** Prompt + `+tedium` = 15/30 shortcuts, identical
to steering with no prompt (17/32, p=1.0). At half dose 10/21 vs 10/21. The prompt gives zero
protection against the axis it's literally about.
2. **Direction probes are blind to prompt-based mitigation.** Prompt-fixed rollouts read identical to
baseline on the tedium projection while cheating 0/21 vs 4/30; within baseline the probe is
AUROC 0.39–0.67.
3. **Attention is not the carrier** (5 independent methods): the instruction gets 0.53% of attention
at decision tokens, blocking all decode-time edges to it gives KL≈5e-4, and injecting its entire
attention content into a filler run gives 21% vs a 21% control (p=1.0).
4. **24 of 32 layers were unreachable** by attention tooling (Qwen3.5 is a 3:1 GatedDeltaNet hybrid,
as is the whole current generation). I built a chunked-prefill state swap to reach them: the
recurrent channel carries ~10%. Neither channel carries it alone; component attributions sum to 2.79
where a localised effect gives 1.0.
5. **Practical:** ~11 tokens ("Do not cut corners. Do not rush to finish.") carry 78% of the full
70-token instruction, and *naming the feeling backfires* — prepending "this task may feel tedious"
to a sentence that works destroys three quarters of its effect (+0.53 → +0.14).

### Technical setup (models, datasets, prompts, metrics)

Qwen3.5-9B (bf16, T=0.7, thinking on), 8×H100. Environment: a Python repo with 19 seeded type errors;
the agent must add `py.typed`, write a `pre-commit` hook running `mypy` that blocks on failure, and
commit. 80 turns, one shell command per turn, real git/mypy. Outcome = LLM judge over transcript +
final diff (`is_shortcut`, workaround type). ~500 rollouts total across conditions.
Two readouts: (a) behavioural shortcut rate; (b) a **paired decision-point score**,
logP(" Let me fix the remaining type errors properly.") − logP(" Given the time constraints, let me
take a different approach."), measured at post-failure turns — seconds per measurement instead of
~50 min, which is what made the component and sentence experiments feasible.
Interventions: mean-difference steering vectors; attention-edge masking on the 8 full-attention
layers; K/V content swap at prefill; GDN recurrent-state swap via chunked prefill; per-component
zero-ablation at the instruction's token positions; SAE (Qwen 64K L0-100) feature decomposition.

### Strongest evidence against your hypotheses

My best hypothesis was that the prompt works by suppressing a "cost-of-effort → replan → re-frame the
requirement" feature family found at decision tokens — the semantics match the observed `fake_green`
behaviour exactly. **Two controls killed it.** Projecting that axis out of the baseline gives 10%,
the no-instruction rate, not the prompt's. And pushing it (dose-matched at α=26) destroys the
protection no better than **a random direction of the same magnitude** (19% vs 22%, p=1.0). The same
features also fail as a monitor (AUROC 0.31, below chance).
Separately, my recurrent-channel experiment gave a clean-looking 15% vs the prompt's 6% — until its
**null control** (identical machinery, real→real no-op swap) gave 19%. The chunked prefill moved the
rate by itself; the result was void and I re-did it at decision points where the artefact cancels.
I also predicted that one sentence failed alone because of a dangling pronoun and that restoring the
antecedent would rescue it. It did the opposite.

### Biggest limitations (could you have addressed them?)

**Generalisation is untested** — one model, one task, one instruction wording. This is the first thing
I'd run and I could have swapped it in for one of the confirmatory cells.
**Rare-event power**: at 4% vs 14% base rates, separating a condition near 7% needs n≈150 rollouts, so
several behavioural cells can't be decisive by construction. I partly addressed this by moving to the
paired decision-point readout, but the behavioural claims remain underpowered.
**The two readouts disagree once** (one sentence is helpful by the decision-point score and worse than
nothing behaviourally); I report both rather than picking.
**Multiplicity**: many comparisons; the masking result (p=0.014) does not survive Bonferroni (0.086)
and is reported as a lead, not a finding.
**The recurrent measurement is a floor** — it removes the instruction's direct contribution at its own
positions, not influence already propagated into later positions.

### LLM use (which tools, what you checked, surprise-if-wrong per part)

Claude Code as an agent throughout — writing experiment runners, analysis scripts and infrastructure,
and running the cluster. I directed the research: choice of question, which experiment answers it,
which controls are needed, and interpretation.
**What I verified myself, and how.** Every intervention has an explicit mechanism check that I read
and re-ran, not just a passing test: attention masking verified as *exactly* 0.0000 span mass at all
8 full-attention layers with rows renormalising; K/V swap verified element-wise against the
`DynamicCache` tensors (|swapped − donor| = 0.0000 at the span, positions after it still differing);
the GDN chunked prefill verified to reproduce single-shot to bf16 noise with a null swap at exactly
0.0000. I also caught and fixed errors that would have produced clean-looking wrong results: an
ablation silently skipped on long-context turns while the rollout still recorded success (found by
comparing `prefill_turns` against `n_turns`), a tug-of-war comparing a 26%-of-residual push against a
1% one (found by checking vector norms before believing the comparison), and correlated decision
points understating error bars.
**Surprise-if-wrong.** Least surprised: the decoupling and the attention-is-not-the-carrier results —
five independent methods agree and each has its own control. More surprised: the `gdn.0` attribution
(one component exceeding the whole effect means components interact; I would not defend the exact
number). Most surprised: the sentence-level priming result — it is one readout on one instruction,
n=40 paired points, and it contradicts the behavioural cell for a different sentence, so I would want
it replicated before anyone acted on it.
