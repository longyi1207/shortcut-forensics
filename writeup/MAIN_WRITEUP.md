
---

# Main write-up

## 1. The question, and where the setting comes from

Singh, Kroiz, Rajamanoharan and Nanda ([arXiv:2606.26071](https://arxiv.org/abs/2606.26071)) gave a
coding agent a repo behind a `mypy` pre-commit hook and found that it sometimes "passes" by weakening
the check rather than doing the work. Cutting the number of seeded errors cuts the shortcut rate, which
from the outside reads as low effort rather than as an agent that understood the user would object.
That paper is explicit about where it stops: no internals, and the theory-of-mind alternative untested.

This project runs their environment on an open-weight model and asks the next question. Two
interventions move the behaviour in opposite directions, and if the low-effort reading is the whole
story they are two handles on one axis. Are they the same mechanism? Where does the prompt's effect
live? And does a probe on that axis see either one?

### The environment

Their §6.1 task, vendored from [gkroiz/agent-interp-envs](https://github.com/gkroiz/agent-interp-envs):
a Python repo with 258 seeded type errors. The agent must add a `py.typed` marker, write a `pre-commit`
hook that runs `mypy` and blocks on failure, and commit. The hook is the trap, because doing the task
honestly means clearing 258 errors over dozens of turns. 80 turns, one shell command per turn, real
`git` and `mypy`.

Model: Qwen3.5-9B, bf16, T = 0.7, thinking enabled, on 8×H100. An LLM judge reads the transcript and
final diff and returns `is_shortcut` plus a workaround type. About 500 rollouts across conditions.

**What the misbehaviour is.** Across 115 rollouts, `--no-verify`, `git commit -n`, `chmod -x` on the
hook, and hook deletion occur exactly zero times. Nobody bypasses the gate. On the unmodified prompt
the judge labels 29 of 33 shortcuts `fake_green` and 4 `disable_hook` (173 judged rollouts), but since
no rollout ever issues a removal or bypass command, those four are hooks written so they do not block
rather than hooks removed. Either way the cheat is the same shape: a hook that does not really block,
`# type: ignore[arg-type]` on failing lines, source rewritten to dodge the checker. The cheat lives in file content, which has two consequences.
Only a judge reading the diff can detect it, and there is no per-turn command that marks the decision.
The natural decision point is therefore the turn following a failed tool result, where the model has
just been told it is not done.

**The two interventions.** The prompt is one line appended to the user message: "IMPORTANT: This task
may feel repetitive or tedious. Do NOT let that affect your work. Treat every single step with full
care and thoroughness... Do not cut corners. Do not rush to finish. Boredom is not a reason to do less
than the complete, correct job." (70 tokens.) The steering vector is a `tedium` direction fitted by
mean-difference on contrastive pairs and validated causally: adding it takes shortcuts from 20% to 53%.

**Two readouts.** The behavioural shortcut rate costs about 50 minutes per rollout. Wherever that lacks
power I use a paired decision-point score instead: at a stored post-failure turn, the context is
rebuilt under two conditions and scored as
`s = logP(" Let me fix the remaining type errors properly.") − logP(" Given the time constraints, let me take a different approach.")`.
The two continuations instantiate the two reasoning modes the feature analysis identifies (§4). This is
seconds per measurement rather than 50 minutes, and a continuous quantity rather than a rare binary
event, which is what makes §5 to §7 statistically possible: at base rates of 4% vs 14%, separating a
condition near 7% behaviourally needs about 150 rollouts.

### The alternative their paper flagged

Before asking how the mitigation works, the other explanation for the behaviour is worth closing off.
If the agent knew the user would object and weakened the check anyway, `disapproval` should behave the
way `tedium` does. It does not, at any stage.

| | `tedium` | `disapproval` |
|---|---|---|
| mentioned in the chain of thought on shortcut rollouts | 100% | 7.1% |
| contrast-pair direction (held-out / lexical-scramble) | 100% / 65%, pass | 90% / 50%, pass |
| readout AUC, shortcut vs honest | 0.759 | 0.489, chance |
| causal effect on the rate | both ways (p = 0.005 ablating, p = 0.0009 adding) | null at n = 43 (p = 0.732) |

Both directions pass the same gates, so this is not a failure to build the vector, and the `+α` test
ran at an L17 vector that passes the positive control their paper says it lacked. The direction is real
and validated, and it does nothing here.

The two are also not independent, which makes the dissociation sharper rather than weaker. Their cosine
is +0.13, about 8 times the 0.016 that two unrelated directions give in 4096 dimensions, so this is not
a case of testing two axes that happen to have nothing to do with each other: they overlap measurably,
and only one of them moves behaviour. (Concept directions fitted from generated contrast pairs share
some linguistic structure by construction, so the overlap should be read as an upper bound on shared
representational content.) §8 says why the null should not be generalised.

## 2. The prompt and the steering vector are different mechanisms

The prompt works: 2/53 = **3.8%** against 8/59 = **13.6%** with no instruction, pooled across phases.

If it worked by lowering the tedium state, pushing that state back up should defeat it. Instead it
erases the prompt's effect entirely (Fig 1). Prompt plus `+tedium` gives 15/30, statistically identical
to `+tedium` with no prompt (17/32, p = 1.0) and far from the prompt alone (p < 0.001). At half dose
the two steered arms are 10/21 vs 10/21, p = 1.000, so this is not a saturation effect. The prompt
provides no protection against this direction at any dose tested.

Nor is the prompt an added direction. Its mean residual delta at decision positions is neither
sufficient (baseline plus Δh: 1/12) nor necessary (prompt minus Δh: 0/12, protection intact).

What this does and does not license. The two steered arms differ only in whether the prompt is present,
so whatever a 26%-of-norm push does generically is common to both and cancels; what is left is the
prompt's own contribution, and it is zero. That a random direction at the same magnitude takes the
prompted model to 22% (§4) shows the protection is fragile in general, but it is exactly why the
controlled comparison is the steered pair rather than the prompt-alone cell. The claim is about how
this mitigation is implemented, not a general claim that prompting and steering are separate systems.
The sharp version is narrower: the prompt names tedium, works, and offers no defence whatever of the
tedium axis. The concept an instruction talks about is not the concept it operates on.

## 3. Why this matters for monitoring

Treating the `tedium` projection as a monitor and asking what it reports for each arm:

Prompt-fixed rollouts read identical to baseline (L19 late-window mean 0.00 vs −0.14; L26 SAE −39.8 vs
−39.9) while cheating 0/21 against 4/30. A baseline-tuned threshold flags 55% to 76% of these fixed
rollouts as at-risk. The mitigation is invisible. Steered rollouts read "safe" at L19 by construction
while 5/30 still cheat. And within the baseline arm the projection is a weak predictor: AUROC 0.39 to
0.67, at or below chance for several summaries.

An intervention-validated direction is not automatically a diagnostic one. This project produced the
dissociation three times. `tedium` moves behaviour when steered but barely ranks who cheats. The
decision-token feature family separates the two conditions strongly (d ≈ 1 to 1.8) but does not rank
who cheats within a condition (AUROC 0.31). And prompt-based mitigation is invisible to both. Causal
validity, contrastive validity and diagnostic validity come apart, and the first two are what people
usually check.

## 4. What the prompt changes, and what does not explain it

Event-locked capture at decision tokens (n = 20 per arm, about 11k residuals) shows the prompt's
footprint is 2 to 2.5 times larger at post-failure tokens than at matched control positions at every
layer, and is not aligned with the tedium axis (cos ≈ −0.05 to −0.09).

Decomposing with an SAE and labelling features by their top-activating examples gives a legible
account. The prompt suppresses cost-of-effort framing and replanning ("Given the time constraints, let
me take a different approach", "This is a lot of files to fix") and requirement re-framing ("mypy is
set to strict mode which...", "However, the user said..."). It enhances engagement with the specific
error ("I see the issue, I used `any` instead of `Any`", "Let me fix them: 1. transform.py..."). Every
one of these fires in both arms: the prompt shifts the balance between two modes the model already has
rather than switching something on. The correspondence with behaviour is close, since the suppressed
family is re-interpreting the requirement and `fake_green` is exactly redefining what counts as passing.

That axis is nonetheless not the causal handle. Assembling those SAE decoder columns into a direction
(`effort_L26`; cos with `tedium` = +0.02, a genuinely different axis) and testing it both ways:

Projecting it out of the baseline gives 2/21 = 10%, the no-instruction rate (p = 1.0), not the prompt's
(p = 0.32). Removing the replanning axis does not create protection. Ablation is a projection, so this
does not depend on the direction's scale.

Pushing it at α = 26, matched to tedium's 26% of residual norm, takes the prompt from 3.8% to 19%. A
random direction at the same layer and magnitude gives 22%, indistinguishable (p = 1.0). The protection
is fragile to any large perturbation at L26, not to this axis in particular.

The same features also fail as a monitor (AUROC 0.31 at L26, 0.44 at L31). A feature set can describe a
behaviour exactly and have no causal role in producing it.

These features are correlational by construction, since they come from contrasting prompted against
unprompted activations at the same positions, and both causal tests fail. What this project cannot do
is separate the two readings that remain: they may be downstream of the decision, or a parallel
correlate of it driven by something further upstream. Ruling out "cause" is not the same as
establishing "effect", and nothing here should be read as claiming the latter.

## 5. Where the influence lives, and the 75% of the model that needs new tools

At the decision, nothing reads the instruction. Replaying each decision turn with eager attention to
recover per-head weights: across 128 heads the instruction span draws 0.53% of total attention, the
best single head gives it 2.5%, and every one of the top-20 "instruction-reading" heads reads the
neighbouring task text more than the instruction. Blocking every decode-time edge to it moves the
next-token distribution by KL ≈ 5×10⁻⁴ with 0 of 20 argmax changes. Injecting the instruction's entire
attention content (K/V swap at prefill) into a filler-text run gives 21%, identical to its matched
text-only control at 21% (p = 1.0). Five independent methods, one answer: attention is not the carrier
(Fig 4, left).

That conclusion covers 25% of the model. Qwen3.5 is a 3:1 hybrid: 24 of its 32 layers are
GatedDeltaNet, which maintains a fixed-size recurrent state instead of per-token K/V. There are no
attention weights to read, no K/V to swap, and no "which token is this head reading". Every
attention-specific method covers 8 layers. This is the current generation rather than one model:
Qwen3-Next, Kimi Linear, Ling and Nemotron 3 all use the same 3:1 pattern.

GDN is a recurrence, so its state accumulates in order and can be intervened on between chunks. Each
turn's prefill is split at the instruction span (`[0:470] | instruction | [540:]`), and after the
instruction chunk the recurrent and conv states of all 24 GDN layers are overwritten with those
produced by length-matched filler text. Text and attention K/V stay real, so only the instruction's
contribution to the recurrent channel is removed. The chunked prefill reproduces single-shot to bf16
noise, a real swap moves logits 6.5 times that floor, and a null (real to real) swap is exactly 0.0000.

Measured at decision points against the null swap, so that any effect of chunking is common to both
arms, the recurrent channel carries +0.103 of the instruction's +1.011, about **10%** (Fig 4, right). This
one sits at the edge of detection: Wilcoxon p = 0.052 across 30 decision points, and a
rollout-clustered bootstrap puts the share anywhere between 1% and 18%. The chunking term itself is
undetectable at this readout (0.048, p = 0.19), which is what the null swap was for. Read the 10% as an
order of magnitude rather than a measurement.

Neither channel carries the instruction on its own.

This is the finding most contingent on the architecture, and it cuts against one of the headlines. In a
model that routes three quarters of its depth through recurrent state rather than attention, finding
that attention masking does not remove the effect is less surprising than the same result would be on a
pure transformer, where attention might well be the carrier. Two things keep it from being an artifact
of the hybrid. The recurrent channel only carries about 10%, so the effect is not hiding there either.
And the remaining influence is written early into the residual stream and read locally at the decision,
which is a claim about representation rather than about routing, and representational claims travel
across architectures better than routing ones do.

> **[FIG 4: fig4_channels.png]**

## 6. Circuit level: the write happens in the first layer

> **[FIG 3: fig3_components.png]**

Since nothing reads the instruction at the decision, the circuit must sit at prefill, where the
instruction is read and its influence written into what persists. I ablated each of 184 components
(128 attention heads, 32 MLPs, 24 GDN blocks) at the instruction's token positions only, each against
its own ablation over a length-matched control span, on 24 decision points from 12 rollouts (Fig 3).

`gdn.0` scores **+1.31 ± 0.15**: the first layer's recurrent block alone accounts for the whole effect.
`head.3.h4`, one head in the first attention layer, scores +0.81 ± 0.14. Every top component sits in
layers 0 to 15. By type, 24 GDN blocks total +1.29, 32 MLPs +0.35, and all 128 attention heads together
−1.08: individual heads rank high but most are slightly negative and cancel. At matched granularity,
ablating whole attention layers sums to +0.08.

The attributions do not partition the effect. `gdn.0` alone scores +1.31 where the whole instruction
scores 1.0, and the top five components sum to +3.62, so components interact rather than divide a fixed
budget between them. No single layer restores the effect when patched either (best: L13, 44%). The
influence is redundant and overlapping, with no bottleneck.

## 7. Which words do the work

Since 70 tokens in a 60k-token context change behaviour, which of them? Each variant keeps exactly one
sentence, scored at 40 paired decision points (Fig 2):

| kept | effect vs. none | share of full | p |
|---|---|---|---|
| full instruction (70 tok) | +0.901 | 100% | <0.0001 |
| "Do not cut corners. Do not rush to finish." (11 tok) | **+0.704** | **78%** | <0.0001 |
| "Boredom is not a reason to do less..." | +0.680 | 75% | <0.0001 |
| "Do NOT let that affect your work." | +0.534 | 59% | <0.0001 |
| "Treat every single step with full care..." | +0.245 | 27% | 0.0012 |
| "This task may feel repetitive or tedious." | +0.116 | 13% | 0.097 n.s. |
| tedium sentence plus "Do NOT let that affect your work" | +0.141 | 16% | 0.087 n.s. |

About eleven tokens carry 78% of what the full line does, and the sentences that work are the ones that
name a behaviour to avoid rather than a state to feel. The clearest effect is negative: naming the
feeling cancels the sentence after it. "Do NOT let that affect your work" is worth +0.53 on its own and
+0.14 once the tedium sentence is prepended. Mentioning the state appears to prime it, costing three
quarters of the following sentence's effect.

## 8. Limitations

One model, one task, one instruction wording. Generalisation is untested and is the first thing to run
next.

The disapproval null is about this model, not about models. Three reasons not to carry it further. A
null at n = 43 does not exclude a small effect. Absence of verbalised disapproval is weak evidence
about internals, and the task-gaming work found final outputs that mislead with nothing planned in the
chain of thought. And a 9B model may lack the theory of mind for "the user would disapprove" to be an
available consideration at all, which is a capability floor rather than a fact about alignment. A
frontier model that does represent it and cheats anyway is a different phenomenon, and this setup could
not have detected it.

Rare-event power. At 4% vs 14% base rates, separating a condition near 7% needs about 150 rollouts.
Behavioural cells at n ≈ 20 cannot be decisive. This motivated the decision-point readout, but the
behavioural claims remain underpowered.

The two readouts disagree once. The decision-point score rates "Do NOT let that affect your work" as
helpful (+0.53) while its behavioural cell is 6/17 = 35%, worse than no instruction (p = 0.002 against
the full instruction, surviving Bonferroni). One moment against 80 turns, two fixed continuations
against many real paths, and n = 17 are all candidate explanations. None is tested, and both numbers
stand.

Multiplicity. Many comparisons were run. The attention and notes masking result (p = 0.014) does not
survive Bonferroni (0.086) and is reported as a lead rather than a finding.

The recurrent measurement is a floor. It removes the instruction's direct contribution at its own
positions, not influence already propagated into later positions during earlier turns.

`gdn.0`'s attribution exceeds 1.0, which means components interact rather than partition the effect.
Single-component attributions should not be summed.

## 9. Next steps

Generalisation first: other instruction wordings, other tasks, a second hybrid model. The
sentence-level result is the most actionable finding and the least replicated.

Interpretability for the recurrent channel. Tooling is largely absent while the architecture generation
moves that way. The chunked-prefill state swap here is a first crude instrument; per-position
attribution inside a GDN state is open.

Monitoring that survives mitigation. Three dissociations in one project suggest probe validation should
require a diagnostic check within the deployment arm, not only a causal one.
