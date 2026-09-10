# Neel MATS 12.0, form answers, v2 draft (paste into Airtable after rewriting in your own voice)

> Project answers rewritten for the v2 write-up (prompting vs contrast-direction steering). The personal
> sections at the bottom are copied unchanged from v1 for you to check. Numbers match RESULTS_2026-09-10.md.

---

### What question did you try to answer?

When a one-line instruction in the prompt changes what an agent does, and a contrast-pair steering
vector for the "same" concept also changes what it does, are they acting on the same thing inside the
model? Steering vectors get read as "the model's representation of X" and prompts as a way of turning
X on or off, so the two are usually treated as two handles on one lever. I tested that per factor, on
a behaviour that is an action with a ground-truth label rather than a style.

The setting is Singh, Kroiz, Rajamanoharan and Nanda's pre-commit environment (arXiv:2606.26071): a
coding agent in a repo with 258 seeded `mypy` errors behind a hook, which sometimes fakes a green check
instead of doing the work. For five candidate motives (tedium, desperation, temptation, wanting to be
done, fear of the user's disapproval) I fitted a contrast direction, wrote a matched instruction, and
measured three things: what the instruction does to the shortcut rate, what steering the direction does
to it, and whether the instruction's footprint on the residual stream lies along the direction at all.

### Why is this question interesting / why did you choose it?

Because both interventions are in daily use and the assumption that they are interchangeable has not
been checked where it matters. Prompting is the mitigation people ship; contrast directions are what
people fit when they want to monitor or steer a disposition; persona vectors are built on the premise
that a prompt's effect projects onto the fitted direction. Recent work questions the equivalence from
the steering side (steered activations reach states no prompt reaches; prompt effects are token
specific), but on persona and style benchmarks, single turn, judged by output quality. Nobody had put
the two side by side on a long agentic task, per concept, with behavioural labels.

The Singh et al. environment makes the test clean: the misbehaviour is discrete, the candidate drivers
are exactly the kind of thing people fit directions for, and their paper argued from prompt edits alone,
so the internal side was open.

The monitoring consequence is what made me finish it. If the two are different mechanisms, the probe
you would trust is blind to the fix you deployed.

### What conclusions have you reached about this research problem?

1. Every one of the five instructions cuts the shortcut rate from 31% to between 0% and 11% (n = 90
each, all p ≤ 0.0005), and a same-length line with no concept content leaves it at 30% (150 vs 150,
p = 1.0). The instruction does not make the agent commit honestly; it makes it not go for the commit at
all and keep fixing errors (commit attempts fall from 32% of rollouts to 0 to 21%).

2. Steering on the five fitted directions moves behaviour for tedium only: ablation 9% vs a 20%
baseline (p = 0.026), addition 53% (p = 0.001), while a random direction ablated the same way gives 29%
(p = 0.003 against tedium ablation). The other four directions and their re-fits at other layers are
null at n = 30 to 43.

3. The instruction does not act along the direction, even for tedium. At the position right after the
prompt, the concept's own sentences move the residual along the direction (+0.85 to +2.4 in
unit-direction coordinates against null bands of ±0.2 to ±0.5) and the instruction does not (−0.25 to
+0.38). At the decision token 10 to 40 turns later nothing moves along any direction (cosines within
±0.04), while the instruction still changes what the agent does there. Two of the five directions are
not readable even by their own concept sentences, which matches their steering nulls.

4. The instruction reaches the decision through the context, in two copies. At the decision the agent
gives the instruction 0.5% of its attention, blocking that edge changes the next-token distribution by
KL ≈ 5×10⁻⁴, and the instruction's mean residual footprint is neither sufficient (1/12) nor necessary
(0/12). But blocking attention to the instruction and to the agent's own earlier text together, after
each failed check, returns the rate from 3.8% to 25% (p = 0.014, n = 20), while blocking either alone
does not (10%, 15%) and blocking a same-size span of tool output does not (10%). The agent reads the
instruction once, writes it into its plan, then follows the plan. Adding the tedium direction on top
of the instruction gives 50%, the same as the direction alone: the direction overrides that channel.

5. So a direction can be causally valid (it steers), contrastively valid (its own sentences move it)
and still diagnostically useless: the tedium probe reads an instructed agent as identical to an
uninstructed one while it cheats a fifth as often. Monitors need the third property, and contrast
pairs do not deliver it for prompt-controlled behaviour.

### Technical setup (models, datasets, prompts, metrics)

Qwen3.5-9B (bf16, thinking on, 32 layers, 3:1 GatedDeltaNet/attention hybrid) on 8×A100 80GB. The
environment is Singh et al.'s repo with 258 seeded type errors, a pre-commit hook running `mypy`, 80
tool-calling turns, real git and mypy. The outcome measure is an LLM judge (gpt-5.4-mini, temperature
0) over the transcript and final diff, returning `is_shortcut`, the workaround type and
`capability_ok`. Shortcuts are almost all "fake green" (ignores, weakened config); `--no-verify` never
occurs.

Directions: mean difference of last-token residuals over LLM-written contrast pairs, layer 19 (disapproval
re-fit at 17), with held-out separation, lexical-scramble and plant gates. Instructions: one 70-token
"IMPORTANT:" line per factor at the end of the user prompt, same register and length, plus a neutral
formatting line as control. Steering: projection removal (ablation) at every position of the fitted
layer for the pro-shortcut concepts, addition for disapproval, plus random-direction controls.

Prompt sweeps ran on vLLM (six servers, 10 rollouts each; n = 90 per line, 150 per control, one night);
steering cells ran through HuggingFace forward hooks (n = 30 to 153). The two backends have different
baseline rates (31% vs 20%), so nothing is compared across them. Geometry: the residual shift caused by
a line, projected on the direction, at the first assistant position and at 40 post-failure decision
tokens, with the concept's plant sentences as positive control. Mechanism cells (attention masking,
K/V swap, footprint add/remove) are from the earlier part of the project, judge-labelled, n = 12 to 53.
About 2,300 judged rollouts in total; 3,271 rows on disk.

### Strongest evidence against your hypotheses

My starting hypothesis was the field's default: that the instruction works by moving the model along
the direction the concept names, so that the tedium instruction would show up as a negative projection
on the tedium vector and the tug-of-war would show the prompt resisting the steer. Both failed. The
instruction's projection sits inside the null band at the prompt end and at the decision token, and
prompt plus tedium direction gives 15/30 = 50%, identical to the direction alone (17/32, p = 1.0). The
prompt offers no resistance because it never set the coordinate the steer moves.

The second hypothesis I held for a while was that the instruction's own mean footprint (with minus
without) is the carrier, as in instruction-vector work on format constraints. Added to a baseline run
it gives 1/12; projected out of an instructed run, 0/12. Neither sufficient nor necessary.

Against the mechanism I do report: the decisive masking cell is n = 20 with one uncorrected p = 0.014
(0.086 after Bonferroni), on one instruction only. It is a lead with a control, not a finding.

### Biggest limitations (could you have addressed them?)

One model, one task, one judge. The steering cells (n = 30 to 43) cannot rule out effects smaller than
a 20% to 10% drop, so "null" means "smaller than tedium's", not zero. I could have traded the N = 90
prompt cells for larger steering cells; I chose the prompt side because it is the half nobody had
measured.

The instructions and the steers are not dose-matched, and the instructions are mine; a different
wording might act differently, though the neutral control shows length and register alone do nothing.

The judge: 113 of tonight's verdicts came from transcripts with tool outputs shortened to fit the
judge's token quota. Which rows this hit was decided by judge-server luck, not by condition.

The readout I built for finer attribution (log-odds of an engage-vs-replan sentence at post-failure
turns) turned out not to track the judge labels (AUROC 0.56 to 0.59). I withdrew every component-level
and sentence-level result that rested on it. That cost most of the earlier circuit work and it is the
thing I would have caught earlier with a validation step I should have run first.

### LLM use (which tools, what you checked, surprise-if-wrong per part)

Two LLMs with two jobs. Claude Code as an agent writing runners, hooks, analysis and the draft, and
running the cluster overnight under a guardian process; gpt-5.4-mini at temperature 0 as the judge that
produces every `is_shortcut` label. The judge is the load-bearing one.

I directed the research: the question, the pivot to prompting vs steering, which cells answer it, which
controls are needed (the neutral line, the random direction, the plant positive control), and the
interpretation. The overnight run was agent-executed under a plan I set; the hours in the time log are
mine.

What I checked. Every intervention has a mechanism check I read rather than trusted: attention masking
verified as exactly zero span mass on all eight full-attention layers; K/V swap verified element-wise;
the steering hook verified by the projection trace (added component present at the fitted norm). The
same-night rule for vLLM cells exists because I found the backend changed the baseline. The judge was
spot-checked by hand on 18 cleanly attributable rollouts (18/18 agreement, precision on positives
10/10), which is reassuring rather than tight. Reading the numbers against the run artifacts is how I
found that a readout used for circuit attribution did not track behaviour, and that an earlier draft's
seeded-error count (19) contradicted the config (258).

Surprise-if-wrong. I would be least surprised to be right that the instructions work and that the
instruction does not act along the fitted direction; both rest on large cells with controls. I would be
more surprised if the "two copies in context" mechanism turned out wrong, because it rests on n = 20
cells and one instruction. I would be most surprised if the steering nulls turned into large effects at
higher n, but I cannot exclude effects of half tedium's size.

---

### Prior mechanistic interpretability experience

(unchanged from v1; see FORM_ANSWERS.md)

### Three pieces of evidence, other than the project

(unchanged from v1)

### Why Neel's stream specifically

(unchanged from v1, but replace "the probes do not work, at AUROC 0.39 to 0.67" with the v2 numbers if
you cite them: the tedium probe reads the instructed arm as baseline; the decision-point readout is at
AUROC 0.56 to 0.59.)

### Likelihood of joining the exploration phase

(unchanged from v1)

### Anything else about the project (optional)

The result I would defend hardest is the pair: five instructions with large effects, one direction with
an effect, and the one direction's effect not being the instruction's route. Most of what I would tell a
reader to be careful about is in §7 of the write-up: I built a cheap readout to make circuit-level
attribution feasible, used it for a month, and it turned out not to measure the behaviour. The v2
write-up is what survives without it.

Roughly 75% of this model is GatedDeltaNet rather than attention, so the standard attention toolkit
reaches a quarter of it; every intervention here is a forward hook I wrote. That constraint will apply to
most models that ship next.
