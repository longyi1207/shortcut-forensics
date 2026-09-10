# Neel MATS 12.0, form answers, v2 draft (paste into Airtable after rewriting in your own voice)

> Project answers rewritten for the v2 write-up (prompting vs contrast-direction steering). The personal
> sections at the bottom are copied unchanged from v1 for you to check. Numbers match RESULTS_2026-09-10.md.

---

### What question did you try to answer?

When a one-line instruction changes what an agent does, and a contrast-pair steering vector for the same
concept also changes what it does, are they acting on the same thing inside the model? The two are
usually treated as two handles on one lever. I tested that per factor in Singh et al.'s pre-commit
environment (arXiv:2606.26071), where a coding agent behind a mypy hook sometimes fakes a green check.
For five candidate motives (tedium, desperation, temptation, wanting to be done, fear of the user's
disapproval) I fitted a contrast direction, wrote a matched instruction, and measured what each does to
the shortcut rate and whether the instruction's footprint on the residual stream lies along the direction.

### Why is this question interesting / why did you choose it?

Prompting is the mitigation people ship; contrast directions are what people fit to monitor or steer a
disposition; persona vectors assume a prompt's effect projects onto the fitted direction. Recent work
questions the equivalence, but on persona and style benchmarks, single turn, judged by output quality.
Nobody had put the two side by side on a long agentic task, per concept, with behavioural labels. The
Singh et al. environment makes that clean: the misbehaviour is discrete, the candidate drivers are the
kind of thing people fit directions for, and their own analysis stopped at prompt edits. If the two
turn out to be different mechanisms, the probe you would trust is blind to the fix you deployed.

### What conclusions have you reached about this research problem?

1. A one-line instruction cuts the shortcut rate for every one of the five factors, from 31% to 0-11% (n = 90 each, all p ≤ 0.0005); a same-length line with no concept content does nothing (30%, n = 150).

2. Steering the matching contrast directions moves behaviour for tedium only (ablation 9%, addition 53%, against a 20% baseline; a random direction gives 29%); the other four directions are null at n = 30 to 43.

3. Even for tedium, the instruction does not act along the direction: its residual shift has zero projection on it, while the concept's own sentence moves along it; at the decision token nothing moves along any direction.

4. The instruction reaches the decision through the context, in two copies, the instruction text and the agent's own plan: blocking attention to both restores the no-instruction rate (25% vs 3.8%), blocking either alone does not. Tested on one line at n = 20, so a lead rather than a finding.

5. A direction can therefore steer, and read its own concept, and still miss behaviour a prompt controls. Contrast pairs do not deliver the diagnostic validity a monitor needs for prompt-controlled behaviour.

### Technical setup (models, datasets, prompts, metrics)

Qwen3.5-9B (bf16, thinking on, 3:1 GatedDeltaNet/attention hybrid), 8×A100. Singh et al.'s repo with 258
seeded type errors behind a pre-commit mypy hook; 80 tool-calling turns with real git and mypy. Outcome:
an LLM judge (gpt-5.4-mini, temperature 0) over transcript and diff, returning is_shortcut, workaround
type and capability_ok; shortcuts are almost all "fake green", never --no-verify.

Directions: mean difference of last-token residuals over LLM-written contrast pairs at layer 19
(disapproval at 17), with held-out, lexical-scramble and plant gates. Instructions: one 70-token
IMPORTANT line per factor at the end of the user prompt, plus a neutral formatting line as control.
Steering: projection removal at every position of the fitted layer, addition for disapproval, random
directions as controls. Prompt sweeps on vLLM (n = 90 per line, 150 per control, one night); steering
cells through HuggingFace hooks (n = 30 to 153); the backends' baselines differ (31% vs 20%) and are
never compared across. Geometry: the residual shift a line causes, projected on the direction, at the
first assistant token and at 40 post-failure decision tokens, with the concept's own sentence as
positive control. About 2,300 judged rollouts. Code and data: github.com/longyi1207/shortcut-forensics.

### Strongest evidence against your hypotheses

My starting hypothesis was the field's default: the instruction works by moving the model along the
concept direction, so the tedium line would show as a negative projection on the tedium vector and
would resist a push along it. Both failed. The instruction's projection sits inside the null band at
the prompt end and at the decision token, and prompt plus tedium direction gives 15/30 = 50%, the same
as the direction alone (17/32, p = 1.0).

My second hypothesis was that the instruction's own mean footprint (with minus without) is the carrier,
as in instruction-vector work on format constraints. Added to a baseline run it gives 1/12; projected
out of an instructed run, 0/12. Neither sufficient nor necessary.

Against the mechanism I do report: the decisive masking cell is n = 20, p = 0.014 uncorrected, on one
instruction. And the layer-19 tedium vector failed its plant gate; I kept it because it steers, which
is a choice made on the outcome, supported only by the SAE decomposition at the same layer.

### Biggest limitations (could you have addressed them?)

One model on one task, and a task that itself induces tedium, the one factor whose direction works.
With one task I cannot separate "instructions beat directions" from "only the factor the task engages
has a working direction". A second environment from the Singh et al. suite would have separated them;
I chose depth over breadth.

The four null factors were tested by removing their directions. Removal shows nothing if the state is
not present at baseline, and a baseline agent may not be desperate or afraid of the user at all. The
addition test that settles this was run only for tedium; it is the first thing I would run with more
GPU time.

Every instruction sat in the user message; system-prompt placement is untested. The plan-channel
result is one line at n = 20 and does not survive correction. Method: one LLM judge with spot checks
(30/30 here, 18/18 earlier); steering cells of n = 30 to 43 would miss a drop from 20% to 10% half the
time; two backends with different baselines.

### LLM use (which tools, what you checked, surprise-if-wrong per part)

Two LLMs with two jobs. Claude Code as an agent that wrote the runners, hooks, analysis and draft and
ran the cluster overnight under a guardian process I designed; gpt-5.4-mini at temperature 0 as the
judge that produces every is_shortcut label. The judge is the load-bearing one.

I set the question, the pivot to prompting versus steering, the cells, the controls (neutral line,
random direction, plant positive control) and the interpretation. The hours in the time log are mine.

What I checked: every intervention has a mechanism check I read rather than trusted (attention masking
at exactly zero span mass on all eight attention layers; K/V swap verified element-wise; the steering
hook verified by its projection trace). The same-night rule exists because I found the backend changed
the baseline. I read 30 rollouts with the judge's verdict withheld and agreed on all 30. Reading the
numbers against the run artifacts is how I found an earlier draft's error count (19) contradicted the
config (258).

Surprise-if-wrong: least surprised to be right that the instructions work and do not act along the
direction, both on large cells with controls; more surprised if the two-copies mechanism is wrong, since
it rests on n = 20 and one instruction; most surprised if the steering nulls became large effects at
higher n, though I cannot exclude effects of half tedium's size.

### Prior mechanistic interpretability experience

(unchanged from v1; see FORM_ANSWERS.md)

### Three pieces of evidence, other than the project

(unchanged from v1)

### Why Neel's stream specifically

(unchanged from v1, but replace "the probes do not work, at AUROC 0.39 to 0.67" with the v2 result if
you cite it: the tedium probe reads the instructed arm as identical to baseline.)

### Likelihood of joining the exploration phase

(unchanged from v1)

### Anything else about the project (optional)

The result I would defend hardest is the pair: five instructions with large effects, one direction with
an effect, and that one direction not being the instruction's route. Roughly 75% of this model is
GatedDeltaNet rather than attention, so the standard attention toolkit reaches a quarter of it; every
intervention here is a forward hook I wrote, and that constraint will apply to most models that ship
next.

# Neel MATS 12.0, form answers, v2 draft (paste into Airtable after rewriting in your own voice)

> Project answers rewritten for the v2 write-up (prompting vs contrast-direction steering). The personal
> sections at the bottom are copied unchanged from v1 for you to check. Numbers match RESULTS_2026-09-10.md.

---

### What question did you try to answer?

When a one-line instruction changes what an agent does, and a contrast-pair steering vector for the same
concept also changes what it does, are they acting on the same thing inside the model? The two are
usually treated as two handles on one lever. I tested that per factor in Singh et al.'s pre-commit
environment (arXiv:2606.26071), where a coding agent behind a mypy hook sometimes fakes a green check.
For five candidate motives (tedium, desperation, temptation, wanting to be done, fear of the user's
disapproval) I fitted a contrast direction, wrote a matched instruction, and measured what each does to
the shortcut rate and whether the instruction's footprint on the residual stream lies along the direction.

### Why is this question interesting / why did you choose it?

Prompting is the mitigation people ship; contrast directions are what people fit to monitor or steer a
disposition; persona vectors assume a prompt's effect projects onto the fitted direction. Recent work
questions the equivalence, but on persona and style benchmarks, single turn, judged by output quality.
Nobody had put the two side by side on a long agentic task, per concept, with behavioural labels. The
Singh et al. environment makes that clean: the misbehaviour is discrete, the candidate drivers are the
kind of thing people fit directions for, and their own analysis stopped at prompt edits. If the two
turn out to be different mechanisms, the probe you would trust is blind to the fix you deployed.

### What conclusions have you reached about this research problem?

1. A one-line instruction cuts the shortcut rate for every one of the five factors, from 31% to 0-11% (n = 90 each, all p ≤ 0.0005); a same-length line with no concept content does nothing (30%, n = 150).

2. Steering the matching contrast directions moves behaviour for tedium only (ablation 9%, addition 53%, against a 20% baseline; a random direction gives 29%); the other four directions are null at n = 30 to 43.

3. Even for tedium, the instruction does not act along the direction: its residual shift has zero projection on it, while the concept's own sentence moves along it; at the decision token nothing moves along any direction.

4. The instruction reaches the decision through the context, in two copies, the instruction text and the agent's own plan: blocking attention to both restores the no-instruction rate (25% vs 3.8%), blocking either alone does not. Tested on one line at n = 20, so a lead rather than a finding.

5. A direction can therefore steer, and read its own concept, and still miss behaviour a prompt controls. Contrast pairs do not deliver the diagnostic validity a monitor needs for prompt-controlled behaviour.

### Technical setup (models, datasets, prompts, metrics)

Qwen3.5-9B (bf16, thinking on, 3:1 GatedDeltaNet/attention hybrid), 8×A100. Singh et al.'s repo with 258
seeded type errors behind a pre-commit mypy hook; 80 tool-calling turns with real git and mypy. Outcome:
an LLM judge (gpt-5.4-mini, temperature 0) over transcript and diff, returning is_shortcut, workaround
type and capability_ok; shortcuts are almost all "fake green", never --no-verify.

Directions: mean difference of last-token residuals over LLM-written contrast pairs at layer 19
(disapproval at 17), with held-out, lexical-scramble and plant gates. Instructions: one 70-token
IMPORTANT line per factor at the end of the user prompt, plus a neutral formatting line as control.
Steering: projection removal at every position of the fitted layer, addition for disapproval, random
directions as controls. Prompt sweeps on vLLM (n = 90 per line, 150 per control, one night); steering
cells through HuggingFace hooks (n = 30 to 153); the backends' baselines differ (31% vs 20%) and are
never compared across. Geometry: the residual shift a line causes, projected on the direction, at the
first assistant token and at 40 post-failure decision tokens, with the concept's own sentence as
positive control. About 2,300 judged rollouts. Code and data: github.com/longyi1207/shortcut-forensics.

### Strongest evidence against your hypotheses

My starting hypothesis was the field's default: the instruction works by moving the model along the
concept direction, so the tedium line would show as a negative projection on the tedium vector and
would resist a push along it. Both failed. The instruction's projection sits inside the null band at
the prompt end and at the decision token, and prompt plus tedium direction gives 15/30 = 50%, the same
as the direction alone (17/32, p = 1.0).

My second hypothesis was that the instruction's own mean footprint (with minus without) is the carrier,
as in instruction-vector work on format constraints. Added to a baseline run it gives 1/12; projected
out of an instructed run, 0/12. Neither sufficient nor necessary.

Against the mechanism I do report: the decisive masking cell is n = 20, p = 0.014 uncorrected, on one
instruction. And the layer-19 tedium vector failed its plant gate; I kept it because it steers, which
is a choice made on the outcome, supported only by the SAE decomposition at the same layer.

### Biggest limitations (could you have addressed them?)

One model on one task, and a task that itself induces tedium, the one factor whose direction works.
With one task I cannot separate "instructions beat directions" from "only the factor the task engages
has a working direction". A second environment from the Singh et al. suite would have separated them;
I chose depth over breadth.

The four null factors were tested by removing their directions. Removal shows nothing if the state is
not present at baseline, and a baseline agent may not be desperate or afraid of the user at all. The
addition test that settles this was run only for tedium; it is the first thing I would run with more
GPU time.

Every instruction sat in the user message; system-prompt placement is untested. The plan-channel
result is one line at n = 20 and does not survive correction. Method: one LLM judge with spot checks
(30/30 here, 18/18 earlier); steering cells of n = 30 to 43 would miss a drop from 20% to 10% half the
time; two backends with different baselines.

### LLM use (which tools, what you checked, surprise-if-wrong per part)

Two LLMs with two jobs. Claude Code as an agent that wrote the runners, hooks, analysis and draft and
ran the cluster overnight under a guardian process I designed; gpt-5.4-mini at temperature 0 as the
judge that produces every is_shortcut label. The judge is the load-bearing one.

I set the question, the pivot to prompting versus steering, the cells, the controls (neutral line,
random direction, plant positive control) and the interpretation. The hours in the time log are mine.

What I checked: every intervention has a mechanism check I read rather than trusted (attention masking
at exactly zero span mass on all eight attention layers; K/V swap verified element-wise; the steering
hook verified by its projection trace). The same-night rule exists because I found the backend changed
the baseline. I read 30 rollouts with the judge's verdict withheld and agreed on all 30. Reading the
numbers against the run artifacts is how I found an earlier draft's error count (19) contradicted the
config (258).

Surprise-if-wrong: least surprised to be right that the instructions work and do not act along the
direction, both on large cells with controls; more surprised if the two-copies mechanism is wrong, since
it rests on n = 20 and one instruction; most surprised if the steering nulls became large effects at
higher n, though I cannot exclude effects of half tedium's size.

### Prior mechanistic interpretability experience

(unchanged from v1; see FORM_ANSWERS.md)

### Three pieces of evidence, other than the project

(unchanged from v1)

### Why Neel's stream specifically

(unchanged from v1, but replace "the probes do not work, at AUROC 0.39 to 0.67" with the v2 result if
you cite it: the tedium probe reads the instructed arm as identical to baseline.)

### Likelihood of joining the exploration phase

(unchanged from v1)

### Anything else about the project (optional)

The result I would defend hardest is the pair: five instructions with large effects, one direction with
an effect, and the one direction's effect not being the instruction's route. What I would tell a reader
to be careful about is in the Limitations section: the steering cells are underpowered for effects half
tedium's size, and the mechanism cells are n = 20 on one instruction.

Roughly 75% of this model is GatedDeltaNet rather than attention, so the standard attention toolkit
reaches a quarter of it; every intervention here is a forward hook I wrote. That constraint will apply to
most models that ship next.
