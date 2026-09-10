# Neel MATS 12.0, form answers, v2 draft (paste into Airtable after rewriting in your own voice)

> Project answers for the v2 write-up (prompting vs contrast-direction steering). Personal sections at the
> bottom are unchanged from v1. Numbers match writeup/RESULTS_2026-09-10.md.

---

### What question did you try to answer?

If a one-line instruction stops an agent from cheating, and a steering vector for the same concept also
stops it, are they pulling the same lever inside the model? People usually assume yes. I checked, one
concept at a time, in Singh et al.'s pre-commit environment (arXiv:2606.26071): a coding agent behind a
mypy hook that sometimes fakes a green check. For five motives (tedium, desperation, temptation, wanting
to be done, fear of the user's disapproval) I fitted a contrast direction, wrote a matching instruction,
and measured what each does to the cheating rate and whether the instruction moves the model along the
direction at all.

### Why is this question interesting / why did you choose it?

Prompting is what people ship. Contrast directions are what people fit when they want to steer or
monitor a disposition. Persona vectors assume the two line up. That assumption had been questioned on
style benchmarks in single turns, never on a long agentic task with a real behaviour to label. Singh et
al.'s environment gives exactly that, and their own analysis stopped at prompt edits. If the two are
different mechanisms, the probe you trust is blind to the fix you deployed.

### What conclusions have you reached about this research problem?

1. Every one of the five instructions cuts the cheating rate, from 31% to 0-11% (n = 90 each). A
same-length line with no content does nothing (30%, n = 150).
2. Steering works for one direction, tedium (ablate 9%, add 53%, baseline 20%, random direction 29%).
The other four are null at n = 30 to 43.
3. Even for tedium, the instruction does not move the model along the direction. The concept's own
sentence does; the instruction has zero projection, at the prompt and at the decision.
4. The instruction reaches the decision through the context, twice over: the instruction text and the
plan the agent wrote from it. Block both and cheating returns (25% vs 3.8%); block either and it does
not. One line, n = 20, so a lead.
5. A direction can steer, and read its own concept, and still not see behaviour a prompt controls.
That is the property a monitor needs, and contrast pairs do not give it.

### Technical setup (models, datasets, prompts, metrics)

Qwen3.5-9B, thinking on, on 8×A100. Singh et al.'s repo: 258 seeded type errors, a pre-commit mypy
hook, 80 tool turns with real git. An LLM judge (gpt-5.4-mini, temperature 0) labels each rollout
shortcut or not; the shortcuts are fake greens, never --no-verify. Directions are mean differences of
last-token residuals over LLM-written contrast pairs at layer 19, with held-out, scramble and plant
gates. Instructions are one 70-token IMPORTANT line at the end of the user prompt; the control is a
same-length line about shell formatting. Prompt sweeps ran on vLLM (n = 90 per line, 150 per control);
steering ran through HuggingFace hooks (n = 30 to 153); the two backends have different baselines and
are never compared across. About 2,300 judged rollouts. Code and data:
github.com/longyi1207/shortcut-forensics.

### Strongest evidence against your hypotheses

I expected the instruction to work by moving the model along the concept direction. It does not: its
projection is zero, and pushing the tedium direction on top of the instruction gives 50%, the same as
pushing without it (53%). I then expected the instruction's own mean footprint to be the carrier, as
in instruction-vector work. Adding it to a baseline run gives 1/12; removing it from an instructed run
gives 0/12. Neither sufficient nor necessary. Against my own mechanism result: the decisive cell is
n = 20, p = 0.014 uncorrected, one instruction. And the tedium vector I report failed its plant gate;
I kept it because it steers, which is a choice made on the outcome.

### Biggest limitations (could you have addressed them?)

One model, one task, and a task that itself makes the agent tedious, which is the one factor whose
direction works. I cannot tell "instructions beat directions" apart from "only the factor the task
engages has a usable direction"; a second environment would have, and I chose depth over breadth.
The four null directions were only removed, never added; if the state was never on, removal shows
nothing. That addition test is the first thing I would run. All instructions sat in the user prompt.
One judge, spot-checked by hand (30/30). Steering cells of 30 to 43 would miss a drop from 20% to 10%
half the time.

### LLM use (which tools, what you checked, surprise-if-wrong per part)

Claude Code wrote the runners, hooks, analysis and draft, and ran the cluster overnight under a guardian
I designed. gpt-5.4-mini is the judge behind every label. I set the question, the pivot to prompting
versus steering, the cells, the controls and the reading of the results; the hours in the time log are
mine.

What I checked: each intervention has a mechanism check I read myself (attention masks at exactly zero
span mass; K/V swaps verified element-wise; steering verified by its projection trace). I read 30
rollouts with the judge's verdict hidden and agreed on all 30. Reading numbers against the run
artifacts caught a wrong error count in an earlier draft (19 against 258 in the config).

Surprise if wrong: least surprised about the instructions working and not acting along the direction,
both from large cells with controls. More surprised if the two-copies mechanism is wrong (n = 20, one
line). Most surprised if the steering nulls turned into large effects at higher n.

---

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

The result I would defend hardest is the pair: five instructions with big effects, one direction with
an effect, and that direction not being the instruction's route. Three quarters of this model is
GatedDeltaNet, so the standard attention toolkit reaches a quarter of it; every intervention here is a
forward hook I wrote, and that will be true of most models that ship next.
