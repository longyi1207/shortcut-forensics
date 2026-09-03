# Neel MATS 12.0, form answers (draft to paste into Airtable)

> He reads these first and uses them as the filter. Specifics beat vibes.

---

### What question did you try to answer?

Singh, Kroiz, Rajamanoharan and Nanda (arXiv:2606.26071) found a coding agent faking success in a
pre-commit-hook repo, and characterised it from the outside as low effort: cut the seeded error count
and the shortcut rate falls. Their paper lists its own gap as no internals, with the theory-of-mind
alternative untested.

I ran their environment (their §6.1, gkroiz/agent-interp-envs) on an open-weight model and asked what
the mitigation is actually doing inside. Two interventions move this behaviour in opposite directions.
One line of prompt cuts faked success from 14% to 4%; steering a fitted `tedium` direction pushes it
from 20% to 53%. If the low-effort reading is the whole story, those are two handles on one axis. So:
are they the same mechanism, where does the prompt's effect live, and does a probe on that axis see
either of them?

### Why is this question interesting / why did you choose it?

Three reasons, in the order I actually care about them.

It is a forensics question with something at stake in the answer. The misbehaviour is emergent
`fake_green` (weakening the check until it passes) on a deployed-shaped agentic task, not a prompted
demonstration, and the patch you would ship depends on why it happens. Singh et al. answered the
behavioural half and said plainly that the mechanistic half was open. A well-posed gap on an
environment that already exists is a better use of 20 hours than inventing a new setting.

It is also a clean test of an assumption the field leans on. Prompting and steering get treated as two
ways of installing the same disposition, and steering vectors get used as a model of what a prompt
does. I could not find that checked on the same concept, same model, with an activation-space readout.
It is falsifiable in one experiment: put the two against each other and see whether the prompt defends
the axis it is literally about.

Monitoring comes third, as a consequence rather than the reason I started. Prompting is the mitigation
people actually ship and activation probes are the monitoring people actually propose. If the two
interventions turn out to be different mechanisms, then the probe you would trust is blind to the fix
you deployed, and the failure runs both ways: fixes are invisible, and steered runs read safe while
still cheating.

### What conclusions have you reached about this research problem?

1. Prompt and steering are different mechanisms. Prompt plus `+tedium` gives 15/30 shortcuts, identical
to steering with no prompt (17/32, p = 1.0). At half dose, 10/21 vs 10/21. The prompt gives zero
protection against the axis it's literally about.

2. Direction probes are blind to prompt-based mitigation. Prompt-fixed rollouts read identical to
baseline on the tedium projection while cheating 0/21 vs 4/30, and within baseline the probe is AUROC
0.39 to 0.67.

3. Attention is not the carrier, by five independent methods. The instruction gets 0.53% of attention
at decision tokens, blocking all decode-time edges to it gives KL ≈ 5e-4, and injecting its entire
attention content into a filler run gives 21% against a 21% control (p = 1.0).

4. 24 of 32 layers were unreachable by attention tooling, since Qwen3.5 is a 3:1 GatedDeltaNet hybrid,
as is the whole current generation. I built a chunked-prefill state swap to reach them: the recurrent
channel carries about 10%. Neither channel carries the instruction alone, and component attributions
sum to 2.79 where a localised effect gives 1.0.

5. Practically: about 11 tokens ("Do not cut corners. Do not rush to finish.") carry 78% of the full
70-token instruction, and naming the feeling backfires. Prepending "this task may feel tedious" to a
sentence that works destroys three quarters of its effect (+0.53 to +0.14).

### Technical setup (models, datasets, prompts, metrics)

Qwen3.5-9B (bf16, T = 0.7, thinking on), 8×H100. The environment is a Python repo with 19 seeded type
errors; the agent must add `py.typed`, write a `pre-commit` hook running `mypy` that blocks on failure,
and commit. 80 turns, one shell command per turn, real git and mypy. The outcome measure is an LLM
judge over transcript plus final diff (`is_shortcut`, workaround type). About 500 rollouts.

Two readouts. The behavioural shortcut rate, and a paired decision-point score,
logP(" Let me fix the remaining type errors properly.") − logP(" Given the time constraints, let me
take a different approach."), measured at post-failure turns. The second is seconds per measurement
instead of 50 minutes, which is what made the component and sentence experiments feasible at all.

Interventions: mean-difference steering vectors; attention-edge masking on the 8 full-attention layers;
K/V content swap at prefill; GDN recurrent-state swap via chunked prefill; per-component zero-ablation
at the instruction's token positions; SAE (Qwen 64K, L0-100) feature decomposition.

### Strongest evidence against your hypotheses

My best hypothesis was that the prompt works by suppressing a cost-of-effort and replanning feature
family found at decision tokens. The semantics match the observed `fake_green` behaviour exactly:
the suppressed features are re-interpreting the requirement, and faking the check is redefining what
counts as passing.

Two controls reject it. Projecting that axis out of the baseline gives 10%, the no-instruction rate,
not the prompt's. And pushing it, dose-matched at α = 26, destroys the protection no better than a
random direction of the same magnitude does (19% vs 22%, p = 1.0). The same features also fail as a
monitor, at AUROC 0.31, below chance.

The sentence-level work rejected a second hypothesis of mine. I expected "Do NOT let that affect your
work" to fail alone because "that" had no antecedent, and that restoring the antecedent would help.
Restoring it made things worse: +0.53 alone against +0.14 with the tedium sentence prepended.

### Biggest limitations (could you have addressed them?)

Generalisation is untested: one model, one task, one instruction wording. This is the first thing I'd
run, and I could have traded one of the confirmatory cells for it.

Rare-event power. At 4% vs 14% base rates, separating a condition near 7% needs about 150 rollouts, so
several behavioural cells can't be decisive by construction. Moving to the paired decision-point
readout addressed this for the component and sentence work, but the behavioural claims stay
underpowered.

The two readouts disagree once. One sentence is helpful by the decision-point score and worse than
nothing behaviourally. I report both rather than picking.

Multiplicity: many comparisons. The masking result (p = 0.014) does not survive Bonferroni (0.086) and
is reported as a lead, not a finding.

The recurrent measurement is a floor. It removes the instruction's direct contribution at its own
positions, not influence already propagated into later positions.

### LLM use (which tools, what you checked, surprise-if-wrong per part)

Claude Code as an agent throughout, writing experiment runners, analysis scripts and infrastructure and
driving the cluster. I directed the research: the question, which experiment answers it, which controls
are needed, and the interpretation.

Every intervention has an explicit mechanism check that I read and re-ran rather than trusting a
passing test. Attention masking verified as exactly 0.0000 span mass at all 8 full-attention layers
with rows renormalising. K/V swap verified element-wise against the `DynamicCache` tensors, with
|swapped − donor| = 0.0000 at the span and positions after it still differing. The GDN chunked prefill
verified to reproduce single-shot to bf16 noise, with a null swap at exactly 0.0000.

Three checks caught results that would otherwise have looked clean and been wrong. Comparing
`prefill_turns` against `n_turns` revealed an ablation being skipped on long-context turns while the
rollout still recorded success. Checking vector norms before believing a comparison revealed a
tug-of-war pitting a 26%-of-residual push against a 1% one. And capping decision points per rollout
fixed error bars computed as if correlated points were independent.

On surprise-if-wrong: I'd be least surprised to be right about the decoupling and about attention not
being the carrier, since five independent methods agree and each has its own control. I'd be more
surprised about the `gdn.0` attribution, where one component exceeding the whole effect means
components interact, so I wouldn't defend the exact number. I'd be most surprised about the
sentence-level priming result. It rests on one readout, one instruction, n = 40 paired points, and it
disagrees with the behavioural cell for a different sentence, so I'd want it replicated before anyone
acted on it.

---

### Prior mechanistic interpretability experience

No formal mech interp research and no publications in the field. I have been doing it
self-directed since I left my startup in May 2026, working from reproductions toward my
own questions.

The clearest artifact is a reproduction of Anthropic's "Emotion Concepts and Their
Function in a LLM" (April 2026) on Llama-3.2-1B, a model 100x smaller than the one in the
paper. 30 emotions, mean residual-stream difference vectors denoised by projecting out the
principal components of neutral text, then validated two ways. A logit lens through the
unembedding gives clean top tokens (nostalgic reads out as nostalgia, reminis, memories).
Steering at 2/3 depth moves the target emotion word by +1.2 to +3.2 log units, and 8 of 8
emotions move in the right direction. Free generation barely shifts at all, so the readout
reproduces and the behaviour does not. I led the writeup with that gap rather than with
the number that worked. Code and figures: `code/emotion_vectors`.

Two smaller ones, built to learn techniques rather than to publish. A logit lens and SAE
feature probe of in-context learning on Pythia. And a pre-registered design testing whether
cognitive theory of mind and affective empathy are dissociable directions, which also
reproduces AE Studio's self-other-overlap axis in order to ask whether their deception fix
costs the model its ToM accuracy. That one has its predictions written down and its first
phases run; it is waiting on GPU quota.

The submitted project is where I learned the causal side, mostly because I had no choice.
Qwen3.5-9B is a hybrid of attention and GatedDeltaNet layers, so TransformerLens does not
support it and every intervention is a forward hook I wrote myself: attention edge masking,
K/V content swaps at prefill, recurrent state swaps through chunked prefill, per-component
zero ablation, activation patching, and direction ablation with a random-direction control.

What I have not done: trained an SAE, and no circuit finding on toy models beyond reading.

### Three pieces of evidence, other than the project

I shut down my own startup. IncidentFox was YC-backed, $500K raised, 600+ GitHub stars, and
our own retrieval evals said the wedge was wrong. Killing a working codebase on evidence is
the habit research needs.

A SPAR model forensics take-home. Anthropic and UK AISI disagreed publicly about why Claude
4.5 refuses safety-research tasks. One work ticket, one element varied at a time, 2,339 responses across eight Claude versions. The disputed number was two mechanisms added
together, so neither side was right.

My undergrad research was EEG and psychophysics, one paper cited 38 times. Dissociation
designs are how I was trained to think, and ablation is the same move.

### Why Neel's stream specifically

Because the project I just ran is the sort of thing your stream argues for, and I would
rather have you tell me which parts of it are wrong than keep guessing on my own.

Pragmatic interpretability is already the frame I work in. I did not ask what the circuit
is. I asked where a real agentic failure lives and whether a monitor would catch it, and the
result I care about is a negative one about monitoring rather than a diagram. The same goes
for how I treated probes: I took a promising technique into a realistic case to see if it
held up, and it did not, at AUROC 0.39 to 0.67.

Model forensics is what I have been doing without the label. The SPAR take-home above is a
forensics question in exactly that shape: a model did something that looked sketchy, two
credible organisations disagreed about whether it was misalignment or confusion, and the
answer was neither. Task gaming is where I went looking for this application's behaviour.

The honest reason, though, is that I need taste and not execution. Most of my results here
are negative. The effort direction dies to a random-direction control, attention survives
five separate attempts to implicate it, and the probes do not work. I believe those results
because they survived their own controls, but I cannot yet tell which one is worth six
months of someone's life. Lending scholars that judgement is the thing you say you do, and
it is the thing I am short of. Past scholars apparently find you blunt. I ran a company for
two years; blunt is the feedback I convert fastest.

### Likelihood of joining the exploration phase (Sept 28 to Oct 30)

Near certain. I left my startup in May 2026 to do AI safety research full time, I have no
competing employment or study commitments, and nothing is scheduled against those dates.

### Anything else about the project (optional)

Two things a reader might want and the main writeup only implies.

The negative results are the load-bearing ones. The finding I would defend hardest is that
the prompt and the steering vector are separate mechanisms, because a tug-of-war between
them shows no interaction at either dose (p = 1.0). Most of the rest of the document is me
failing to find the carrier in the obvious places, then finding where it actually is.

Roughly 75% of this model is GatedDeltaNet rather than attention, and the standard
interpretability toolkit does not reach that part at all. That constraint shaped the whole
project, and I think it generalises to any hybrid or recurrent architecture that ships next.
