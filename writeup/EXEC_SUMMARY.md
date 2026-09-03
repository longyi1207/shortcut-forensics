# Prompts and steering vectors change the same behaviour by different means, and a probe sees neither

**Executive summary** · Qwen3.5-9B · agentic coding environment, 80-turn rollouts

A coding agent on a tedious 80-turn refactor sometimes fakes success: a pre-commit hook that doesn't
really block, `# type: ignore` until the checker passes. One prompt line ("This task may feel tedious.
Do not cut corners...") cuts that from 14% to 4%. Steering a fitted `tedium` direction moves the same
behaviour the other way, 20% to 53%.

Two interventions, one behaviour. Are they the same mechanism? If they are, one probe monitors both.
If they aren't, a deployed prompt fix may be invisible to exactly the monitor you would trust.

> **[FIG 1: fig1_decoupling.png]**

## Findings

**1.** They are not the same mechanism, and the prompt has no purchase on the steered concept. Adding
`+tedium` to the prompted model gives 15/30 shortcuts: identical to steering with no prompt (17/32,
p = 1.0), nothing like the prompt alone (2/53). At half dose, 10/21 vs 10/21. The prompt does not
defend the axis it is literally about. *(Fig 1)*

**2.** Direction probes are blind to prompt-based mitigation, in both directions. Prompt-fixed
rollouts read identical to baseline on the `tedium` projection while cheating 0/21 vs 4/30. Steered
rollouts read "safe" by construction while 5/30 still cheat. Within baseline the probe is barely
diagnostic (AUROC 0.39 to 0.67). Causal validity and diagnostic validity come apart, three times in
this project.

**3.** The instruction is barely read when it matters. Across 128 heads at the decision tokens it
draws **0.53%** of attention, and the ordinary task text beside it draws 3 to 5 times more. Blocking
every decode-time edge to it shifts the next-token distribution by KL ≈ 5×10⁻⁴. Injecting its entire
attention content into a filler-text run buys nothing (21% vs 21%, p = 1.0). Five independent methods
agree that attention is not the carrier. *(Fig 4)*

**4.** Three quarters of the model needs different tools. Qwen3.5 is a 3:1 hybrid: 24 of 32 layers are
GatedDeltaNet, with no attention weights and no per-token K/V, so attention-specific methods cover 25%
of the stack. That is true of this whole model generation (Qwen3-Next, Kimi Linear, Nemotron 3, Ling).
Exploiting the recurrence, I chunk the prefill and swap the GDN state between chunks to measure that
channel directly. It carries about **10%** of the effect. Neither channel carries the instruction
alone, and no single layer restores it (best: 44%). *(Fig 3, 4)*

**5.** About 11 tokens carry 78% of the effect, and naming the feeling backfires. Keeping only "Do not
cut corners. Do not rush to finish." recovers **78%** of the full 70-token instruction. Keeping only
"This task may feel repetitive or tedious" does nothing (n.s.), and prepending it to a sentence that
works destroys three quarters of that sentence's effect (+0.53 to +0.14). Naming the state appears to
prime it. *(Fig 2)*

> **[FIG 2: fig2_sentences.png]**

## The hypothesis the evidence rejects

The features the prompt suppresses at decision tokens are legible and match the behaviour well: cost-
of-effort framing, replanning, and re-reading the requirement to license a workaround, which is
precisely what faking the check consists of. That axis is not the causal handle. Projecting it out of
the baseline gives the no-instruction rate (10%, p = 1.0), not the prompt's. Pushing it, dose-matched,
takes the prompt to 19%, but a random direction of the same magnitude gives 22%, indistinguishable
(p = 1.0). The same features fail as a monitor (AUROC 0.31, below chance). A feature set can describe
a behaviour exactly and have no causal role in producing it.

Main write-up below. Raw findings and full experiment log: `WRITEUP.md` in the repo.
