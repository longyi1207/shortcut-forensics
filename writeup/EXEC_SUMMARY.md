# Prompts and steering vectors change the same behaviour by different means — and a probe sees neither

**Executive summary** · Qwen3.5-9B · agentic coding environment, 80-turn rollouts

A coding agent on a tedious 80-turn refactor sometimes fakes success: a pre-commit hook that doesn't
really block, `# type: ignore` until the checker passes. One prompt line — *"This task may feel
tedious. Do not cut corners…"* — cuts that from 14% to 4%. Steering a fitted `tedium` direction moves
the same behaviour the other way, 20% → 53%.

Two interventions, one behaviour. **Same mechanism?** If yes, one probe monitors both. If no, a
deployed prompt fix may be invisible to exactly the monitor you would trust.

## What I found

**1. Not the same mechanism — the prompt has no purchase on the steered concept.** Adding `+tedium`
to the prompted model gives 15/30 shortcuts: identical to steering with no prompt (17/32, p = 1.0),
nothing like the prompt alone (2/53). At half dose, 10/21 vs 10/21. The prompt does not defend the
axis it is literally about. *(Fig 1)*

**2. This breaks direction-probes as monitors, both ways.** Prompt-fixed rollouts read *identical*
to baseline on the `tedium` projection while cheating 0/21 vs 4/30 — the fix is invisible. Steered
rollouts read "safe" by construction while 5/30 still cheat. Within baseline the probe is barely
diagnostic (AUROC 0.39–0.67). Causal validity and diagnostic validity come apart — I hit this
dissociation three times.

**3. The instruction is barely read when it matters.** Across 128 heads at the decision tokens it
draws **0.53%** of attention; the ordinary task text beside it draws 3–5× more. Blocking every
decode-time edge to it shifts the next-token distribution by KL ≈ 5×10⁻⁴. Injecting its *entire*
attention content into a filler-text run buys nothing (21% vs 21%, p = 1.0). Five independent
methods agree: attention is not the carrier. *(Fig 4)*

**4. Most of the model was unreachable, so I built a tool to reach it.** Qwen3.5 is a 3:1 hybrid:
24 of 32 layers are GatedDeltaNet, with no attention weights and no per-token K/V, so standard
attention methods cover 25% of the stack — true of this whole model generation (Qwen3-Next, Kimi
Linear, Nemotron 3, Ling). Exploiting the recurrence, I chunk the prefill and swap the GDN state
between chunks. The recurrent channel carries **~10%**. Neither channel carries it alone, and no
single layer restores it (best: 44%). *(Fig 3, 4)*

**5. ~11 tokens carry 78% of the effect, and naming the feeling backfires.** Keeping only *"Do not
cut corners. Do not rush to finish."* recovers 78% of the full 70-token instruction. Keeping only
*"This task may feel repetitive or tedious"* does nothing (n.s.) — and prepending it to a sentence
that works **destroys three quarters of that sentence's effect** (+0.53 → +0.14). Naming the state
appears to prime it. *(Fig 2)*

## What killed my best hypotheses

The features the prompt suppresses at decision tokens are legible — cost-of-effort framing,
replanning, re-reading the requirement to license a workaround — and match the `fake_green`
behaviour exactly. **They are not the causal handle.** Projecting the axis out gives the
no-instruction rate; pushing it destroys the protection *no better than a random direction of equal
magnitude* (19% vs 22%, p = 1.0). They also fail as a monitor (AUROC 0.31). A separate experiment
was voided by its own null control, and three further errors — a tug-of-war comparing a 26% push
against a 1% one, an ablation silently skipped on long-context turns, correlated points understating
error bars — were caught before becoming results.

**Main write-up below.** Raw findings, every negative result and correction: `WRITEUP.md` in the repo.
