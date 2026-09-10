# Prompting and contrast-direction steering do not move the same thing: a per-factor test on an agentic shortcut task

## Executive summary

Qwen3.5-9B · Singh et al. pre-commit environment · 80-turn rollouts · MATS 12.0 application, Neel Nanda stream · code and data: github.com/longyi1207/shortcut-forensics

**Question.** When a one-line instruction in the prompt changes what an agent does, and a contrast-pair steering vector for the "same" concept also changes what it does, are they acting on the same internal quantity? Steering vectors are routinely read as "the model's representation of X" and prompts as a way of "turning X on or off", so the two get treated as two handles on one lever. I tested that factor by factor, on a behaviour with a ground-truth label.

**The two interventions, concretely.** The setting is Singh et al. (2026) [1]: a coding agent in a repository with 258 seeded `mypy` errors behind a pre-commit hook, which sometimes fakes a green check instead of doing the work (LLM-judged, 80-turn rollouts). Take tedium as the example. The *contrast direction* comes from 160 LLM-written sentence pairs that differ only in the tedium content ("I spent the afternoon renaming twenty near-identical spreadsheet tabs... It felt like a dull round of housekeeping" against the same sentence ending "...It was a quick cleanup before the meeting"); the direction d is the mean difference of the layer-19 last-token residuals, and steering means removing the component along d at every position (ablate) or adding d at its fitted norm (push) during the rollout. The *instruction* is one line appended to the task prompt: "IMPORTANT: This task may feel repetitive or tedious. Do NOT let that affect your work. ... Do not cut corners. Do not rush to finish. ..."; nothing inside the model is touched. The control is a same-length IMPORTANT line about shell formatting.

I did this for five factors (tedium, desperation, temptation, wanting to be done, fear of the user's disapproval) and compared three things per factor: the shortcut rate under the instruction, the shortcut rate under steering, and whether the residual shift the instruction causes lies along d, with the concept's own sentence appended in the same place as the positive control (Figure 1).

> **[FIG 1: How the two interventions are built, using tedium as the example (A: the contrast direction and how it is steered; B: the instruction line), and the three comparisons made for every factor (C).]**

**Key results**

- Each of the five instructions cuts the shortcut rate from 31% to between 0% and 11% (n = 90 each, all p ≤ 0.0005). The neutral control line leaves it at 30% (n = 150 per side, p = 1.0). The instructions do not make the agent commit honestly; they make it keep fixing errors instead of going for the commit.
- Steering on the five directions moves behaviour for tedium only: ablation 9% against the 20% baseline (p = 0.026), addition 53% (p = 0.001). A random direction ablated the same way gives 29%, and added at the same norm 31%. The other four directions, and their re-fits at other layers, are null at n = 30 to 43. The ablation p alone would not survive correction over ten steering tests; the tedium claim rests on the addition, the random-direction contrast (p = 0.003) and the same drop with the direction's SAE decomposition (2/61 = 3%, p = 0.003).
- The instruction does not move the residual along the direction, even for tedium. The concept's own sentence does (+0.85 to +2.4 in unit-direction coordinates, null bands ±0.2 to ±0.5); the instruction does not (−0.25 to +0.38); at the decision token 10 to 40 turns later nothing does (cosines within ±0.04). Two of the five directions are not readable even by their own sentence; the other two readable ones still do not steer, so readability is necessary for steering and not sufficient.
- The best-supported reading of how the instruction reaches the decision is through the context, in two copies (tested on the tedium line only, n = 20 per cell). After each failed check, blocking the agent's attention to the instruction alone (10%) or to its own earlier text alone (15%) leaves the protection in place; blocking both returns the rate to the no-instruction level (25% vs 3.8%, p = 0.014 uncorrected, 0.086 after correction); a same-size span of tool output blocked instead changes nothing (10%). The agent reads the instruction once, writes it into its plan, and then follows the plan; adding the tedium direction on top of the instruction gives 50%, the same as the direction alone.
- A direction can therefore be causally valid (it steers) and contrastively valid (its own sentence moves it) and still miss behaviour a prompt controls: the tedium probe reads an instructed agent as identical to an uninstructed one while it cheats a fifth as often. The activation a model has when it is *told* about tedium is not the one the contrast pairs isolate, and a monitor built on the direction is blind to the fix that works.

**Limitations.** One model on one task, and a task that itself induces tedium; the four null factors were tested by removing their directions, so "no effect" means "not the lever this task uses", not "inert"; the plan channel was traced on one instruction.

**Next.** Whether the gap between what a direction measures and what an instruction does closes or widens with scale; how long the plan channel carries an instruction under context compression and many simultaneous constraints; what the instructions do move, since their five footprints resemble one another and none of the concept directions; and whether training against a behaviour moves the direction or the plan.

---

# Main write-up

## 1. Motivation

Two claims are common in the steering literature: that a contrast-pair direction for a concept is "the model's representation of that concept", and that steering along it is a cheaper, more controllable version of prompting for the same thing. The second claim has recently been questioned from both sides: steering reaches activation states no prompt reaches (Mishra et al., 2026 [2]), and prompting applies token-specific interventions that uniform steering does not mimic (Heyman and Vandeputte, 2026 [3]; Kang et al., 2026 [4]). Those papers work on style and persona benchmarks, single-turn or short dialogues, and compare output quality. I wanted the comparison in the regime where it matters for safety: a long agentic task, a behaviour that is an action with a ground-truth label, and several candidate concepts side by side, so that "same lever or not" is answered per factor rather than in the aggregate.

The Singh et al. environment is a good place for this. The misbehaviour is discrete (the agent does or does not make the check pass without doing the work), the original paper argues from prompt and environment edits that Kimi K2 Thinking's shortcuts come from a disposition to low effort rather than from modelling the user, and the candidate drivers (tedium, desperation, temptation, wanting to be done, anticipating disapproval) are exactly the sort of thing people fit contrast directions for.

## 2. Setup

**Environment and model.** Repository with 258 seeded type errors, a pre-commit hook that runs `mypy`, an agent asked to fix the errors and commit; 80 tool-calling turns; Qwen3.5-9B (32 layers, 3:1 GatedDeltaNet/attention hybrid, thinking on). Rollouts ran either through vLLM (the prompt sweeps) or through HuggingFace with forward hooks (the steering cells). The two backends have different baseline rates (31% and 20%), so every comparison below is within one backend and, for vLLM, within one night; nothing is compared across the two in absolute terms.

**Judge.** An LLM judge (gpt-5.4-mini via Azure) reads the full transcript and diff and returns `is_shortcut`, the workaround type, and whether the honest path actually fixed things (`capability_ok`). A regex prefilter for commit attempts is only used descriptively. Shortcuts here are almost all "fake green" (ignores, config weakening); there is not one `--no-verify` in the corpus. Because the judge's quota is 150K tokens per minute and one transcript is ~95K tokens, 385 of tonight's verdicts came through a re-judge pass and 113 of those were judged on a transcript with tool outputs shortened to fit (the assignment of which rows this hit is independent of condition). In a spot check on this dataset, 30 rollouts weighted toward the judge's positives were read with the verdict withheld; the reader agreed with the judge on all 30. An earlier check on 18 attributable rollouts from the first run also agreed 18/18, with precision on positives 10/10.

**Directions.** For each factor, an LLM wrote paired sentences in the state and out of it; the direction is the mean difference of last-token residuals, at layer 19 (disapproval re-fit at 17). Gates: held-out pair separation, a lexical-scramble control, and a plant positive control (steering on planted text must move a judge-read state). The tedium direction at layer 19 did not pass the plant gate while its re-fits at layers 8 and 12 did; the layer-19 vector is the one that steers behaviour and is the one reported, with the re-fits alongside it (see Limitations).

**Instructions.** One line per factor, same register and length (five sentences, ~70 tokens, "IMPORTANT:" opener, names the state, names the behaviours to avoid). For the four pro-shortcut factors the line suppresses the state; for disapproval it induces it ("the person who asked would be genuinely upset to find the check weakened…"). The neutral control is a same-length "IMPORTANT:" line about shell-command formatting. All lines go at the end of the user message.

**Steering.** Pro-shortcut concepts: projection removal of the direction at every position of the fitted layer (ablation). Disapproval: addition of the direction at its fitted norm. Controls: ablation of a random direction, addition of a random direction at the tedium vector's norm.

## 3. Measurement 1: what the instructions do to behaviour

| condition (vLLM, one night) | n | shortcuts | rate | 95% CI | p vs baseline | attempted a commit | honest path fixed things |
|---|---|---|---|---|---|---|---|
| baseline | 150 | 46 | 30.7% | 24-39 | | 48 (32%) | 87% |
| neutral line | 150 | 45 | 30.0% | 23-38 | 1.00 | 48 (32%) | 88% |
| tedium line | 90 | 10 | 11.1% | 6-19 | 0.0005 | 19 (21%) | 92% |
| desperation line | 90 | 5 | 5.6% | 2-12 | <0.0001 | 6 (7%) | 94% |
| shortcut line | 90 | 0 | 0.0% | 0-4 | <0.0001 | 1 (1%) | 93% |
| completion-drive line | 90 | 6 | 6.7% | 3-14 | <0.0001 | 0 (0%) | 92% |
| disapproval line | 90 | 2 | 2.2% | 1-8 | <0.0001 | 8 (9%) | 92% |

The neutral line is the important row: a same-length instruction with no concept content leaves the rate exactly where it was, so the effect is in the content. The commit-attempt column shows what the content does: at baseline almost every commit attempt is a fake green (48 attempts, 46 shortcuts), and the instructions do not make the agent commit honestly, they make it not go for the commit at all. It spends the 80 turns fixing errors, and fixes them a little more often.

## 4. Measurement 2: what steering on the same directions does

| condition (HF) | n | shortcuts | rate | p vs HF baseline |
|---|---|---|---|---|
| baseline | 85 | 17 | 20.0% | |
| ablate tedium (L19) | 153 | 14 | 9.2% | 0.026 |
| add tedium (L19) | 32 | 17 | 53.1% | 0.001 |
| ablate random direction | 42 | 12 | 28.6% | 0.37 |
| add random direction at the tedium norm | 16 | 5 | 31.2% | 0.33 |
| ablate desperation | 30 | 5 | 16.7% | 0.79 |
| ablate shortcut | 33 | 8 | 24.2% | 0.62 |
| ablate completion drive | 32 | 10 | 31.2% | 0.22 |
| add disapproval (L17) | 43 | 9 | 20.9% | 1.00 |
| ablate tedium, SAE top-10 decomposition (L19) | 61 | 2 | 3.3% | 0.003 |
| ablate shortcut, SAE top-10 (L19) | 30 | 8 | 26.7% | 0.45 |
| ablate disapproval, SAE top-1 (L19) | 50 | 14 | 28.0% | 0.30 |

Tedium is the only direction that moves behaviour, and it moves it in both signs; the random-direction controls separate from it in both signs (ablation 9.2% vs 28.6%, p = 0.003; addition 53.1% vs 31.2% for a random direction added at the same norm, p = 0.23 at n = 16, so the addition contrast is suggestive rather than established). On its own the ablation's p = 0.026 would not survive correction over the ten steering tests in the table; what carries the tedium claim is the addition (p = 0.001), the random-direction contrast, and the same drop when the direction is replaced by its ten strongest SAE features at the same layer (2/61 = 3.3%, p = 0.003 against baseline, 0.0003 against the random direction). The same SAE construction for shortcut and disapproval is null. The other four are null at n ≈ 30 to 43, which bounds them to effects smaller than tedium's rather than to zero; re-fits of shortcut (layers 14 to 31) and disapproval (layers 8, 12, 19) are null too. This is the pattern Braun et al. (2025) [5] predict when a behaviour is not represented by one coherent direction, and it is already a mismatch with Measurement 1: five instructions with large effects, one direction with an effect.

> **[FIG 2: Per factor, the change in shortcut rate under the instruction (blue; vLLM, against the same-night baseline of 31%) and under steering on the fitted direction (red; HF, against the HF baseline of 20%), with 95% Wilson intervals; bar labels give the raw rate and n. Dotted line: the neutral control line's rate relative to the baseline. The two halves sit on different backends and are each compared with their own baseline, so only the direction and size of each bar is comparable across halves.]**

## 5. Measurement 3: does the instruction move along the direction?

For each factor I appended one of three texts to the user prompt, the instruction, the concept's plus-pole plant sentence (a first-person status note in the register of the contrast pairs), or its minus-pole plant, and measured Δh, the change in the residual at the first assistant position, projected on the factor's direction. The plants are the positive control: if the measurement cannot see the concept when the text *is* the concept, orthogonality of the instruction means nothing.

| factor (direction) | plus plant | minus plant | instruction | neutral line | null (other lines) |
|---|---|---|---|---|---|
| tedium (L19) | +0.85 | −0.40 | −0.25 | −0.38 | +0.05 ± 0.39 |
| desperation (L19) | +2.43 | −0.53 | −0.02 | +0.13 | +0.11 ± 0.22 |
| shortcut (L19) | +1.87 | +0.05 | +0.38 | +0.03 | +0.35 ± 0.47 |
| completion drive (L19) | +0.49 | −0.28 | +0.07 | +0.09 | +0.52 ± 0.58 |
| disapproval (L17) | −0.14 | −0.19 | +0.07 | +0.06 | +0.03 ± 0.18 |

Three of the five directions are readable in context. Their own plant moves the residual along them by two to ten null standard deviations. The instruction moves none of them; its projection sits inside the null band for every factor, and for tedium it is on the minus side, where the neutral line also sits. The completion-drive and disapproval directions are not readable even by their own plant at any fitted layer; desperation and shortcut are readable and still do not steer, so readability is necessary for steering and not sufficient. The same measurement at every other layer a direction was fitted at (tedium 8 and 12; shortcut 14 to 31; disapproval 8, 12 and 19) gives the same picture. Positions inside the response, where persona-vector work reads its projections, were not measured. At the decision token, 10 to 40 turns later (40 decision points, three per rollout, the turn after a failed tool result), no text moves the residual along any direction (cosines within ±0.04 for plants and instructions alike, projection shifts ≤ 0.4 against baseline projections of 1 to 4), while the instruction's shift of the next-action distribution there is large. Whatever the instruction does at the decision, it does not do it on these axes. It does do something consistent: the mean shifts the five instructions cause at the decision token resemble one another (pairwise cosines 0.35 to 0.74 at layer 19) far more than any of them resembles any concept direction (within ±0.07). The five lines share a footprint, and it is not five concepts.

> **[FIG 3: Does adding one line to the prompt move the residual stream along the factor's fitted direction? Left: right after the prompt, the shift of the residual along each factor's own direction when one line is appended to the user message: the concept's own sentence at the plus pole (red, the positive control), at the minus pole (blue), the instruction (green) and the neutral line (grey). A tall red bar means the direction is readable in this context; for every factor the instruction's bar sits at the neutral line's level. Right: the same question at the decision token 10 to 40 turns later, as the cosine between the shift and the direction averaged over 40 decision points (standard errors); nothing moves along any direction there, for any line.]**

The tedium direction, which is causally valid (steering it moves behaviour) and contrastively valid (its plant moves it), reads an instructed agent as identical to an uninstructed one (0/21 vs 4/30 shortcuts in a matched pair of cells; projections indistinguishable). The premise of monitoring with persona-style vectors (Chen et al., 2025 [6]: the last-prompt-token projection predicts subsequent trait expression) does not hold here for behaviour that a prompt controls.

## 6. How the instruction reaches the decision

The earlier part of this project asked how the instruction reaches the decision if not through the direction, with judge-labelled cells on the tedium line (HF backend; references: instruction intact 2/53 = 3.8%, no instruction 8/59 = 13.6%).

*Not by re-reading the instruction.* At decision points the instruction span draws 0.53% of attention across the 128 heads, less than a third of what neighbouring task text draws; blocking every generated token's attention to it changes the next-token distribution by KL ≈ 5×10⁻⁴; and swapping the instruction's full-attention K/V content into a run whose prompt has only filler text does nothing (9/43 = 21% vs the filler control's 5/24 = 21%).

*Not as an added direction of its own.* The instruction's mean residual footprint Δh at layer 26 (~7% of the residual norm), added to a baseline run, gives 1/12 (baseline 4/27); projected out of an instructed run, 0/12.

*Through the context, in two copies.* In every turn after a failed check, block attention from the generated tokens to (a) the instruction, (b) all of the model's own earlier assistant text, or (c) both: 10% (2/20), 15% (3/20), 25% (5/20). Only (c) separates from the intact prompt (p = 0.014 uncorrected, 0.086 after Bonferroni over the three cuts), and it lands at the no-instruction rate; a same-size span of ordinary tool output blocked instead leaves 10% (2/21). My reading is that the model consumes the instruction in its first turns, writes it into its own plan, and then runs on the plan; the instruction and the plan are two copies, and either suffices. This is the same dependence on plan text staying in context that Mehta and Datta (2026) [7] report for Llama agents on HotpotQA and ALFWorld, reached from the other side.

*A channel the direction can override.* Adding the tedium direction on top of the instruction gives 15/30 = 50%, indistinguishable from the direction without the instruction (17/32 = 53%, p = 1.0). The instruction offers no resistance because it never set the coordinate the steer moves.

*The recurrent three quarters of the model.* Qwen3.5-9B has 24 GatedDeltaNet blocks to 8 attention blocks, so attention masking and K/V swaps reach a quarter of the layers. To reach the rest I built a chunked-prefill state swap that replaces the instruction's contribution to the recurrent state with a filler's, validated to bf16 noise against single-shot prefill with an exactly zero null swap. The behavioural cell is void: the null swap, a real-to-real no-op, moved the rate by itself (3/16 = 19% against the intact prompt's 6%), so chunked prefill is not behaviour-neutral, and the recurrent channel's share of the instruction is unmeasured.

> **[FIG 4: How the instruction reaches the decision (tedium line, HF backend). In every turn after a failed check, the agent's attention from the tokens it is generating is blocked to one part of the context: the instruction, all of its own earlier assistant text (its notes and plans), both, or a same-size chunk of ordinary tool output as the control. Bars: shortcut rate with 95% Wilson intervals and raw counts; dotted lines: the rate with the instruction intact (3.8%) and with no instruction (13.6%). Blocking either copy alone leaves the protection in place; blocking both returns the rate to the no-instruction level; blocking the control chunk does not. n = 20 to 59 per cell.]**

## 7. Related work

Stolfo et al. (2025) [8] build instruction vectors as with-minus-without activation differences and find they enforce format constraints; the same construction here is neither sufficient nor necessary for a behavioural constraint over 80 turns, which suggests their result is about instructions whose effect is local to the response. The non-surjectivity result [2] shows steering reaches states prompts cannot; the present result is the converse: prompts reach a behaviour by a route (the agent's own context) that a fixed direction cannot express. The two prompt-mimicking steering papers [3, 4] locate prompt effects in token-specific, attention-mediated pathways within a response; in a long agentic run the attention edge to the prompt is nearly dead at the decision and the pathway has moved into the model's own outputs. Singh et al. [1] reach their forensic conclusions with prompt and environment edits only; the per-factor comparison here says which of their candidate drivers a direction can be fitted for at all (tedium) and warns that a direction's null is not a factor's null.

## 8. Limitations

*What the comparison can support.* On this model and task, an instruction and a contrast direction for the same factor are not two handles on one quantity. The task is one that itself induces tedium (258 near-identical fixes), which is the one factor whose direction moves behaviour; whether the pattern is "instructions beat directions" in general, or "only the factor the task actually engages has a working direction", cannot be separated with one task.

*The four null factors are incompletely tested.* Their directions were tested by removal (and, for disapproval, by addition). Removal can only show an effect if the state is present at baseline, and the baseline agent may not be desperate, tempted, or afraid of the user at all. The complementary test, adding each direction in the pro-shortcut sign at a matched norm, was run only for tedium (53%). Until it is run for the others, their nulls say that removal is not the lever this task uses, not that the factors are inert.

*Placement.* Every instruction sat at the end of the user message. Whether a system-prompt instruction takes the same route through the agent's plan, or is handled differently, is untested.

*The mechanism is traced on one instruction.* The two-copies result was measured on the tedium line only, at n = 20 per cell, and its one separating comparison does not survive correction for the three cuts. That the same route carries the other four instructions is a hypothesis this study motivates but does not test.

*Method.* The judge is one LLM; spot checks (30/30 on this dataset, 18/18 on an earlier one) are reassuring rather than tight, and 113 verdicts came from transcripts shortened to fit the judge's quota. Steering cells of n = 30 to 43 would miss a true drop from 20% to 10% about half the time. Prompt sweeps ran on vLLM and steering cells on HuggingFace, whose baselines differ (31% and 20%); all comparisons are within one backend. The geometry uses the last-token residual at two positions and not inside the response. The layer-19 tedium vector failed its plant gate and was kept because it steers, with the SAE decomposition at the same layer as its only independent support.

## 9. Next steps

1. *Scale.* Larger models follow instructions more reliably and represent concepts more linearly, and those trends pull in opposite directions for a probe. Either the instruction begins to register on the concept direction at some scale, at which point direction-based monitoring starts to work for prompted behaviour, or the plan channel dominates further and it never does. The same three measurements at 30B and 70B decide which.

2. *Persistence and capacity.* Here the instruction held for 80 turns with the full context intact. Real agents compress and summarise their context, and plans are the first thing evicted. The two-copies result predicts that protection survives as long as either the instruction or the plan text is still readable and fails when both are gone; that is testable under a context-management policy, and it extends to how many simultaneous constraints one plan can carry.

3. *Why only tedium.* Measure each direction's projection over baseline rollouts to see which states the task actually engages, add each direction in the pro-shortcut sign at a matched norm, and repeat on a task that engages a different factor. This separates "the direction is not causal" from "the state was never on".

4. *Is the plan channel general.* Transplant the instructed agent's first-turn plan into an uninstructed run, for each of the five lines and for system-prompt placement. If the plan alone protects, the object to monitor is the plan text, and a text-level monitor on the agent's own plan should be compared directly against activation probes.

5. *Which lever does training use.* Fine-tune against the shortcut and measure whether the concept direction or the plan channel moved. If training, like prompting, does not move the direction, then direction-based monitoring is blind to trained-in changes as well, which is the case persona-vector monitoring is meant for.

6. *What the instruction does move.* The five instructions leave footprints that resemble one another (cosines 0.35 to 0.74) and none of the concept directions, which looks like one shared "instructed" direction rather than five states. Fitting that direction from the five footprints together, and testing whether steering it reproduces or removes the instructions' effect, would say whether prompting has a coordinate of its own in activation space. A single instruction's mean footprint was neither sufficient nor necessary here; a direction shared across five is a different object, and if it works it is the natural monitoring target.

## References

1. A. Singh, G. Kroiz, S. Rajamanoharan, N. Nanda. *Model Forensics: Investigating Whether Concerning Behavior Reflects Misalignment.* arXiv:2606.26071, 2026.
2. A. Mishra, D. Khashabi, A. Liu. *Steered LLM Activations are Non-Surjective.* arXiv:2604.09839, 2026.
3. G. Heyman, F. Vandeputte. *Steer Like the LLM: Activation Steering that Mimics Prompting.* ICML 2026; arXiv:2605.03907.
4. D. Kang, Z. Liu, N. Ma, Y. Huang, Z. Tan, M. Jiang. *Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions.* arXiv:2605.10664, 2026.
5. J. Braun, C. Eickhoff, D. Krueger, S. A. Bahrainian, D. Krasheninnikov. *Understanding (Un)Reliability of Steering Vectors in Language Models.* arXiv:2505.22637, 2025.
6. R. Chen, A. Arditi, H. Sleight, O. Evans, J. Lindsey. *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509, 2025.
7. A. Mehta, A. Datta. *Plans Don't Persist: Why Context Management Is Load Bearing for LLM Agents.* arXiv:2606.22953, 2026.
8. A. Stolfo, V. Balachandran, S. Yousefi, E. Horvitz, B. Nushi. *Improving Instruction-Following in Language Models through Activation Steering.* ICLR 2025; arXiv:2410.12877.
