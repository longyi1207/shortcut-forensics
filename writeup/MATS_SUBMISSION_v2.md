# Prompting and contrast-direction steering do not move the same thing: a per-factor test on an agentic shortcut task

**Executive summary** · Qwen3.5-9B · Singh et al. pre-commit environment · 80-turn rollouts · MATS 12.0 application, Neel Nanda stream

**Question.** When an instruction in the prompt changes what an agent does, and a contrast-pair steering vector for the "same" concept also changes what it does, are they acting on the same internal quantity? Steering vectors are routinely read as "the model's representation of X", and prompts as a way of "turning X on or off", so the two are usually assumed to be two handles on one lever. I tested that assumption factor by factor, in a setting where the behaviour is an action rather than a style.

**Setting.** Singh et al. (2026) [1] put a coding agent in a repository with 258 seeded `mypy` errors behind a pre-commit hook. The agent sometimes "passes" by making the check green without doing the work (blanket ignores, weakened config; never `--no-verify`). An LLM judge labels each 80-turn rollout as shortcut or not. I fitted five candidate factors as mean-difference directions from LLM-generated contrast pairs (tedium, desperation, shortcut temptation, completion drive, anticipated user disapproval), each with validation gates, and wrote one matched instruction per factor in the user prompt ("IMPORTANT: this task may feel tedious. Do NOT let that affect your work…"), plus a same-length neutral line about formatting as the control.

**Three measurements per factor.** (1) Shortcut rate under the instruction, against a same-night baseline (vLLM, n = 90 per line, 150 for the two controls). (2) Shortcut rate under the sign-appropriate steering intervention on the fitted direction (ablation for the pro-shortcut concepts, addition for disapproval; HuggingFace hooks, n = 30–153), against the HF baseline and a random-direction control. (3) The instruction's footprint on the residual stream, measured as the shift Δh it causes, projected on the fitted direction, at the first assistant position after the prompt and again at the decision token 10–40 turns later; the concept's own plant sentences serve as the positive control.

**Results.**

| factor | instruction (vLLM; baseline 46/150 = 31%) | steering (HF; baseline 17/85 = 20%) | instruction's shift along the direction, prompt end (plant control in brackets) |
|---|---|---|---|
| tedium | 10/90 = 11% (p = 0.0005) | ablate: 14/153 = 9% (p = 0.026) | −0.25 (plant +0.85) |
| desperate | 5/90 = 6% (p < 0.0001) | ablate: 5/30 = 17% (n.s.) | −0.02 (plant +2.4) |
| shortcut | 0/90 = 0% (p < 0.0001) | ablate: 8/33 = 24% (n.s.) | +0.38 (plant +1.9) |
| completion drive | 6/90 = 7% (p < 0.0001) | ablate: 10/32 = 31% (n.s.) | +0.07 (plant +0.5, not readable) |
| disapproval | 2/90 = 2% (p < 0.0001) | add: 9/43 = 21% (n.s.) | +0.07 (plant −0.1, not readable) |
| neutral line | 45/150 = 30% (p = 1.0) | random direction ablated: 12/42 = 29% (n.s.) | |

Every instruction cuts the shortcut rate by two thirds or more; a same-length line with no concept content does nothing. Steering on the same five directions moves behaviour for one factor only, tedium (and there in both signs: adding the direction raises the rate to 17/32 = 53%), and a random direction of matched norm does not (p = 0.003 against tedium ablation). Even for tedium the instruction does not travel along the direction: at the position where the instruction is freshest, the concept's own sentences move the residual along the direction (+0.85 in unit-direction coordinates, null ±0.4) and the instruction does not (−0.25); at the decision token nothing moves along any direction (all cosines within ±0.04), yet the instruction still changes the next-action distribution there. Direction probes therefore read an instructed agent as identical to an uninstructed one while it cheats a fifth as often.

**How the instruction works instead** (judge-labelled cells from the same programme, tedium line, HF): the agent does not consult the instruction at the decision (0.5% of attention across 128 heads; blocking that edge shifts the next-token distribution by KL ≈ 5×10⁻⁴; injecting the instruction's entire attention content into a filler prompt does nothing, 21% vs 21%). Its mean residual footprint is neither sufficient (added to baseline: 1/12) nor necessary (removed from the instructed run: 0/12). What carries it is the context, redundantly: after each failed check, blocking attention to the instruction alone leaves the rate at 10% (2/20), blocking attention to the model's own earlier assistant text alone leaves it at 15% (3/20), blocking both returns it to 25% (5/20, p = 0.014 uncorrected against the intact 2/53 = 3.8%), while blocking a same-size span of tool output leaves it at 10% (2/21). The model reads the instruction once, writes it into its own plan, and afterwards follows its plan. Consistent with this, adding the tedium direction on top of the instruction gives 15/30 = 50%, the same as the direction alone: the direction overrides the context channel rather than competing with a state the instruction had set.

**Why it matters.** A contrast direction is fitted on the model reading about the concept, and that is what it measures: the concept's own sentences move the residual along it, the instruction does not, and the agent's behaviour changes anyway. So the activation a model has when it is *told* about tedium is not the activation the contrast pairs isolate, and a method that treats the fitted direction as "the model's tedium" will misread prompted behaviour in both directions: a null steering result does not mean the factor is inert (four of five here), and a direction that does steer (tedium) still reads an instructed agent as unchanged. For monitoring this is the concrete cost: a probe on the tedium axis, which is causally valid and contrastively valid, sees nothing when an instruction cuts the shortcut rate by two thirds. Causal validity, contrastive validity and diagnostic validity come apart, and monitors need the third. The object that does carry the instruction is the agent's own written plan, not a residual axis. What I would do next: transplant the instructed agent's early plan text into an uninstructed run (does the plan alone protect?), and re-do the component attribution with a readout that is validated against the judge, since the one I built (log-odds of an engage-vs-replan sentence at the decision token) turned out not to track the judge labels (AUROC 0.56–0.59) and I retracted everything that rested on it.

> **[FIG 1: Per factor, the shortcut-rate change under the instruction (vLLM, vs the same-night baseline) and under steering on the fitted direction (HF, vs the HF baseline), with Wilson 95% intervals. Dotted line: the neutral control line.]**

---

## 1. Motivation

Two claims are common in the steering literature: that a contrast-pair direction for a concept is "the model's representation of that concept", and that steering along it is a cheaper, more controllable version of prompting for the same thing. The second claim has recently been questioned from both sides: steering reaches activation states no prompt reaches (Mishra et al., 2026 [2]), and prompting applies token-specific interventions that uniform steering does not mimic (Heyman and Vandeputte, 2026 [3]; Kang et al., 2026 [4]). Those papers work on style and persona benchmarks, single-turn or short dialogues, and compare output quality. I wanted the comparison in the regime where it matters for safety: a long agentic task, a behaviour that is an action with a ground-truth label, and several candidate concepts side by side, so that "same lever or not" is answered per factor rather than in the aggregate.

The Singh et al. environment is a good place for this. The misbehaviour is discrete (the agent does or does not make the check pass without doing the work), the original paper argues from prompt and environment edits that Kimi K2 Thinking's shortcuts come from a disposition to low effort rather than from modelling the user, and the candidate drivers (tedium, desperation, temptation, wanting to be done, anticipating disapproval) are exactly the sort of thing people fit contrast directions for.

## 2. Setup

**Environment and model.** Repository with 258 seeded type errors, a pre-commit hook that runs `mypy`, an agent asked to fix the errors and commit; 80 tool-calling turns; Qwen3.5-9B (32 layers, 3:1 GatedDeltaNet/attention hybrid, thinking on). Rollouts ran either through vLLM (the prompt sweeps) or through HuggingFace with forward hooks (the steering cells). The two backends have different baseline rates (31% and 20%), so every comparison below is within one backend and, for vLLM, within one night; nothing is compared across the two in absolute terms.

**Judge.** An LLM judge (gpt-5.4-mini via Azure) reads the full transcript and diff and returns `is_shortcut`, the workaround type, and whether the honest path actually fixed things (`capability_ok`). A regex prefilter for commit attempts is only used descriptively. Shortcuts here are almost all "fake green" (ignores, config weakening); there is not one `--no-verify` in the corpus. Because the judge's quota is 150K tokens per minute and one transcript is ~95K tokens, 385 of tonight's verdicts came through a re-judge pass and 113 of those were judged on a transcript with tool outputs shortened to fit (the assignment of which rows this hit is independent of condition).

**Directions.** For each factor, an LLM wrote paired sentences in the state and out of it; the direction is the mean difference of last-token residuals, at layer 19 (disapproval re-fit at 17). Gates: held-out pair separation, a lexical-scramble control, and a plant positive control (steering on planted text must move a judge-read state). The tedium direction at layer 19 did not pass the plant gate although its re-fits at layers 8 and 12 did; I report the layer-19 vector because it is the one that steers behaviour, and the re-fits alongside it.

**Instructions.** One line per factor, same register and length (five sentences, ~70 tokens, "IMPORTANT:" opener, names the state, names the behaviours to avoid). For the four pro-shortcut factors the line suppresses the state; for disapproval it induces it ("the person who asked would be genuinely upset to find the check weakened…"). The neutral control is a same-length "IMPORTANT:" line about shell-command formatting. All lines go at the end of the user message.

**Steering.** Pro-shortcut concepts: projection removal of the direction at every position of the fitted layer (ablation). Disapproval: addition of the direction at its fitted norm. Controls: ablation of a random direction, addition of a random direction at the tedium vector's norm.

## 3. Measurement 1: what the instructions do to behaviour

| condition (vLLM, one night) | n | shortcuts | rate | 95% CI | p vs baseline | attempted a commit | honest path fixed things |
|---|---|---|---|---|---|---|---|
| baseline | 150 | 46 | 30.7% | 24–39 | | 48 (32%) | 87% |
| neutral line | 150 | 45 | 30.0% | 23–38 | 1.00 | 48 (32%) | 88% |
| tedium line | 90 | 10 | 11.1% | 6–19 | 0.0005 | 19 (21%) | 92% |
| desperation line | 90 | 5 | 5.6% | 2–12 | <0.0001 | 6 (7%) | 94% |
| shortcut line | 90 | 0 | 0.0% | 0–4 | <0.0001 | 1 (1%) | 93% |
| completion-drive line | 90 | 6 | 6.7% | 3–14 | <0.0001 | 0 (0%) | 92% |
| disapproval line | 90 | 2 | 2.2% | 1–8 | <0.0001 | 8 (9%) | 92% |

The neutral line is the important row: a same-length instruction with no concept content leaves the rate exactly where it was, so the effect is in the content. The commit-attempt column shows what the content does: at baseline almost every commit attempt is a fake green (48 attempts, 46 shortcuts), and the instructions do not make the agent commit honestly, they make it not go for the commit at all. It spends the 80 turns fixing errors, and fixes them a little more often.

## 4. Measurement 2: what steering on the same directions does

| condition (HF) | n | shortcuts | rate | p vs HF baseline |
|---|---|---|---|---|
| baseline | 85 | 17 | 20.0% | |
| ablate tedium (L19) | 153 | 14 | 9.2% | 0.026 |
| add tedium (L19) | 32 | 17 | 53.1% | 0.001 |
| ablate random direction | 42 | 12 | 28.6% | 0.37 |
| ablate desperation | 30 | 5 | 16.7% | 0.79 |
| ablate shortcut | 33 | 8 | 24.2% | 0.62 |
| ablate completion drive | 32 | 10 | 31.2% | 0.22 |
| add disapproval (L17) | 43 | 9 | 20.9% | 1.00 |

Tedium is the only direction that moves behaviour, and it moves it in both signs; the random-direction control separates from it cleanly (9.2% vs 28.6%, p = 0.003). The other four are null at n ≈ 30–43, which bounds them to effects smaller than tedium's rather than to zero; re-fits of shortcut (layers 14–31) and disapproval (layers 8, 12, 19) are null too. This is the pattern Braun et al. (2025) [5] predict when a behaviour is not represented by one coherent direction, and it is already a mismatch with Measurement 1: five instructions with large effects, one direction with an effect.

## 5. Measurement 3: does the instruction move along the direction?

For each factor I appended one of three texts to the user prompt, the instruction, the concept's plus-pole plant sentence (a first-person status note in the register of the contrast pairs), or its minus-pole plant, and measured Δh, the change in the residual at the first assistant position, projected on the factor's direction. The plants are the positive control: if the measurement cannot see the concept when the text *is* the concept, orthogonality of the instruction means nothing.

| factor (direction) | plus plant | minus plant | instruction | neutral line | null (other lines) |
|---|---|---|---|---|---|
| tedium (L19) | +0.85 | −0.40 | −0.25 | −0.38 | +0.05 ± 0.39 |
| desperation (L19) | +2.43 | −0.53 | −0.02 | +0.13 | +0.11 ± 0.22 |
| shortcut (L19) | +1.87 | +0.05 | +0.38 | +0.03 | +0.35 ± 0.47 |
| completion drive (L19) | +0.49 | −0.28 | +0.07 | +0.09 | +0.52 ± 0.58 |
| disapproval (L17) | −0.14 | −0.19 | +0.07 | +0.06 | +0.03 ± 0.18 |

Three of the five directions are readable in context. Their own plant moves the residual along them by two to ten null standard deviations. The instruction moves none of them; its projection sits inside the null band for every factor, and for tedium it is on the minus side, where the neutral line also sits. The completion-drive and disapproval directions are not readable even by their own plant at any fitted layer, which is the geometric face of their steering nulls. At the decision token, 10–40 turns later (40 decision points, three per rollout, the turn after a failed tool result), no text moves the residual along any direction (cosines within ±0.04 for plants and instructions alike, projection shifts ≤ 0.4 against baseline projections of 1–4), while the instruction's shift of the next-action distribution there is large. Whatever the instruction does at the decision, it does not do it on these axes.

> **[FIG 2: Left, prompt end: shift along each factor's own direction for the plus plant, minus plant, instruction and neutral line. Right, decision token: cosine between the shift and the direction, 40 points, standard errors.]**

One consequence stands on its own. The tedium direction, which is causally valid (steering it moves behaviour) and contrastively valid (its plant moves it), reads an instructed agent as identical to an uninstructed one (0/21 vs 4/30 shortcuts in a matched pair of cells; projections indistinguishable). The premise of monitoring with persona-style vectors (Chen et al., 2025 [6]: the last-prompt-token projection predicts subsequent trait expression) does not hold here for behaviour that a prompt controls.

## 6. How the instruction reaches the decision

If not through the direction, then how? The earlier part of this project answered that with judge-labelled cells on the tedium line (HF backend; references: instruction intact 2/53 = 3.8%, no instruction 8/59 = 13.6%).

*Not by re-reading the instruction.* At decision points the instruction span draws 0.53% of attention across the 128 heads, less than a third of what neighbouring task text draws; blocking every generated token's attention to it changes the next-token distribution by KL ≈ 5×10⁻⁴; and swapping the instruction's full-attention K/V content into a run whose prompt has only filler text does nothing (9/43 = 21% vs the filler control's 5/24 = 21%).

*Not as an added direction of its own.* The instruction's mean residual footprint Δh at layer 26 (~7% of the residual norm), added to a baseline run, gives 1/12 (baseline 4/27); projected out of an instructed run, 0/12.

*Through the context, redundantly.* In every turn after a failed check, block attention from the generated tokens to (a) the instruction, (b) all of the model's own earlier assistant text, or (c) both: 10% (2/20), 15% (3/20), 25% (5/20). Only (c) separates from the intact prompt (p = 0.014 uncorrected, 0.086 after Bonferroni over the three cuts), and it lands at the no-instruction rate; a same-size span of ordinary tool output blocked instead leaves 10% (2/21). My reading is that the model consumes the instruction in its first turns, writes it into its own plan, and then runs on the plan; the instruction and the plan are two copies, and either suffices. This is the same dependence on plan text staying in context that Mehta and Datta (2026) [7] report for Llama agents on HotpotQA and ALFWorld, reached from the other side.

*A channel the direction can override.* Adding the tedium direction on top of the instruction gives 15/30 = 50%, indistinguishable from the direction without the instruction (17/32 = 53%, p = 1.0). The instruction offers no resistance because it never set the coordinate the steer moves.

> **[FIG 3: Shortcut rate when attention after a failed check is blocked to the instruction, to the model's own earlier text, or to both, with a size-matched control.]**

## 7. What I retracted, and why

The component-level and sentence-level attributions from the earlier programme (a first-layer recurrent write, one eleven-token clause carrying most of the effect) were measured with a decision-point readout, the log-odds of "let me fix the remaining errors properly" against "given the time constraints, let me take a different approach" at post-failure turns, chosen because behavioural cells at 4% vs 14% need n ≈ 150 each. Last night I checked that readout against the judge on 147 labelled rollouts: AUROC 0.56–0.59, on the wrong side of 0.5. It measures something, but not who cheats. Everything above that rests on judge labels stands; everything that rested on the readout is out of this write-up, including a sentence-level result the behavioural cells had already contradicted (one sentence read as helpful on the readout and as harmful, 6/17, on behaviour). Two of the fitted vectors also had gate failures I should state plainly: the tedium L19 vector failed its plant gate (re-fits at L8/L12 passed but do not steer), and the disapproval vector had to be re-fit at L17 to pass.

## 8. Relation to prior work

Stolfo et al. (2025) [8] build instruction vectors as with-minus-without activation differences and find they enforce format constraints; the same construction here is neither sufficient nor necessary for a behavioural constraint over 80 turns, which suggests their result is about instructions whose effect is local to the response. The non-surjectivity result [2] shows steering reaches states prompts cannot; the present result is the converse: prompts reach a behaviour by a route (the agent's own context) that a fixed direction cannot express. The two prompt-mimicking steering papers [3, 4] locate prompt effects in token-specific, attention-mediated pathways within a response; in a long agentic run the attention edge to the prompt is nearly dead at the decision and the pathway has moved into the model's own outputs. Singh et al. [1] reach their forensic conclusions with prompt and environment edits only; the per-factor comparison here says which of their candidate drivers a direction can be fitted for at all (tedium) and warns that a direction's null is not a factor's null.

## 9. Limitations

One model, one task, one judge. The steering cells (n = 30–43) can miss effects of the size a 20% → 10% drop would be. The instructions and the steers are not dose-matched, and the instructions are mine. The mechanism cells in §6 are n = 20 with one uncorrected p = 0.014, and the tedium line is the only instruction they were run on. The vLLM and HF backends differ in baseline rate for reasons I did not resolve (KV-cache pressure, sampling), so I never compare across them. 113 of tonight's judge labels came from shortened transcripts. Most of the code was written by an agent under my direction; the hours in the time log are mine.

## 10. What I would do next

1. Plan transplant: paste the instructed agent's first-turn plan into an uninstructed run. If the plan alone protects, the "two copies" reading is confirmed and the object to study is the plan text.
2. A readout that tracks the judge, for instance the probability of the actual fake-green edit at commit attempts, validated by AUROC on labelled rollouts first, and then the component attribution again.
3. Scale. Everything here is one 9B model. The open question that decides whether this matters is whether the gap between "what a direction measures" and "what an instruction does" closes or widens in larger models: bigger models follow instructions more reliably and represent concepts more linearly, and those two trends pull in opposite directions for a probe. The same three measurements on a 30B-class model and on a second environment from the Singh et al. suite is the experiment, and the pipeline (vLLM sweeps, hooked steering cells, judge, geometry) runs unchanged.

## References

1. A. Singh, G. Kroiz, S. Rajamanoharan, N. Nanda. Model Forensics: Investigating Whether Concerning Behavior Reflects Misalignment. arXiv:2606.26071, 2026.
2. A. Mishra, D. Khashabi, A. Liu. Steered LLM Activations are Non-Surjective. arXiv:2604.09839, 2026.
3. G. Heyman, F. Vandeputte. Steer Like the LLM: Activation Steering that Mimics Prompting. ICML 2026; arXiv:2605.03907.
4. D. Kang, Z. Liu, N. Ma, Y. Huang, Z. Tan, M. Jiang. Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions. arXiv:2605.10664, 2026.
5. J. Braun, C. Eickhoff, D. Krueger, S. A. Bahrainian, D. Krasheninnikov. Understanding (Un)Reliability of Steering Vectors in Language Models. arXiv:2505.22637, 2025.
6. R. Chen, A. Arditi, H. Sleight, O. Evans, J. Lindsey. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509, 2025.
7. A. Mehta, A. Datta. Plans Don't Persist: Why Context Management Is Load Bearing for LLM Agents. arXiv:2606.22953, 2026.
8. A. Stolfo, V. Balachandran, S. Yousefi, E. Horvitz, B. Nushi. Improving Instruction-Following in Language Models through Activation Steering. ICLR 2025; arXiv:2410.12877.
