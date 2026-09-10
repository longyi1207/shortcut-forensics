# Neel MATS 12.0, form answers, v2 (copy-paste; personal facts marked [check] need your confirmation)

### What question did you try to answer?

When a one-line instruction in the prompt changes what an agent does, and a contrast-pair steering vector for the "same" concept also changes what it does, are they acting on the same thing inside the model? Steering vectors get read as "the model's representation of X" and prompts as a way of turning X on or off, so the two are usually treated as two handles on one lever. I tested that per factor, on a behaviour that is an action with a ground-truth label.

The setting is Singh, Kroiz, Rajamanoharan and Nanda's pre-commit environment (arXiv:2606.26071): a coding agent in a repo with 258 seeded mypy errors behind a hook, which sometimes fakes a green check instead of doing the work. For five candidate motives (tedium, desperation, temptation, wanting to be done, fear of the user's disapproval) I fitted a contrast direction, wrote a matched instruction, and measured three things: what the instruction does to the shortcut rate, what steering the direction does to it, and whether the instruction's footprint on the residual stream lies along the direction at all.

### Why is this question interesting / why did you choose it?

Both interventions are in use, and the assumption that they are interchangeable has not been checked where it matters. Prompting is the mitigation people ship; contrast directions are what people fit when they want to monitor or steer a disposition; persona vectors (Chen et al. 2025, arXiv:2507.21509), contrast directions fitted for character traits, are built on the premise that a prompt's effect projects onto the fitted direction. Recent work questions the equivalence from the steering side, but on persona and style benchmarks, single turn, judged by output quality. Nobody had put the two side by side on a long agentic task, per concept, with behavioural labels. The answer bears on how we should make a model follow behavioural instructions, and how we monitor for when it does not.

### What conclusions have you reached about this research problem?

1. Every one of the five instructions cuts the cheating rate, from 31% to 0-11% (n = 90 each). A same-length line with no concept content does nothing (30%, n = 150).
2. Steering works for one direction, tedium (ablate 9%, add 53%, baseline 20%, random direction 25%). The other four are null at n = 30 to 43.
3. Even for tedium, the instruction does not move the model along the direction. The concept's own sentence does; the instruction has zero projection, at the prompt and at the decision.
4. The instruction reaches the decision through the context, twice over: the instruction text and the plan the agent wrote from it. Block both and cheating returns (25% vs 3.8%); block either and it does not. One instruction, n = 20, so a lead rather than a finding.
5. So a direction can steer, and read its own concept, and still not see behaviour a prompt controls. That is the property a monitor needs, and contrast-pair directions do not give it.

### Technical setup

Model: Qwen3.5-9B, thinking on. Hardware: 8×A100 on Azure.
Environment: Singh et al.'s repo, 258 seeded type errors, a pre-commit mypy hook, 80 tool turns with real git.
Scoring: an LLM judge (gpt-5.4-mini, temperature 0) labels each rollout shortcut or not; the shortcuts are fake greens, never --no-verify. I labelled 30 rollouts by hand to check the judge (30/30 agreement).
Directions: mean difference of last-token residuals over 160 LLM-written contrast pairs per factor, at layer 19, with held-out, lexical-scramble and plant gates.
Instructions: one 70-token IMPORTANT line per factor at the end of the user prompt; the control is a same-length line about shell formatting.
Steering: remove the direction's component at every position of layer 19 (ablate) or add it at the fitted norm; random directions as controls.
Geometry: the residual shift a line causes, projected on the direction, at the first assistant token and at 40 decision tokens, with the concept's own sentence as the positive control.
Sample sizes: 90 per instruction and 150 per control (vLLM); 30 to 153 per steering cell (HuggingFace hooks). The two backends have different baselines and are never compared across. About 2,300 judged rollouts.

### What is the strongest evidence you found against these hypotheses?

I expected the instruction to work by moving the model along the concept direction, but it does not: its projection is zero, and pushing the tedium direction on top of the instruction gives 50%, the same as pushing without it (53%). I then expected the instruction's own mean footprint to be the carrier, as in instruction-vector work. Adding it to a baseline run gives 1/12; removing it from an instructed run gives 0/12. Neither sufficient nor necessary.

### What are the biggest limitations to your results? Could you have addressed them?

Sample-level: one model, one task, one judge (checked by hand on 30 rollouts); the four null directions were only removed, never added, so if a state was never on, removal shows nothing; the mechanism result is one instruction at n = 20. More GPU time and a second task would have addressed these.

Two deeper limits the design cannot fix. First, "not the same thing" here means "not the same linear direction at the positions I measured, with the steering recipe I used". A shared cause that is nonlinear, spread over several directions, or present only at certain moments would be invisible to this test. And the directions themselves are fitted on sentences about tedium; the positive control shows they read text about tedium, not that the model has a state of being tedious, which this design cannot establish.

Second, the plan channel is almost certainly a trained behaviour: post-training teaches models to restate the task and follow their plan. I showed the plan carries the instruction, not how the instruction becomes a plan, whether the plan is followed as a plan or just as more tokens, or whether it can be manipulated, for instance by injecting text into the agent's notes. Answering that needs base-versus-instruct comparisons or training interventions, not more rollouts.

### How did you use LLMs in this research task?

I read the literature and designed the experiments myself. I used Claude Code to brainstorm, never to decide. Once the design was fixed, Claude Code did almost all of the implementation and the overnight execution on the cluster, under a plan and controls I set (the neutral line, the random direction, the plant positive control). It also drafted the instruction lines, which I reviewed one by one. gpt-5.4-mini is the judge behind every label; I labelled 30 rollouts by hand with its verdict hidden and agreed on all 30.

Mistakes I caught: an early draft said 19 seeded errors; the config says 258. A tug-of-war compared a push of 1% of the residual norm against one of 26%; I checked the vector norms and had it redone dose-matched. The same condition had been run across several launches with different id prefixes; I checked that they were launch labels, not variants, before pooling. [Optional: A proxy metric the agent had built for cheap attribution turned out not to track the judge labels when I validated it; I dropped every result that rested on it.]

Writing: I drafted the structure and bullet points, the LLM filled in numbers and prose from the results files, and I edited. These form answers are mine.

### What, if any, prior experience do you have with mechanistic interpretability?

No publications; self-directed since I left my startup in May 2026, working from reproductions toward my own questions. I reproduced Anthropic's "Emotion Concepts and Their Function in a LLM" on Llama-3.2-1B: 30 mean-difference directions, 8 of 8 emotions steer the right way, and the readout reproduces while free generation barely moves, which I made the headline. I then designed a pre-registered study of whether theory of mind and empathy are separable directions in open models [check: say "ran" if it ran]. The submitted project is where I learned the causal side: Qwen3.5 is an attention/GatedDeltaNet hybrid no library supports, so every intervention is a forward hook I wrote.

### Other than your research task, what are 1-3 pieces of evidence that you'd be able to do good research in the program?

Engineering. Before research I worked in IT consulting, then as an infrastructure engineer, then as CTO of a YC-backed AI startup (IncidentFox). This project needed that: hooks on an unsupported architecture and an eight-GPU pipeline run overnight.

Research taste. In a SPAR take-home, Anthropic and UK AISI had disagreed publicly about why Claude 4.5 refuses safety-research tasks; I varied one element at a time across 2,339 responses and found the disputed number was two mechanisms added together, so neither side was right. In college I did an interpretability project on attention [check], my first contact with the field.

Selection. Accepted into SPAR Fall 2026; top 5% in [AIAF, full name].

### Why are you interested in Neel's stream specifically?

I have learned much of what I know about this field from your papers and write-ups, and this project is built on one of them. So far I have only had the "what": the methods and the results. Working with you would give me the "why" and the "how", the reasoning behind which questions are worth asking and which results to trust, and that is the part I cannot get from reading. For where I want to go, this is the best learning opportunity I can see.

### What is the likelihood you will join Neel's training program (Sept 28 - Oct 30) if accepted?

Almost certainly.

### (Optional) Is there anything else important I should know about your application project?

Code, data and the full run log are public at github.com/longyi1207/shortcut-forensics. Three quarters of the model is GatedDeltaNet, so the standard attention toolkit reaches a quarter of it; every intervention here is a hook I wrote, which will be true of most models that ship next.
