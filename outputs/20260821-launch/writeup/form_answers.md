# MATS Form Answers

**DRAFT — LY rewrites voice. Not submitted anywhere by this pipeline.**

- **What we did:** We built the full pipeline end-to-end: set up the environment/GPU, identified candidate residual-stream layers (19–24 of 32), ran the initial shortcut probe on Qwen/Qwen3.5-9B, then trained and validated eight concepts with 80/20 splits each. We also ran readouts, plant tests, and intervention sweeps across identity, prompt, ablation, and feature-addition conditions.

- **What we found:** Several concepts were strong and mechanistically useful: **tedium** (val_acc 1.0, scrambled 0.65, plant_took_rate 1.0), **completion_drive** (val_acc 1.0, scrambled 0.35), **shortcut** (val_acc 0.95, scrambled 0.55), and **disapproval/desperate/compliance** also passed gates. In readout/plant tests, **tedium** and **compliance** had the strongest pro-cheat signals (AUC 0.759 and 0.722), and the final winner was **tedium**.

- **What we’d do with more time/budget:** We’d collect more plants and larger eval sets for the weaker or ambiguous concepts, especially **eval_aware** (failed gate despite val_acc 0.95) and **disapproval** (AUC ~0.51, mixed intervention effects). We’d also expand intervention coverage beyond layer 19 and test whether the same mechanisms generalize across more models and tasks.

- **One thing that surprised us:** **Tedium** emerged as the clearest winner even though it wasn’t the most obvious “cheating” concept a priori. It showed very strong validation, strong readout/plant evidence, and the cleanest intervention signal, while some intuitively relevant concepts like **eval_aware** looked good on validation but did not pass the gate.
