You write contrastive sentence pairs to extract a linear direction for the concept: **{{concept}}**.

Meaning of this concept:
{{concept_definition}}

Return a JSON list of {{n}} objects:
{"plus": "<sentence that HAS the concept>", "minus": "<matched sentence that LACKS it>"}

Hard rules:
- plus/minus similar length, syntax, and topic; only the concept differs.
- Vocabulary diversity ACROSS the batch (separate from the rule above, which is about WITHIN one pair): don't let any word that signals {{concept}} itself — or an obvious synonym/same-family variant of one already used (e.g. "endless," "slog," and "monotonous" count as one family) — appear on the same side (all plus, or all minus) in more than 2 of the {{n}} pairs. This caps concept-signaling adjectives, intensifiers, and verbs only, not domain nouns (those are already varied by "Mix domains" below). Once a word/family hits its cap, don't reach for a rarer synonym — change the concrete scenario, action, or consequence instead, so the concept comes through what happens, not a repeated label. If a per-concept rule below anchors the concept to one specific phrase (e.g. compliance's "don't cheat / follow the spec"), that anchor phrase is exempt from this cap — vary the surrounding scenario around it instead.
- Self-check before returning the JSON: lay all plus sentences and all minus sentences side by side and tally concept-signaling words/families per side. Any word present on one side in more than 2 pairs and absent from the other lets a classifier detect {{concept}} from word presence alone, which survives word-order scrambling and will fail validation even at 100% accuracy — rewrite every pair past the 2nd occurrence before returning.
- English, 1–2 sentences each.
- Mix domains: workplace, school, cooking, travel, software — but **do not** mention: pre-commit, mypy, py.typed, 258 type errors, git hooks, this paper's environment.
- No numbered lists, no quotes around the JSON keys beyond JSON itself.
- For eval_aware: include pairs that **decorrelate format from eval** (casual wording that is still an eval; formal/benchmark-looking wording that is a real deployed user).
- For disapproval: this is the user's attitude toward a violation, NOT the instruction "don't take shortcuts."
- For tedium: amount of boring work, not difficulty-as-intelligence and not eval.
- For desperate: emotional last-resort / cornered, not merely busy.
- For shortcut: choosing an illegitimate easy path vs doing the real work, in non-coding and generic-coding settings that are NOT git hooks.
- For completion_drive: the *need to pass / get the green check / be done* vs caring that the underlying work is actually complete. Not conscientiousness, not "the user would be mad." Passing by cheating still counts as plus.
- For compliance: explicit instruction-following ("do it properly / don't cheat / follow the spec") vs a matched request with no such extra line. Not user emotion.
- For conscientiousness: thorough, careful, owns quality vs slapdash / not my problem. Not "this is tedious" and not "must pass the check."

CONCEPT_DEFINITION extra notes:
{{notes}}
