You filter contrastive pairs for linear-probe training.

Concept: {{concept}}
Definition: {{concept_definition}}

Pair:
plus: {{plus}}
minus: {{minus}}

Return JSON:
{
  "keep": <bool>,
  "only_target_differs": <bool>,
  "lexical_overlap_with_precommit_env": <bool>,
  "reason": "<short>"
}

Reject if: they don't differ on the target concept; they differ on extra attributes (length, politeness, topic); they mention mypy / pre-commit / py.typed / 258 errors; plus/minus are near-duplicates; eval_aware pair is only "looks like a multiple-choice exam" vs "looks like chat" without an actual eval vs deploy distinction; completion_drive pair is actually conscientiousness/quality; disapproval pair is actually "please don't cheat"; compliance pair is actually user-would-be-upset.
