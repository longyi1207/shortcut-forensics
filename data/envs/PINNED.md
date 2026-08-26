# Vendored: agent-interp-envs

Source: https://github.com/gkroiz/agent-interp-envs
Pinned commit: 56fd0c11e6cb973b9e1f752ba7c1f35ec3f570bb
Cloned: 2026-08-13

We use `environments/precommit_hook/` content only (src_258, src_20 [generated
locally, see _gen_easy_variant.py — not upstream], src_0, pyproject.toml,
tools.py schema, score.py regex prefilter) plus `src/agent_interp_envs/`
tool_calling.py's EXECUTE_COMMAND_TOOL definition as a reference for the tool
schema. We do NOT use their Docker/Modal runner or API providers — our
subject model (Qwen, open-weight) runs in-process via transformers on the
Azure GPU VM so we can register residual-stream hooks, which none of their
API-based providers support. Our own src/agent_loop.py + src/env_precommit.py
reimplement the same task setup (system/user prompt from
configs/precommit_hook/default.yaml, single `execute_command` tool, one
command per turn, plain-text-no-tool-call ends session) against a local
tmpdir instead of their Docker container.
