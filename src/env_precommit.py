"""Pre-commit-hook task setup + regex prefilter scorer. SPEC.md §4.3.

Mirrors data/envs/agent-interp-envs/environments/precommit_hook/entrypoint.py
(same src_N variant, same pyproject.toml, same initial --no-verify commit)
but against a local tempdir instead of a Docker container, since the subject
model runs in-process (transformers) for hook access and doesn't need
container isolation from the harness the way an API-called model would.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = REPO_ROOT / "data" / "envs" / "agent-interp-envs" / "environments" / "precommit_hook"
CONFIGS_DIR = REPO_ROOT / "data" / "envs" / "agent-interp-envs" / "configs" / "precommit_hook"

sys.path.insert(0, str(ENV_ROOT))
import score as _upstream_score  # noqa: E402  (vendored score_precommit_hook)

DEFAULT_CONFIG = yaml.safe_load((CONFIGS_DIR / "default.yaml").read_text())
SYSTEM_PROMPT = DEFAULT_CONFIG["prompts"]["system_prompt"]
USER_PROMPT = DEFAULT_CONFIG["prompts"]["user_prompt"]

# Commands that would escape the sandboxed task workspace or nuke the VM.
# Not a security boundary (this is a disposable spot VM we own) — just a
# guard against an agent going wildly off-script and eating the whole disk.
_DANGEROUS_PATTERNS = ("rm -rf /", "rm -rf /*", "rm -rf ~", "rm -rf $HOME", "mkfs", ":(){:|:&};:")


class TaskWorkspace:
    """One disposable pre-commit-hook task instance in its own tempdir."""

    def __init__(self, target_errors: int, base_tmp_dir: Path | None = None):
        self.target_errors = target_errors
        src_variant = ENV_ROOT / f"src_{target_errors}"
        if not src_variant.exists():
            available = sorted(
                int(d.name.split("_")[1]) for d in ENV_ROOT.iterdir() if d.is_dir() and d.name.startswith("src_")
            )
            raise FileNotFoundError(f"src_{target_errors}/ not found. Available: {available}")

        self.workdir = Path(tempfile.mkdtemp(prefix=f"shortcut_task_{target_errors}_", dir=base_tmp_dir))
        self.commands: list[str] = []
        self._setup(src_variant)

    def _setup(self, src_variant: Path) -> None:
        src_dir = self.workdir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        for f in src_variant.iterdir():
            if f.is_file():
                shutil.copy(f, src_dir / f.name)
        shutil.copy(ENV_ROOT / "pyproject.toml", self.workdir / "pyproject.toml")

        def git(*args):
            return subprocess.run(["git", *args], cwd=self.workdir, capture_output=True, text=True, check=True)

        git("init")
        git("config", "user.email", "dev@example.com")
        git("config", "user.name", "Developer")
        git("add", "-A")
        result = subprocess.run(
            ["git", "commit", "--no-verify", "-m", "Initial commit: data ingestion service"],
            cwd=self.workdir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"initial commit failed: {result.stdout}\n{result.stderr}")

    def run_command(self, command: str, timeout: int = 30) -> tuple[int, str]:
        self.commands.append(command)
        if any(p in command for p in _DANGEROUS_PATTERNS):
            return 1, "ERROR: command blocked by workspace safety guard."
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            return r.returncode, r.stdout.decode(errors="replace")
        except subprocess.TimeoutExpired:
            return 0x4B, f"<TIMEOUT after {timeout}s>{command}</TIMEOUT>"

    def diff_text(self, max_chars: int = 20000) -> str:
        """Diff of everything changed since the initial commit (staged + unstaged + untracked)."""
        parts = []
        r1 = subprocess.run(["git", "diff", "HEAD"], cwd=self.workdir, capture_output=True, text=True)
        parts.append(r1.stdout)
        r2 = subprocess.run(
            ["git", "status", "--porcelain"], cwd=self.workdir, capture_output=True, text=True
        )
        untracked = [line[3:] for line in r2.stdout.splitlines() if line.startswith("??")]
        for rel in untracked:
            p = self.workdir / rel
            if p.is_file():
                try:
                    parts.append(f"\n--- new file: {rel} ---\n{p.read_text(errors='replace')[:2000]}")
                except OSError:
                    pass
        hook_path = self.workdir / ".git" / "hooks" / "pre-commit"
        if hook_path.exists():
            parts.append(f"\n--- .git/hooks/pre-commit ---\n{hook_path.read_text(errors='replace')}")
        text = "\n".join(parts)
        return text[:max_chars]

    def prefilter_score(self) -> dict:
        """Cheap regex prefilter (SPEC.md §4.2(a)) — the judge still confirms."""
        return _upstream_score.score_precommit_hook(self.commands, self.workdir)

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


def user_prompt_with_commit_msg(commit_msg: str = "release: v1.0.0") -> str:
    return USER_PROMPT
