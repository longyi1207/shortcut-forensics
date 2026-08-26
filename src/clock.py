"""clock.json + STATUS.md heartbeat. SPEC.md §2 / prompt §2.

Atomic writes (tmp file + os.replace) so a crash mid-write never corrupts the
file a resuming session reads.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SPOT_RATE_USD_PER_HOUR = 0.68
PAYG_RATE_USD_PER_HOUR = 3.67
BUDGET_CAP_USD = 600.0
CREDITS_CONFIRM_THRESHOLD_USD = 30.0


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Clock:
    run_id: str
    phase_active: list[int] = field(default_factory=list)
    phase_done: list[int] = field(default_factory=list)
    gpu_hours: float = 0.0
    usd_est: float = 0.0
    usd_cap: float = BUDGET_CAP_USD
    usd_remaining: float = BUDGET_CAP_USD
    credits_confirmed: bool = False
    credits_check: dict = field(
        default_factory=lambda: {"at_usd": None, "method": None, "balance_usd": None, "quota_id": None}
    )
    spot: bool = True
    n_gpus: int = 0
    rollouts_done: int = 0
    shortcuts: int = 0
    eta_phase_complete_utc: str | None = None
    autokill_utc: str | None = None
    blocker: str | None = None
    updated_utc: str = field(default_factory=now_utc_iso)

    @classmethod
    def load(cls, path: Path) -> "Clock":
        data = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load_or_new(cls, path: Path, run_id: str) -> "Clock":
        if path.exists():
            return cls.load(path)
        return cls(run_id=run_id)

    def save(self, path: Path) -> None:
        self.updated_utc = now_utc_iso()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".clock.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(self), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def add_gpu_hours(self, hours: float) -> None:
        self.gpu_hours += hours
        rate = SPOT_RATE_USD_PER_HOUR if self.spot else PAYG_RATE_USD_PER_HOUR
        self.usd_est += hours * rate * max(self.n_gpus, 1)
        self.usd_remaining = self.usd_cap - self.usd_est

    def credits_gate_ok(self) -> bool:
        """SPEC §2.1 hard gate: once usd_est >= $30, credits must be confirmed."""
        return self.usd_est < CREDITS_CONFIRM_THRESHOLD_USD or self.credits_confirmed


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn last line from a mid-write crash
    return out


def read_existing_ids(path: Path, id_key: str = "id") -> set[str]:
    """Resume support: ids already present in a JSONL file, so a relaunched
    phase can skip work it already did. SPEC.md §4 'resume = skip ids
    already present'."""
    return {obj[id_key] for obj in read_jsonl(path) if id_key in obj}


def append_jsonl(path: Path, obj: dict) -> None:
    """Append one JSON object, fsync before returning. Crash-safe per SPEC §4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def log_decision(run_dir: Path, type_: str, detail: dict) -> None:
    append_jsonl(
        run_dir / "decisions.jsonl",
        {"ts": now_utc_iso(), "type": type_, "detail": detail},
    )


def log_ledger(run_dir: Path, event: str, gpu_hours: float, usd: float, detail: dict | None = None) -> None:
    append_jsonl(
        run_dir / "ledger.jsonl",
        {"ts": now_utc_iso(), "event": event, "gpu_hours": gpu_hours, "usd": usd, "detail": detail or {}},
    )


def write_status_md(
    run_dir: Path,
    clock: Clock,
    *,
    running: str,
    tmux_session: str,
    last_error: str | None,
    next_action: str,
    resume_command: str,
) -> None:
    """Human-readable, cold-resume-complete STATUS.md. SPEC.md §2 / prompt §2."""
    path = run_dir / "STATUS.md"
    lines = [
        f"# STATUS — {clock.run_id}",
        "",
        f"_Last updated: {now_utc_iso()}_",
        "",
        f"**Phase active:** {clock.phase_active}  ",
        f"**Phase done:** {clock.phase_done}  ",
        f"**Blocker:** {clock.blocker or 'none'}  ",
        "",
        "## Clock",
        "",
        f"- GPU-hours: {clock.gpu_hours:.2f}",
        f"- USD est: ${clock.usd_est:.2f} / ${clock.usd_cap:.0f} cap (${clock.usd_remaining:.2f} remaining)",
        f"- Credits confirmed: {clock.credits_confirmed}",
        f"- Spot: {clock.spot}, N GPUs: {clock.n_gpus}",
        f"- Rollouts done: {clock.rollouts_done}, shortcuts: {clock.shortcuts}",
        f"- ETA phase complete (UTC): {clock.eta_phase_complete_utc or 'unknown'}",
        f"- Autokill (UTC): {clock.autokill_utc or 'unset'}",
        "",
        "## What is running",
        "",
        running,
        "",
        f"tmux session: `{tmux_session}`",
        "",
        "## Last error",
        "",
        last_error or "none",
        "",
        "## Next action",
        "",
        next_action,
        "",
        "## Cold-resume command",
        "",
        "```bash",
        resume_command,
        "```",
        "",
    ]
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, path)
