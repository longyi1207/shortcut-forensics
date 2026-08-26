#!/usr/bin/env python3
"""State-machine controller. Default entrypoint on the Azure GPU VM
(tmux session `shortcut`). Survives a dead Claude Code session: cold resume
is `outputs/<run_id>/STATUS.md` + `clock.json` only. SPEC.md §6.1 / prompt §3.

Usage (on the VM, inside tmux):
    python scripts/controller.py --run-id 20260813-120000

Loop, once per POLL_SECONDS:
  1. heartbeat clock.json + STATUS.md
  2. enforce the credits-vs-card hard gate (prompt §2.1) — every heartbeat,
     not just phase boundaries
  3. enforce budget cap / autokill
  4. launch any phase whose deps are satisfied and isn't already running
     (GPU phases run one at a time; non-GPU phases may run alongside)
  5. reap finished subprocesses, retry crashes with --resume (bounded)
  6. pull.sh every PULL_INTERVAL_S
  7. exit 0 when all phases 0-8 are done (COMPLETE) or a hard stop fires
     (ABORTED / credits_unconfirmed / autokill-approaching)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_phase import (  # noqa: E402
    PHASE_NEEDS_GPU,
    is_shortcut,
    load_config,
    phase_deps_satisfied,
    read_phase_status,
    run_dir_for,
)
from src.clock import (  # noqa: E402
    CREDITS_CONFIRM_THRESHOLD_USD,
    Clock,
    log_decision,
    log_ledger,
    read_jsonl,
    write_status_md,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s controller: %(message)s")
logger = logging.getLogger("controller")

POLL_SECONDS = 60
HEARTBEAT_SECONDS = 300  # <=5 min, SPEC.md §2
PULL_INTERVAL_SECONDS = 900  # 15 min, SPEC.md §3.6
AUTOKILL_HOURS = int(os.environ.get("SCFX_AUTOKILL_HOURS", "8"))
AUTOKILL_MARGIN_MINUTES = 30
MAX_PHASE_RETRIES = 3
# Distinctly namespaced, not "shortcut" -- this runs on a SHARED box (the
# GPU cluster is split across 3 coordinated experiments); a generic name
# collided with an independent session already (2026-08-21).
TMUX_SESSION = os.environ.get("SCFX_TMUX_SESSION", "scfx_ly")

ACTIVE_ENV_PATH = REPO_ROOT / "infra" / ".active"


def find_vm_env(run_id: str) -> Path | None:
    p = ACTIVE_ENV_PATH / f"{run_id}.env"
    return p if p.exists() else None


def read_env_file(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v
    return out


def deallocate_self(run_id: str) -> None:
    """Fire-and-forget stop against our own compute. This process will be
    killed mid-flight when it stops; that's fine — the JSONL/clock state on
    disk is already fsynced, and Azure's control plane processes the
    stop/deallocate over the network independent of the local process dying.

    Backend-aware: launch.sh (raw VM, BACKEND unset) uses `az vm
    deallocate`; launch_aml.sh (Azure ML Compute Instance, BACKEND=aml) uses
    `az ml compute stop` -- a compute instance is not a directly-addressable
    Microsoft.Compute/virtualMachines resource, so `az vm deallocate`
    against it would just fail to find anything."""
    env_path = find_vm_env(run_id)
    if env_path is None:
        logger.error("no infra/.active/%s.env found; cannot self-stop. Deallocate/stop manually.", run_id)
        return
    env = read_env_file(env_path)
    # On the VM/compute, auth via a system-assigned managed identity granted
    # a role scoped to just this resource (infra/launch.sh / launch_aml.sh).
    # Off-box (e.g. this function invoked manually from the laptop for
    # testing), an existing `az login` user session is used instead — this
    # call is a harmless no-op failure in that case.
    subprocess.run(["az", "login", "--identity", "--output", "none"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    if env.get("BACKEND") == "aml":
        compute, rg, ws = env.get("AML_COMPUTE_NAME"), env.get("AML_RESOURCE_GROUP"), env.get("AML_WORKSPACE")
        if not (compute and rg and ws):
            logger.error("infra/.active/%s.env missing AML_COMPUTE_NAME/AML_RESOURCE_GROUP/AML_WORKSPACE", run_id)
            return
        logger.warning("self-stopping AML compute instance %s in %s/%s ...", compute, rg, ws)
        subprocess.Popen(
            ["az", "ml", "compute", "stop", "--name", compute, "--resource-group", rg, "--workspace-name", ws, "--no-wait"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        vm_name, rg = env.get("AZURE_VM_NAME"), env.get("AZURE_RESOURCE_GROUP")
        if not vm_name or not rg:
            logger.error("infra/.active/%s.env missing AZURE_VM_NAME/AZURE_RESOURCE_GROUP", run_id)
            return
        logger.warning("self-deallocating VM %s in %s ...", vm_name, rg)
        subprocess.Popen(
            ["az", "vm", "deallocate", "--name", vm_name, "--resource-group", rg, "--no-wait"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def run_pull(run_id: str) -> None:
    """Mirror outputs/ to the VM's data disk (survives Deallocate). This runs
    ON the VM, so it can't also SSH-pull to the laptop (wrong direction) —
    that half happens from the laptop side (`infra/pull.sh <run_id>`,
    manually or via a resuming session), per SPEC.md §3.6."""
    pull_sh = REPO_ROOT / "infra" / "pull.sh"
    if not pull_sh.exists():
        return
    try:
        subprocess.run(["bash", str(pull_sh), run_id, "--vm-side"], timeout=120, check=False)
    except Exception as e:  # noqa: BLE001 — pull failures must not crash the controller
        logger.warning("pull.sh --vm-side failed (non-fatal): %s", e)


class PhaseRunner:
    """Tracks one in-flight `run_phase.py --phase N` subprocess."""

    def __init__(self, phase: int, run_id: str):
        self.phase = phase
        self.run_id = run_id
        self.proc: subprocess.Popen | None = None
        self.retries = 0
        self.log_path = REPO_ROOT / "outputs" / run_id / "logs" / f"phase{phase}.log"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(self.log_path, "a")
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_phase.py"), "--phase", str(self.phase), "--run-id", self.run_id, "--resume"]
        logger.info("launching: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))

    def poll(self) -> int | None:
        if self.proc is None:
            return None
        return self.proc.poll()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def gpu_hours_since(run_dir: Path, since_iso: str) -> float:
    """Estimate GPU-hours burned since a timestamp from ledger.jsonl."""
    ledger_path = run_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return 0.0
    total = 0.0
    for line in ledger_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ts", "") >= since_iso:
            total += row.get("gpu_hours", 0.0)
    return total


def run_controller(run_id: str, once: bool = False) -> None:
    run_dir = run_dir_for(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    clock_path = run_dir / "clock.json"
    clock = Clock.load_or_new(clock_path, run_id)

    start_time = datetime.now(timezone.utc)
    if not clock.autokill_utc:
        clock.autokill_utc = (start_time + timedelta(hours=AUTOKILL_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    gpu_hours_baseline = clock.gpu_hours
    wall_start = time.time()

    gpu_runner: PhaseRunner | None = None
    non_gpu_runners: dict[int, PhaseRunner] = {}
    last_heartbeat = 0.0
    last_pull = 0.0

    logger.info("controller starting run_id=%s autokill_utc=%s", run_id, clock.autokill_utc)

    while True:
        now = time.time()

        # --- 1. heartbeat clock + STATUS.md (accrue GPU-hours from wall time while a GPU phase runs) ---
        if gpu_runner is not None and gpu_runner.is_running():
            elapsed_h = (now - wall_start) / 3600
            clock.add_gpu_hours(max(0.0, elapsed_h - gpu_hours_baseline))
            gpu_hours_baseline = elapsed_h
            log_ledger(run_dir, "gpu_running", elapsed_h, clock.usd_est)

        if now - last_heartbeat >= HEARTBEAT_SECONDS or once:
            done_phases = [n for n in range(9) if read_phase_status(run_dir, n).get("done")]
            clock.phase_done = done_phases
            clock.phase_active = [r.phase for r in ([gpu_runner] if gpu_runner else []) + list(non_gpu_runners.values()) if r and r.is_running()]
            # Real progress, not just phase booleans -- these fields existed
            # in Clock from the start but were never actually populated,
            # so STATUS.md/clock.json looked permanently stuck at 0 even
            # while real rollout collection was happening.
            rows = read_jsonl(run_dir / "rollouts.jsonl")
            ok_rows = [r for r in rows if r.get("status") == "ok"]
            clock.rollouts_done = len(ok_rows)
            clock.shortcuts = sum(is_shortcut(r) for r in ok_rows)
            clock.save(clock_path)
            write_status_md(
                run_dir, clock,
                running=f"phase(s) active: {clock.phase_active or 'none'}; done: {clock.phase_done}; rollouts_done={clock.rollouts_done} shortcuts={clock.shortcuts}",
                tmux_session=TMUX_SESSION,
                last_error=None,
                next_action="controller loop continues automatically" if not clock.blocker else f"BLOCKED: {clock.blocker}",
                resume_command=f"tmux new -s {TMUX_SESSION} 'cd {REPO_ROOT} && python scripts/controller.py --run-id {run_id}'",
            )
            last_heartbeat = now

        # --- 2. credits-vs-card hard gate (prompt §2.1) — every heartbeat ---
        if clock.usd_est >= CREDITS_CONFIRM_THRESHOLD_USD and not clock.credits_confirmed:
            clock.blocker = "credits_unconfirmed"
            clock.save(clock_path)
            logger.error("HARD STOP: usd_est=%.2f >= $%d and credits not confirmed. Deallocating.", clock.usd_est, CREDITS_CONFIRM_THRESHOLD_USD)
            log_decision(run_dir, "kill_rule", {"reason": "credits_unconfirmed_at_threshold", "usd_est": clock.usd_est})
            deallocate_self(run_id)
            write_status_md(
                run_dir, clock, running="STOPPED", tmux_session=TMUX_SESSION,
                last_error="credits_unconfirmed", next_action="LY must confirm Azure credits (SPEC.md §2.1) before this can resume.",
                resume_command=f"# after confirming credits, set clock.json credits_confirmed=true, then:\ntmux new -s {TMUX_SESSION} 'cd {REPO_ROOT} && python scripts/controller.py --run-id {run_id}'",
            )
            return

        # --- 3. budget cap ---
        if clock.usd_est >= clock.usd_cap:
            clock.blocker = "budget_exhausted"
            clock.save(clock_path)
            logger.error("HARD STOP: usd_est=%.2f >= cap=%.2f", clock.usd_est, clock.usd_cap)
            log_decision(run_dir, "kill_rule", {"reason": "budget_exhausted", "usd_est": clock.usd_est})
            deallocate_self(run_id)
            return

        # --- 4. autokill approaching ---
        autokill_dt = datetime.strptime(clock.autokill_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) >= autokill_dt - timedelta(minutes=AUTOKILL_MARGIN_MINUTES):
            clock.blocker = "needs_relaunch"
            clock.save(clock_path)
            logger.warning("autokill window approaching (%s); checkpointing and exiting 0 for relaunch.", clock.autokill_utc)
            log_decision(run_dir, "kill_rule", {"reason": "autokill_approaching", "autokill_utc": clock.autokill_utc})
            run_pull(run_id)
            write_status_md(
                run_dir, clock, running="STOPPED for autokill window", tmux_session=TMUX_SESSION,
                last_error=None, next_action="infra/relaunch.sh (cron on VM) restarts the controller after the VM cycles.",
                resume_command=f"tmux new -s {TMUX_SESSION} 'cd {REPO_ROOT} && python scripts/controller.py --run-id {run_id}'",
            )
            return

        # --- 5. reap finished subprocesses ---
        if gpu_runner is not None and not gpu_runner.is_running():
            rc = gpu_runner.proc.returncode
            if rc == 0:
                logger.info("phase %d finished ok", gpu_runner.phase)
                gpu_runner = None
            else:
                gpu_runner.retries += 1
                logger.error("phase %d exited %d (retry %d/%d)", gpu_runner.phase, rc, gpu_runner.retries, MAX_PHASE_RETRIES)
                log_decision(run_dir, "retry", {"phase": gpu_runner.phase, "returncode": rc, "attempt": gpu_runner.retries})
                if gpu_runner.retries >= MAX_PHASE_RETRIES:
                    incidents_dir = run_dir / "incidents"
                    incidents_dir.mkdir(parents=True, exist_ok=True)
                    (incidents_dir / f"phase{gpu_runner.phase}_{int(now)}.md").write_text(
                        f"# Incident: phase {gpu_runner.phase} failed {gpu_runner.retries}x\n\n"
                        f"See outputs/{run_id}/logs/phase{gpu_runner.phase}.log for the traceback.\n"
                        f"SPEC.md §5: same failure 3x -> incident + SPEC fallback; 4th time -> stop that branch.\n"
                    )
                    clock.blocker = f"phase{gpu_runner.phase}_failed_{gpu_runner.retries}x"
                    clock.save(clock_path)
                    write_status_md(
                        run_dir, clock, running="STOPPED", tmux_session=TMUX_SESSION,
                        last_error=f"phase {gpu_runner.phase} failed {gpu_runner.retries}x, see incidents/",
                        next_action="Needs a human or fresh Claude Code session to read the incident + apply a SPEC.md fallback.",
                        resume_command=f"tmux new -s {TMUX_SESSION} 'cd {REPO_ROOT} && python scripts/controller.py --run-id {run_id}'",
                    )
                    return
                gpu_runner.start()

        for phase, runner in list(non_gpu_runners.items()):
            if not runner.is_running():
                rc = runner.proc.returncode
                if rc == 0:
                    del non_gpu_runners[phase]
                else:
                    runner.retries += 1
                    log_decision(run_dir, "retry", {"phase": phase, "returncode": rc, "attempt": runner.retries})
                    if runner.retries >= MAX_PHASE_RETRIES:
                        del non_gpu_runners[phase]  # give up quietly on this non-GPU phase; controller keeps going
                    else:
                        runner.start()

        # --- 6. launch newly-eligible phases ---
        all_done = all(read_phase_status(run_dir, n).get("done") for n in range(9))
        if all_done:
            break

        for phase in range(9):
            if read_phase_status(run_dir, phase).get("done"):
                continue
            ok, _reason = phase_deps_satisfied(run_dir, phase, cfg)
            if not ok:
                continue
            needs_gpu = PHASE_NEEDS_GPU[phase]
            if needs_gpu:
                if gpu_runner is None:
                    gpu_runner = PhaseRunner(phase, run_id)
                    gpu_runner.start()
            else:
                if phase not in non_gpu_runners:
                    r = PhaseRunner(phase, run_id)
                    r.start()
                    non_gpu_runners[phase] = r

        # --- 7. periodic pull ---
        if now - last_pull >= PULL_INTERVAL_SECONDS:
            run_pull(run_id)
            last_pull = now

        if once:
            logger.info("--once: single iteration done, not all phases complete yet, exiting without COMPLETE/deallocate")
            return
        time.sleep(POLL_SECONDS)

    # only reachable via `if all_done: completed = True; break` above
    clock.blocker = None
    clock.save(clock_path)
    run_pull(run_id)
    write_status_md(
        run_dir, clock, running="COMPLETE", tmux_session=TMUX_SESSION,
        last_error=None, next_action="none — experiment complete. Deallocate the VM if not already.",
        resume_command="# nothing to resume; run is COMPLETE",
    )
    (run_dir / "STATUS.md").write_text((run_dir / "STATUS.md").read_text() + "\n\n**COMPLETE**\n")
    logger.info("all phases done. run_id=%s COMPLETE", run_id)
    deallocate_self(run_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--once", action="store_true", help="single loop iteration, for testing")
    args = ap.parse_args()
    run_controller(args.run_id, once=args.once)


if __name__ == "__main__":
    main()
