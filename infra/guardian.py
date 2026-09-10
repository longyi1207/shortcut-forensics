#!/usr/bin/env python
"""Night guardian for the 2026-09-10 run. Runs on the box under nohup, loops every 5 min.

Owns three things so the GPUs never idle and nothing dies silently:
  1. vLLM servers (GPUs 0-2): relaunch a server whose /health fails and whose
     process is gone, or that stays unhealthy for three checks.
  2. The vLLM prompt sweeps (worker ids pd, pt): relaunch if the process is gone
     while any of its variants is below target; when every variant reaches the
     target, raise the target one step (60 -> 90) so the servers stay busy.
  3. HF workers on GPUs 3-7: a priority list of cells with targets; any GPU with
     no HF worker gets the first unmet cell.
Status goes to /mnt/scfx_logs/guardian.log and /mnt/scfx_logs/status.json
(the laptop watcher reads the latter).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

RUN = "/mnt/scfx_ly_run"
LOGDIR = "/mnt/scfx_logs"
PY = f"{RUN}/.venv/bin/python"
ROLLOUTS = f"{RUN}/outputs/20260821-launch/rollouts.jsonl"
SERVERS = {}  # vLLM stage finished 13:30 UTC (all prompt cells at target); port = 8123 + gpu when set
URLS = ",".join(f"http://127.0.0.1:{p}" for p in SERVERS.values())
PD_VARIANTS = ["vllm_identity_r2", "vllm_user_neutral", "vllm_user_desperate", "vllm_user_shortcut",
               "vllm_user_completion", "vllm_user_disapproval"]
PT_VARIANTS = ["vllm_user_tedium_strong_r2"]
# Third sweep: the two control cells every prompt comparison is measured against, taken further once
# the concept lines are done. Runs at 10 slots while pd is alive, 60 once pd has exited.
PB_VARIANTS = ["vllm_identity_r2", "vllm_user_neutral"]
PB_TARGET = 150
TARGET_STEPS = [60, 90]
HF_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
# (cell name, phase for counting, condition, target, launcher env, script args)
HF_CELLS = [
    # control for the one steer effect that replicated (ablate_tedium 9% vs 20%): a random direction ablated
    # the same way; n=10 so far
    ("ar", "signed_pack", "ablate_random", 40,
     {"SCFX_P6_CONDITIONS": "ablate_random", "SCFX_P6_N_OVERRIDE": "40"}, ["scripts/run_phase.py", "--phase", "6", "--run-id", "20260821-launch", "--resume"]),
    ("cd", "signed_pack", "ablate_completion_drive", 30,
     {"SCFX_P6_CONDITIONS": "ablate_completion_drive"}, ["scripts/run_phase.py", "--phase", "6", "--run-id", "20260821-launch", "--resume"]),
    ("rc", "prompt_channel", "pc_prompt_add_rand19", 24,
     {"SCFX_PC_CONDITION": "pc_prompt_add_rand19", "SCFX_PC_PROMPT": "1", "SCFX_PC_N": "24", "SCFX_PC_VECS": "randdir_tedium19",
      "SCFX_PC_MODE": "add", "SCFX_PC_LAYER": "19", "SCFX_PC_ALPHA": "1.0"}, ["scripts/prompt_channel.py"]),
    ("ra", "prompt_channel", "pc_add_rand19", 16,
     {"SCFX_PC_CONDITION": "pc_add_rand19", "SCFX_PC_PROMPT": "0", "SCFX_PC_N": "16", "SCFX_PC_VECS": "randdir_tedium19",
      "SCFX_PC_MODE": "add", "SCFX_PC_LAYER": "19", "SCFX_PC_ALPHA": "1.0"}, ["scripts/prompt_channel.py"]),
]
REJUDGE_PHASES = ["prompt_sweep_vllm", "signed_pack", "prompt_channel"]  # ok rows whose judge call died on a 429
REJUDGE_SIDE = f"{RUN}/outputs/20260821-launch/rejudge.jsonl"
# (shard, extra env): shard 0 on the Azure deployment (150K TPM), shard 1 on direct OpenAI (200K TPM), same model
# The direct-OpenAI route ran out of account credits at 12:12 UTC after 37 verdicts; single Azure pass again.
REJUDGE_SHARDS = [("0/1", {})]
RESERVED_FILE = f"{LOGDIR}/reserved_gpus"  # whitespace-separated GPU indices the HF pool must leave alone
BASE_ENV = {"HF_HOME": "/mnt/scfx_ly_cache", "PYTHONUNBUFFERED": "1", "PATH": os.environ.get("PATH", "")}
state = {"pd_target": 60, "pt_target": 60, "unhealthy": {g: 0 for g in SERVERS}}


def log(msg: str):
    with open(f"{LOGDIR}/guardian.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def health(port: int) -> bool:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status == 200
    except Exception:
        return False


def reserved() -> set:
    try:
        return {int(x) for x in open(RESERVED_FILE).read().split()}
    except Exception:
        return set()


def procs():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            if not any(k in cmd for k in ("prompt_sweep_vllm.py", "prompt_channel.py", "run_phase.py", "vllm serve", "rejudge_phase.py")):
                continue
            env = dict(kv.split("=", 1) for kv in open(f"/proc/{pid}/environ", "rb").read().decode(errors="ignore").split("\0") if "=" in kv)
            out.append((int(pid), cmd, env))
        except Exception:
            continue
    return out


def counts() -> dict:
    c: dict = {}
    with open(ROLLOUTS) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                k = (r.get("phase"), r.get("condition"))
                c[k] = c.get(k, 0) + 1
                if r.get("phase") in REJUDGE_PHASES and not (isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None):
                    c.setdefault("_unjudged", set()).add(r["id"])
    if os.path.exists(REJUDGE_SIDE):
        for line in open(REJUDGE_SIDE):
            if line.strip():
                try:
                    r = json.loads(line)
                    if isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None:
                        c.get("_unjudged", set()).discard(r["id"])
                except Exception:
                    pass
    return c


def launch(env: dict, args: list, logname: str, gpu: int | None = None):
    e = dict(BASE_ENV); e.update(env)
    if gpu is not None:
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(f"{RUN}/logs/{logname}.log", "a") as lf:
        subprocess.Popen(["setsid", PY, *args], cwd=RUN, env=e, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log(f"launched {logname}: {' '.join(args)} gpu={gpu} env={ {k: v for k, v in env.items() if k.startswith('SCFX')} }")


def launch_server(gpu: int):
    subprocess.run(["bash", f"{RUN}/infra/launch_vllm.sh", str(gpu)], cwd=RUN, env={**os.environ, "VLLM_VENV": "/mnt/scfx_vllm3"})
    log(f"relaunched vllm server on GPU {gpu}")


# rollouts in flight per sweep; the total must keep every server's live contexts inside its KV cache
# (75k-token contexts late in a rollout; the cache thrashed at 20 rollouts per server)
CONC = {"pd": int(os.environ.get("SCFX_CONC_PD", "50")), "pt": int(os.environ.get("SCFX_CONC_PT", "10")), "pb": 10}


def sweep_env(variants: list, n: int, wid: str) -> dict:
    return {"SCFX_VLLM_URLS": URLS, "SCFX_SWEEP_N": str(n), "SCFX_SWEEP_CONC": str(CONC[wid]),
            "SCFX_WORKER_ID": wid, "SCFX_SWEEP_VARIANTS": ",".join(variants), "SCFX_LLM_RETRY_S": "90"}


def tick():
    ps = procs()
    c = counts()
    # 1. servers
    all_healthy = True
    for g, port in SERVERS.items():
        ok = health(port)
        have_proc = any("vllm serve" in cmd and env.get("CUDA_VISIBLE_DEVICES") == str(g) for _, cmd, env in ps)
        if ok:
            state["unhealthy"][g] = 0
            continue
        all_healthy = False
        state["unhealthy"][g] += 1
        if not have_proc or state["unhealthy"][g] >= 3:
            for pid, cmd, env in ps:
                if "vllm serve" in cmd and env.get("CUDA_VISIBLE_DEVICES") == str(g):
                    subprocess.run(["kill", str(pid)])
            time.sleep(5)
            launch_server(g)
            state["unhealthy"][g] = 0
    # 2. sweeps (only while vLLM servers are part of the plan)
    for wid, variants, tkey in ((("pd", PD_VARIANTS, "pd_target"), ("pt", PT_VARIANTS, "pt_target")) if SERVERS else ()):
        target = state[tkey]
        got = {v: c.get(("prompt_sweep_vllm", v), 0) for v in variants}
        running = any("prompt_sweep_vllm.py" in cmd and env.get("SCFX_WORKER_ID") == wid for _, cmd, env in ps)
        if all(n >= target for n in got.values()):
            nxt = [t for t in TARGET_STEPS if t > target]
            if nxt:
                state[tkey] = nxt[0]
                log(f"{wid}: all variants reached {target}; raising target to {nxt[0]}")
                target = nxt[0]
            elif running:
                continue
            else:
                continue  # finished for the night
        if not running and all_healthy and any(n < target for n in got.values()):
            launch(sweep_env(variants, target, wid), ["scripts/prompt_sweep_vllm.py"], f"sweep_{wid}")
    # 2b. control-cell sweep (pb): 10 slots beside pd, 60 alone; relaunched at the wider width once pd exits
    pd_running = any("prompt_sweep_vllm.py" in cmd and env.get("SCFX_WORKER_ID") == "pd" for _, cmd, env in ps)
    pb_procs = [(pid, env) for pid, cmd, env in ps if "prompt_sweep_vllm.py" in cmd and env.get("SCFX_WORKER_ID") == "pb"]
    pb_need = any(c.get(("prompt_sweep_vllm", v), 0) < PB_TARGET for v in PB_VARIANTS)
    want_conc = 10 if pd_running else 60
    if pb_need and all_healthy and SERVERS:
        if pb_procs and not pd_running and pb_procs[0][1].get("SCFX_SWEEP_CONC") == "10":
            for pid, _ in pb_procs:
                subprocess.run(["kill", str(pid)])
            log("pb: pd has exited; relaunching the control sweep at 60 slots")
            time.sleep(5)
            pb_procs = []
        if not pb_procs:
            CONC["pb"] = want_conc
            launch(sweep_env(PB_VARIANTS, PB_TARGET, "pb"), ["scripts/prompt_sweep_vllm.py"], "sweep_pb")
    # 3. HF workers: one per GPU, first unmet cell in priority order
    busy = {}
    for pid, cmd, env in ps:
        if ("prompt_channel.py" in cmd or "run_phase.py" in cmd) and env.get("CUDA_VISIBLE_DEVICES", "").isdigit():
            busy[int(env["CUDA_VISIBLE_DEVICES"])] = env.get("SCFX_WORKER_ID", "?")
    held = reserved()
    for g in HF_GPUS:
        if g in busy or g in held:
            continue
        for name, phase, cond, target, env, args in HF_CELLS:
            have = c.get((phase, cond), 0)
            in_flight = sum(1 for w in busy.values() if w.startswith(name))
            if have + in_flight < target:
                launch({**env, "SCFX_WORKER_ID": f"{name}{g}"}, args, f"hf_{name}{g}", gpu=g)
                busy[g] = f"{name}{g}"
                break
    # 4. judge casualties: two sequential rejudge passes, one per judge quota (Azure deployment and
    #    direct OpenAI, same model), each on its own shard of the unjudged rows
    unjudged = len(c.get("_unjudged", set()))
    running_shards = {env.get("SCFX_REJUDGE_SHARD") for _, cmd, env in ps if "rejudge_phase.py" in cmd}
    rejudging = bool(running_shards)
    if unjudged:
        for shard, extra in REJUDGE_SHARDS:
            if shard not in running_shards:
                launch({"SCFX_REJUDGE_SHARD": shard, **extra}, ["scripts/rejudge_phase.py", *REJUDGE_PHASES], f"rejudge_{shard[0]}")
    status = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "unjudged": unjudged, "rejudging": rejudging, "servers": {g: health(p) for g, p in SERVERS.items()},
        "pd_target": state["pd_target"], "pt_target": state["pt_target"],
        "vllm": {v: c.get(("prompt_sweep_vllm", v), 0) for v in PD_VARIANTS + PT_VARIANTS},
        "pb_target": PB_TARGET,
        "hf": {cond: c.get((phase, cond), 0) for _, phase, cond, _, _, _ in HF_CELLS},
        "hf_busy": {str(g): w for g, w in busy.items()}, "reserved": sorted(held),
        "sweeps_running": sorted({env.get("SCFX_WORKER_ID", "?") for _, cmd, env in ps if "prompt_sweep_vllm.py" in cmd}),
    }
    json.dump(status, open(f"{LOGDIR}/status.json", "w"), indent=1)
    log("status " + json.dumps({k: status[k] for k in ("servers", "pd_target", "vllm", "hf", "hf_busy", "reserved", "sweeps_running", "unjudged")}))


if __name__ == "__main__":
    log("guardian started")
    while True:
        try:
            tick()
        except Exception as e:  # noqa: BLE001
            log(f"tick error: {e!r}")
        time.sleep(300)
