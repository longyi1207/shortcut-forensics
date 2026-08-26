# STATUS — 20260821-launch

_Last updated: 2026-08-23T17:53:24Z_

**Phase active:** [5]  
**Phase done:** [0, 1, 2, 3, 4]  
**Blocker:** phase5_failed_3x  

## Clock

- GPU-hours: 4.23
- USD est: $2.87 / $600 cap ($597.13 remaining)
- Credits confirmed: True
- Spot: True, N GPUs: 0
- Rollouts done: 136, shortcuts: 24
- ETA phase complete (UTC): unknown
- Autokill (UTC): 2026-08-23T19:44:14Z

## What is running

phase(s) active: [5]; done: [0, 1, 2, 3, 4]; rollouts_done=136 shortcuts=24

tmux session: `scfx_ly`

## Last error

none

## Next action

BLOCKED: phase5_failed_3x

## Cold-resume command

```bash
tmux new -s scfx_ly 'cd /mnt/scfx_ly_run && python scripts/controller.py --run-id 20260821-launch'
```
