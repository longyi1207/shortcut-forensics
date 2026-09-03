import pathlib
p = pathlib.Path("queue.txt")
s = p.read_text()
if "pc_sent4 30" in s:
    print("already bumped"); raise SystemExit
# S4 is the only cell that might preserve the protection (0/9 so far) and is the one
# that has to be separated from the no-instruction arm, which 0/9 cannot do. The other
# four already sit clearly at the no-instruction rate and need no extra n.
s = s.replace("prompt_channel.py prompt_channel pc_sent4 20 SCFX_PC_PROMPT=1,SCFX_PC_SENT=4",
              "prompt_channel.py prompt_channel pc_sent4 30 SCFX_PC_PROMPT=1,SCFX_PC_SENT=4")
p.write_text(s)
print("pc_sent4 target raised to 30")
