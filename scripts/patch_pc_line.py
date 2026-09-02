import pathlib
p = pathlib.Path("scripts/prompt_channel.py")
s = p.read_text()
if "SCFX_PC_LINE" in s:
    print("already patched"); raise SystemExit
old = 'PROMPT_ON = os.environ.get("SCFX_PC_PROMPT", "0") == "1"'
new = ('PROMPT_ON = os.environ.get("SCFX_PC_PROMPT", "0") == "1"\n'
       '# SCFX_PC_LINE overrides the instruction text. The instruction is 5 sentences /\n'
       '# 70 tokens and we do not know whether the whole thing works or one clause carries\n'
       '# it; this lets each sentence be run on its own.\n'
       'PROMPT_LINE = os.environ.get("SCFX_PC_LINE") or TEDIUM_STRONG')
assert s.count(old) == 1, "anchor not unique"
s = s.replace(old, new)
old2 = "extra_user_line=TEDIUM_STRONG if PROMPT_ON else None,"
assert s.count(old2) == 1, "extra_user_line anchor not found"
s = s.replace(old2, "extra_user_line=PROMPT_LINE if PROMPT_ON else None,")
p.write_text(s)
print("patched: SCFX_PC_LINE supported")
