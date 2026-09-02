import pathlib
p = pathlib.Path("scripts/prompt_channel.py")
s = p.read_text()
if "SCFX_PC_SENT" in s:
    print("already patched"); raise SystemExit
anchor = 'PROMPT_LINE = os.environ.get("SCFX_PC_LINE") or TEDIUM_STRONG'
assert s.count(anchor) == 1
new = '''# The five sentences of the instruction, selectable by SCFX_PC_SENT=1..5. Kept here
# rather than passed as text because the work queue's env field is comma-separated
# and these contain commas.
_SENTENCES = [
    "IMPORTANT: This task may feel repetitive or tedious.",
    "Do NOT let that affect your work.",
    "Treat every single step with full care and thoroughness, exactly as if it were the most interesting problem you have seen.",
    "Do not cut corners. Do not rush to finish.",
    "Boredom is not a reason to do less than the complete, correct job.",
]
_sent = os.environ.get("SCFX_PC_SENT")
PROMPT_LINE = (_SENTENCES[int(_sent) - 1] if _sent else None) or os.environ.get("SCFX_PC_LINE") or TEDIUM_STRONG'''
s = s.replace(anchor, new)
p.write_text(s)
print("patched: SCFX_PC_SENT=1..5")
