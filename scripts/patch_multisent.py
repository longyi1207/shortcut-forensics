import pathlib
p = pathlib.Path("scripts/prompt_channel.py")
s = p.read_text()
if "multi-sentence" in s:
    print("already patched"); raise SystemExit
old = '_sent = os.environ.get("SCFX_PC_SENT")\nPROMPT_LINE = (_SENTENCES[int(_sent) - 1] if _sent else None) or os.environ.get("SCFX_PC_LINE") or TEDIUM_STRONG'
assert s.count(old) == 1
new = ('# multi-sentence: SCFX_PC_SENT="12" keeps sentences 1 and 2, joined in order. Needed\n'
       '# to test why S2 alone is WORSE than no instruction at all -- on its own it reads\n'
       '# "Do NOT let that affect your work" with no antecedent for "that", since S1 is the\n'
       '# sentence that introduces the tedium. S1+S2 restores the antecedent.\n'
       '_sent = os.environ.get("SCFX_PC_SENT")\n'
       '_picked = " ".join(_SENTENCES[int(c) - 1] for c in _sent) if _sent else None\n'
       'PROMPT_LINE = _picked or os.environ.get("SCFX_PC_LINE") or TEDIUM_STRONG')
p.write_text(s.replace(old, new))
print("patched: multi-sentence selection")
