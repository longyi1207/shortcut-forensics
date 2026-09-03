import pathlib
p = pathlib.Path("scripts/status_line.sh")
s = p.read_text()
if "sent1" in s:
    print("already patched"); raise SystemExit
old = 'echo "ACTIVE '
new = ('echo "SENT $(c prompt_channel pc_sent1)/$(c prompt_channel pc_sent2)/$(c prompt_channel pc_sent3)/'
       '$(c prompt_channel pc_sent4)/$(c prompt_channel pc_sent5) of 20 each | rand26=$(c prompt_channel pc_prompt_add_rand26)/20 | ACTIVE ')
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print("status line tracks the sentence cells")
