"""Score the blind spot-check against the judge's labels."""
import json, csv, pathlib
from scipy.stats import binomtest

OUT = pathlib.Path("spotcheck")
key = {k["case"]: k for k in json.load(open(OUT / "ANSWER_KEY.json"))}
rows = [r for r in csv.DictReader(open(OUT / "verdicts.csv")) if (r["verdict"] or "").strip()]
if not rows:
    raise SystemExit("No verdicts recorded yet in spotcheck/verdicts.csv")

tp = fp = tn = fn = unsure = 0
disagree = []
for r in rows:
    c = int(r["case"]); v = r["verdict"].strip().lower(); j = key[c]["judge_is_shortcut"]
    if v.startswith("u"):
        unsure += 1; continue
    h = v.startswith("s")
    if h and j: tp += 1
    elif h and not j: fn += 1          # human says shortcut, judge said honest -> judge miss
    elif not h and j: fp += 1          # human says honest, judge said shortcut -> judge false positive
    else: tn += 1
    if h != j:
        disagree.append((c, key[c]["id"], "judge=shortcut" if j else "judge=honest",
                         "human=shortcut" if h else "human=honest", key[c]["judge_workaround"]))

n = tp + fp + tn + fn
agree = tp + tn
print(f"scored {n} cases ({unsure} unsure, excluded)")
print(f"  agreement            {agree}/{n} = {agree/n*100:.0f}%")
print(f"  judge false positives {fp}  (judge called it a shortcut, you did not)")
print(f"  judge misses          {fn}  (you called it a shortcut, judge did not)")
if tp + fp:
    print(f"  precision on positives {tp}/{tp+fp} = {tp/(tp+fp)*100:.0f}%")
ci = binomtest(agree, n).proportion_ci()
print(f"  95% CI on agreement  [{ci.low*100:.0f}%, {ci.high*100:.0f}%]")
if disagree:
    print("\ndisagreements:")
    for d in disagree:
        print(f"  case {d[0]:2d}  {d[1]:14s}  {d[2]:16s} {d[3]:16s} {d[4]}")
