"""Build a single HTML file that pastes into Google Docs with formatting and
figures intact: open in a browser, select all, copy, paste. Images are inlined as
base64 so the paste carries them."""
import base64, html, pathlib, re

import sys
# v2 build: python to_gdoc_v2.py MATS_SUBMISSION_v2.md MATS_SUBMISSION_v2.html
SRC, DST = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("MATS_SUBMISSION_v2.md", "MATS_SUBMISSION_v2.html")
FIGS = {"FIG 1": "figs/prompt_vs_steer.png", "FIG 2": "figs/geometry.png", "FIG 3": "figs/channels.png"}
md = pathlib.Path(SRC).read_text()


def img(name):
    b = base64.b64encode(pathlib.Path(FIGS[name]).read_bytes()).decode()
    return f'<p style="margin:14pt 0"><img src="data:image/png;base64,{b}" style="width:100%;max-width:660px"></p>'


def inline(t):
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r'<span style="font-family:Consolas,monospace;font-size:9.5pt">\1</span>', t)
    return t


out, lines, i = [], md.split("\n"), 0
while i < len(lines):
    ln = lines[i]
    m = re.match(r"^> \*\*\[(FIG \d): [^\]]+\]\*\*", ln)
    if m:
        out.append(img(m.group(1))); i += 1; continue
    if ln.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
        hdr = [c.strip() for c in ln.strip("|").split("|")]
        i += 2
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append([c.strip() for c in lines[i].strip("|").split("|")]); i += 1
        th = "".join(f'<th style="border:1px solid #ccc;padding:5px 8px;background:#f4f3f0;text-align:left">{inline(c)}</th>' for c in hdr)
        tb = "".join("<tr>" + "".join(f'<td style="border:1px solid #ccc;padding:5px 8px">{inline(c)}</td>' for c in r) + "</tr>" for r in rows)
        out.append(f'<table style="border-collapse:collapse;font-size:10pt;margin:10pt 0">{("<tr>" + th + "</tr>")}{tb}</table>')
        continue
    if ln.startswith("# "):
        out.append(f'<h1 style="font-size:17pt;margin:18pt 0 6pt">{inline(ln[2:])}</h1>')
    elif ln.startswith("## "):
        out.append(f'<h2 style="font-size:13pt;margin:16pt 0 5pt">{inline(ln[3:])}</h2>')
    elif ln.strip() == "---":
        out.append('<hr style="border:none;border-top:1px solid #ddd;margin:16pt 0">')
    elif ln.startswith("> "):
        out.append(f'<p style="margin:8pt 0;color:#444">{inline(ln[2:])}</p>')
    elif ln.strip():
        buf = [ln]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||>|---)", lines[i + 1]):
            i += 1; buf.append(lines[i])
        out.append(f'<p style="margin:7pt 0;line-height:1.45">{inline(" ".join(x.strip() for x in buf))}</p>')
    i += 1

pathlib.Path(DST).write_text(
    '<div style="font-family:Georgia,serif;font-size:11pt;color:#111;max-width:680px">'
    + "\n".join(out) + "</div>")
print("wrote", DST)
