"""Build a self-contained HTML to visually compare SLT operators (ot_adain vs W2
vs SW, standalone and hybrid). Thumbnails base64-embedded so it opens anywhere.
Rows = styles (prompt kp0), cols = operators. Reads results/t2i_sltopt.csv for
per-image CSD/CLIP labels + mean summary."""
import csv, base64, io, os, collections, statistics as st
from PIL import Image

CWD = "/home/ubuntu/chengyanli/items/save/attention_20260426_090948"
IMG = f"{CWD}/d_grand/t2i_sltopt"
STY = f"{CWD}/sty"
OUT = f"{CWD}/d_grand/sltopt_compare.html"
CSV = f"{CWD}/results/t2i_sltopt.csv"

STYLES = ["00", "10", "14", "20", "27", "28", "33", "38"]
PID = "kp0"   # "A panda eat bamboo"

# (column label, method key in csv/filenames, None for style-ref)
COLS = [
    ("Style ref",      None),
    ("ot_adain α0.1",  "slt_otadain"),
    ("W2 α0.1",        "slt_w2"),
    ("W2 α0.4 ★", "slt_w2_a04"),
    ("SW α0.1",        "slt_sw"),
    ("Hybrid ot_adain","hyb_otadain"),
    ("Hybrid W2",      "hyb_w2"),
]

# per-(method,sty,pid) metrics
m = {}
means = collections.defaultdict(lambda: [[], []])
for r in csv.DictReader(open(CSV)):
    m[(r["method"], r["sty_id"], r["content_id"])] = (float(r["csd"]), float(r["clip_t"]))
    means[r["method"]][0].append(float(r["csd"]))
    means[r["method"]][1].append(float(r["clip_t"]))

def thumb_b64(path, size=200):
    try:
        im = Image.open(path).convert("RGB"); im.thumbnail((size, size))
        b = io.BytesIO(); im.save(b, "JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    except Exception:
        return ""

def cell(label, method, sty):
    if method is None:
        src = thumb_b64(f"{STY}/{sty}.png")
        return f'<td><img src="{src}"><div class="lab">style</div></td>'
    src = thumb_b64(f"{IMG}/{method}_{PID}_s{sty}.png")
    csd, clipt = m.get((method, sty, PID), (float("nan"), float("nan")))
    return (f'<td><img src="{src}">'
            f'<div class="lab">CSD <b>{csd:.2f}</b> · T {clipt:.2f}</div></td>')

# summary table
summ_order = ["slt_otadain", "slt_blend", "slt_w2", "slt_w2_a04", "slt_sw", "slt_sw_a04",
              "hyb_otadain", "hyb_w2", "hyb_sw"]
base_csd = st.mean(means["slt_otadain"][0])
srows = ""
for k in summ_order:
    if k not in means: continue
    c = st.mean(means[k][0]); t = st.mean(means[k][1])
    delta = (c - base_csd) / base_csd * 100
    hi = ' class="win"' if k in ("slt_w2_a04", "hyb_w2") else ""
    srows += (f"<tr{hi}><td>{k}</td><td>{c:.4f}</td><td>{t:.4f}</td>"
              f"<td>{delta:+.0f}%</td></tr>")

rows_html = ""
head = "".join(f"<th>{lab}</th>" for lab, _ in COLS)
for sty in STYLES:
    cells = "".join(cell(lab, meth, sty) for lab, meth in COLS)
    rows_html += f"<tr><th class='sty'>sty{sty}</th>{cells}</tr>"

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SLT operator comparison</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#fafafa;color:#222}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:28px}}
table{{border-collapse:collapse}} td,th{{padding:4px;text-align:center;font-size:12px}}
img{{width:150px;height:150px;object-fit:cover;border-radius:4px;display:block}}
.lab{{font-size:11px;color:#555;margin-top:2px}} .sty{{writing-mode:vertical-rl;font-size:12px;color:#888}}
.grid th{{font-size:12px;color:#333;border-bottom:1px solid #ddd}}
.sum td,.sum th{{border:1px solid #ddd;padding:6px 12px;font-size:13px}}
.sum tr.win{{background:#fde8e8;font-weight:600}} .sum th{{background:#f0f0f0}}
.note{{color:#666;font-size:13px;max-width:880px;line-height:1.5}}
</style></head><body>
<h1>SLT operator comparison — ot_adain vs W2 vs SW</h1>
<p class="note">8 styles &times; prompt "A panda eat bamboo". <b>W2</b> = closed-form full-covariance Gaussian-W2 (Monge) map;
<b>ot_adain</b> = current mean+per-channel-std op; <b>SW</b> = sliced-Wasserstein. Standalone columns are pure
colour pre-align (no Adam); Hybrid columns add Adam-2. ★ marks the standout.</p>
<h2>Mean over 8 styles &times; 5 prompts (n=40)</h2>
<table class="sum"><tr><th>config</th><th>CSD</th><th>CLIP-T</th><th>&Delta;CSD vs ot_adain</th></tr>{srows}</table>
<h2>Per-style images</h2>
<table class="grid"><tr><th></th>{head}</tr>{rows_html}</table>
</body></html>"""

with open(OUT, "w") as f:
    f.write(html)
print("saved:", OUT, f"({os.path.getsize(OUT)//1024} KB)")
