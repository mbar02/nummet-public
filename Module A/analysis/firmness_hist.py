# We'll parse the file and generate three histograms:
# 1) "Parity firmness check"  (Δβ/σβ)
# 2) "Ignoring one point check: max"  (Δβ/σβ)
# 3) "Ignoring one point check: mean" (Δβ/σβ)
#
# We'll save figures to /mnt/data and also display them inline.
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO
from matplotlib        import colormaps as cmap


plt.rcParams.update({
    "font.size": 10,
    "font.family": "Times New Roman",
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "figure.dpi": 500,
    "figure.figsize": (4.3,3.5),
    "legend.labelspacing": 0.1,
    "legend.handletextpad": 0.2,
    "legend.borderpad": 0.2,
    "text.usetex": True,
})
colmap = cmap['winter']

src = Path("./firmness-beta-fit.txt").read_text(encoding="utf-8", errors="ignore")

# Patterns
pat_alg_geo_L  = re.compile(r"ALG:\s*([^,]+),\s*GEO:\s*([^,]+),\s*L:\s*(\d+)", re.I)
pat_parity     = re.compile(r"Parity firmness check\s*:.*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
pat_max_ign    = re.compile(r"Ignoring one point check\s*:\s*max\s+.*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.I)
pat_mean_ign   = re.compile(r"Ignoring one point check\s*:\s*mean\s+.*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.I)

rows = []
current = {"ALG": None, "GEO": None, "L": None, "parity": None, "max_ign": None, "mean_ign": None}

for line in src.splitlines():
    line = line.strip()
    if not line:
        continue
    m_head = pat_alg_geo_L.search(line)
    if m_head:
        # if we already have a record accumulating, flush if any values present
        if any(v is not None for k,v in current.items() if k not in ("ALG","GEO","L")) and current["L"] is not None:
            rows.append(current)
        # start a new record
        current = {"ALG": m_head.group(1).strip(), "GEO": m_head.group(2).strip(), "L": int(m_head.group(3)),
                   "parity": None, "max_ign": None, "mean_ign": None}
        continue
    
    m_p = pat_parity.search(line)
    if m_p:
        current["parity"] = float(m_p.group(1))
        continue
    
    m_max = pat_max_ign.search(line)
    if m_max:
        current["max_ign"] = float(m_max.group(1))
        continue
    
    m_mean = pat_mean_ign.search(line)
    if m_mean:
        current["mean_ign"] = float(m_mean.group(1))
        continue

# Append last
if any(v is not None for k,v in current.items() if k not in ("ALG","GEO","L")) and current["L"] is not None:
    rows.append(current)

df = pd.DataFrame(rows).sort_values("L").reset_index(drop=True)

# Helper to plot histogram with vertical lines for mean and max of the series
def make_hist(data, title, filename, xlabel=r"$\Delta\beta/\sigma_\beta$"):
    plt.figure(figsize=(5,3.5), dpi=150)
    plt.hist(data.dropna(), bins=10)
    # Annotate mean and max as vertical lines with labels
    mean_val = float(data.mean())
    plt.axvline(mean_val, linestyle="--", label=f"mean = {mean_val:.3g}")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()
    return filename

out1 = make_hist(df["parity"],  "Parity firmness check: Δβ/σβ",       "firmness_parity_ratio.svg")
out2 = make_hist(df["max_ign"], "Ignoring one point (max): Δβ/σβ",    "firmness_max_ratio.svg")
out3 = make_hist(df["mean_ign"],"Ignoring one point (mean): Δβ/σβ",   "firmness_mean_ratio.svg")

print("Saved figures:")
print(out1, out2, out3)
