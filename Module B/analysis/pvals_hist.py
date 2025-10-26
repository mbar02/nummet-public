# Re-run the parsing and plotting in a fresh session.
import re
from pathlib import Path
import matplotlib.pyplot as plt
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

txt_path = Path("./nonstdupdates-pvals.txt")
text = txt_path.read_text(encoding="utf-8", errors="ignore")

num_pat = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
re_wolff = re.compile(r"metropolis.*?vs\s+alg\s+wolff.*?:\s*" + num_pat, re.IGNORECASE)
re_multi = re.compile(r"metropolis.*?vs\s+alg\s+multicluster.*?:\s*" + num_pat, re.IGNORECASE)

p_wolff = [float(m.group(1)) for m in re_wolff.finditer(text)]
p_multi = [float(m.group(1)) for m in re_multi.finditer(text)]

out1 = Path(".//hist_pvals_metropolis_vs_wolff.svg")

plt.figure(figsize=(5,3.5), dpi=150)
plt.hist(p_wolff, bins=50, range=(0,1), density=True)
plt.plot
plt.xlabel("p-value")
plt.ylabel("Frequency")
# plt.title("Metropolis vs Wolff — p-values")
plt.tight_layout()
plt.savefig(out1)
# plt.show()

out2 = Path("./hist_pvals_metropolis_vs_multicluster.svg")
plt.figure(figsize=(5,3.5), dpi=150)
plt.hist(p_multi, bins=50, range=(0,1),density=True)
plt.xlabel("p-value")
plt.ylabel("Frequency")
# plt.title("Metropolis vs Multicluster — p-values")
plt.tight_layout()
plt.savefig(out2)
# plt.show()

print(f"Found {len(p_wolff)} 'metropolis vs wolff' p-values and {len(p_multi)} 'metropolis vs multicluster' p-values.")
print("Saved figures:", out1, out2)