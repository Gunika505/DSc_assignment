"""
lr_d.py  -  (d) Residual of the full solution v = D b - y. For m in {50, 500, 5000}, draw 100
independent CountSketches and compute ||S v||^2 / ||v||^2. Histogram, mean and std for each m.
"""
import matplotlib.pyplot as plt
from q2_helper import *

M_D = [50, 500, 5000]
N_DRAWS = 100

D, y = load_data()
n = D.shape[0]
base = get_baseline(D, y)
v = D @ base["b"] - y
v2 = float(v @ v)

ratios = {}
for m in M_D:
    vals = []
    for t in range(N_DRAWS):
        rng = np.random.default_rng([SEED, m, 5000 + t])
        h, sigma = cs_params(n, m, rng)
        Sv = cs_apply(v, h, sigma, m)
        vals.append(float(Sv @ Sv) / v2)
    ratios[m] = np.array(vals)

stats = pd.DataFrame({"m": M_D,
                      "mean": [ratios[m].mean() for m in M_D],
                      "std": [ratios[m].std(ddof=1) for m in M_D],
                      "min": [ratios[m].min() for m in M_D],
                      "max": [ratios[m].max() for m in M_D]})
stats["std*sqrt(m)"] = stats["std"] * np.sqrt(stats["m"])
stats.to_csv(os.path.join(OUT_DIR, "table_d.csv"), index=False)
print(f"||v||^2 = {v2:.6f}")
print(stats.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
for ax, m in zip(axes, M_D):
    ax.hist(ratios[m], bins=20, color="tab:blue", edgecolor="white")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_title(f"m = {m}: mean {ratios[m].mean():.3f}, sd {ratios[m].std(ddof=1):.3f}")
    ax.set_xlabel(r"$\|Sv\|^2 / \|v\|^2$"); ax.set_ylabel("count")
fig.suptitle("(d) CountSketch of the residual, 100 draws")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_d_hist.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_d.csv, fig_d_hist.png")