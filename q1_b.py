"""
part_b.py  -  (b) For m in {5,10,15,20,25}: draw A, E = D A^T, k-means on E, form B_hat
(average the rows of D in each cluster). Ratio cost(D, B_hat) / cost(D, B), 5 repeats per m.
Table of mean / min / max ratio + bar graph.
"""
import matplotlib.pyplot as plt
from q1_helper import *

D = load_D()
base = get_baseline(D)
res = get_runs(D)                    # computes the 25 runs once, then reuses them

table = res.groupby("m").agg(
    ratio_mean=("ratio_hat", "mean"),
    ratio_min=("ratio_hat", "min"),
    ratio_max=("ratio_hat", "max"),
).reset_index()
table.to_csv(os.path.join(OUT_DIR, "table_b.csv"), index=False)

print(f"\ncost(D, B) = {base['cost_D']:.6f}")
print("\nAll runs: cost(D, B_hat) and ratio")
print(res[["m", "run", "cost_hat", "ratio_hat"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nRatio cost(D, B_hat) / cost(D, B) per m:")
print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# bar graph: mean with min-max whiskers
x = np.arange(len(M_LIST))
mean = table["ratio_mean"].values
err = np.vstack([mean - table["ratio_min"].values, table["ratio_max"].values - mean])
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x, mean, yerr=err, capsize=4, color="tab:blue")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(M_LIST)
ax.set_xlabel("m"); ax.set_ylabel(r"cost(D, $\hat{B}$) / cost(D, B)")
ax.set_title(r"(b) $\hat{B}$: mean ratio, bars = min/max over 5 runs")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_b_ratio.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_b.csv, {OUT_DIR}/fig_b_ratio.png")
print(f"Number of runs with ratio < 1: {(res['ratio_hat'] < 1).sum()} of {len(res)}")