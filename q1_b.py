#q1 part (b) - compute cost(D, B_hat) and ratio for 25 runs of each m in {5,10,15,20,25}
import matplotlib.pyplot as plt
from q1_helper import *

D = load_D()
base = get_baseline(D)
res = get_runs(D)

agg_table = res.groupby("m").agg(
    ratio_mean=("ratio_hat", "mean"),
    ratio_min=("ratio_hat", "min"),
    ratio_max=("ratio_hat", "max"),).reset_index()
agg_table.to_csv(os.path.join(OUT_DIR, "agg_table_b.csv"), index=False)

print(f"\ncost(D, B) = {base['cost_D']:.6f}")
print("\nAll runs: cost(D, B_hat) and ratio")
print(res[["m", "run", "cost_hat", "ratio_hat"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nRatio cost(D, B_hat) / cost(D, B) per m:")
print(agg_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# bar graph
x = np.arange(len(M_LIST))
mean = agg_table["ratio_mean"].values
err = np.vstack([mean - agg_table["ratio_min"].values, agg_table["ratio_max"].values - mean])
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x, mean, yerr=err, capsize=4, color="tab:blue")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(M_LIST)
ax.set_xlabel("m"); ax.set_ylabel(r"cost(D, $\hat{B}$) / cost(D, B)")
ax.set_title(r"(b) $\hat{B}$: mean ratio, bars = min/max over 5 runs")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_b_ratio.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/agg_table_b.csv, {OUT_DIR}/fig_b_ratio.png")
print(f"Number of runs with ratio < 1: {(res['ratio_hat'] < 1).sum()} of {len(res)}")