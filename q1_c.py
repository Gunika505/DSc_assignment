"""
part_c.py  -  (c) In the same runs as (b), map each centre a found on E back to R^d as A^+ a
(numpy.linalg.pinv) to get B^+. Report cost(D, B^+) / cost(D, B) next to the B_hat ratio from (b).
"""
import matplotlib.pyplot as plt
from q1_helper import *

D = load_D()
res = get_runs(D)                    # same 25 runs as part (b)

table = res.groupby("m").agg(
    hat_mean=("ratio_hat", "mean"), hat_min=("ratio_hat", "min"), hat_max=("ratio_hat", "max"),
    plus_mean=("ratio_plus", "mean"), plus_min=("ratio_plus", "min"), plus_max=("ratio_plus", "max"),
).reset_index()
table.to_csv(os.path.join(OUT_DIR, "table_c.csv"), index=False)

print("\nAll runs: ratio_hat (B_hat) next to ratio_plus (B^+)")
print(res[["m", "run", "ratio_hat", "ratio_plus"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nPer m (mean / min / max over 5 runs):")
print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# grouped bar graph (log scale because B^+ can be far worse than B_hat)
x = np.arange(len(M_LIST))
w = 0.38
fig, ax = plt.subplots(figsize=(7, 4))
for off, key, lab, col in [(-w / 2, "hat", r"$\hat{B}$", "tab:blue"),
                           (w / 2, "plus", r"$B^{+}$", "tab:orange")]:
    mean = table[f"{key}_mean"].values
    err = np.vstack([mean - table[f"{key}_min"].values, table[f"{key}_max"].values - mean])
    ax.bar(x + off, mean, w, yerr=err, capsize=3, label=lab, color=col)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(M_LIST)
ax.set_xlabel("m"); ax.set_ylabel("cost ratio vs. cost(D, B)")
ax.set_yscale("log")
ax.set_title(r"(c) $\hat{B}$ vs $B^{+}$ (mean, min/max over 5 runs)")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_c_ratio.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_c.csv, {OUT_DIR}/fig_c_ratio.png")