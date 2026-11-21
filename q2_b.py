"""
lr_b.py  -  (b) CountSketch regression for m = 50,...,5000, 5 fresh sketches per m.
Reports Loss(D,y,a~), ratio to Loss(D,y,b) and the sketched loss ||S D a~ - S y||^2.
Log-log plot of the mean ratio with min/max error bars.
"""
import matplotlib.pyplot as plt
from q2_helper import *

D, y = load_data()
base = get_baseline(D, y)
res = get_runs(D, y)

table = res.groupby("m").agg(
    ratio_mean=("ratio", "mean"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max"),
    loss_mean=("loss", "mean"), sketched_mean=("sketched_loss", "mean"),
    sketched_min=("sketched_loss", "min"), sketched_max=("sketched_loss", "max"),
).reset_index()
table.to_csv(os.path.join(OUT_DIR, "table_b.csv"), index=False)

print(f"\nLoss(D, y, b) = {base['loss_b']:.6f},  rank(D) = {base['rank']}")
print("\nAll runs:")
print(res[["m", "run", "loss", "ratio", "sketched_loss"]].to_string(index=False,
      float_format=lambda v: f"{v:.5g}"))
print("\nPer m:")
print(table.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

# log-log plot: mean ratio, error bars = smallest / largest of the 5 ratios
mean = table["ratio_mean"].values
err = np.vstack([mean - table["ratio_min"].values, table["ratio_max"].values - mean])
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(table["m"], mean, yerr=err, fmt="o-", capsize=4)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.axvline(base["rank"], color="r", ls=":", label=f"rank(D) = {base['rank']}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("m"); ax.set_ylabel(r"Loss(D, y, $\tilde a$) / Loss(D, y, b)")
ax.set_title("(b) CountSketch regression: mean ratio, bars = min/max of 5")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_b_ratio.png"), dpi=150); plt.close(fig)

# sketched loss vs m (helps to describe the behaviour below / above rank(D))
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(table["m"], table["sketched_mean"], "o-", label=r"mean sketched loss $\|SD\tilde a - Sy\|^2$")
ax.axhline(base["loss_b"], color="k", ls="--", lw=1, label="Loss(D, y, b)")
ax.axvline(base["rank"], color="r", ls=":", label=f"rank(D) = {base['rank']}")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-6)
ax.set_xlabel("m"); ax.set_ylabel("sketched loss")
ax.set_title("(b) sketched loss vs m"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_b_sketched_loss.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_b.csv, fig_b_ratio.png, fig_b_sketched_loss.png")