"""
lr_c.py  -  (c) Time to draw S, form SD and Sy and compute a~, averaged over the 5 runs of (b),
compared with the time to compute b in (a).
"""
import matplotlib.pyplot as plt
from q2_helper import *

D, y = load_data()
base = get_baseline(D, y)
res = get_runs(D, y)                     # same 35 runs as part (b)

table = res.groupby("m").agg(time_mean=("time", "mean"), time_min=("time", "min"),
                             time_max=("time", "max")).reset_index()
table["speedup_vs_full"] = base["time_b"] / table["time_mean"]
table.to_csv(os.path.join(OUT_DIR, "table_c.csv"), index=False)

print(f"\nTime to compute b (full lstsq): {base['time_b']:.3f} s")
print("Mean time of sketch + lstsq on the sketch, over 5 runs:")
print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(table["m"], table["time_mean"], "o-", label="CountSketch + lstsq (mean of 5)")
ax.axhline(base["time_b"], color="r", ls="--", label="full lstsq (a)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("m"); ax.set_ylabel("wall-clock time (s)")
ax.set_title("(c) running time vs m"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_c_time.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_c.csv, fig_c_time.png")