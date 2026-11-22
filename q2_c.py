import matplotlib.pyplot as plt
from q2_helper import *
"""
lr_c.py  for c Time to draw S, forming SD and Sy and computing a bar, averaged over the 5 runs we got from b and then finally compared
with the time to compute b in part a
"""

D, target=load_data()
baseline=get_baseline(D, target)
res=get_runs(D, target)                     # same 35 runs as part (b)

req_table=res.groupby("m").agg(time_mean=("time", "mean"), time_min=("time", "min"),
                                 time_max=("time", "max")).reset_index()
req_table["speedup_vs_full"] = baseline["time_b"]/req_table["time_mean"]
req_table.to_csv(os.path.join(OUT_DIR, "table_c.csv"), index=False)

print(f"\nTime to compute b (full lstsq): {baseline['time_b']:.3f} s")
print("Mean time of sketch + lstsq on the sketch, over 5 runs:")
print(req_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

fig, ax=plt.subplots(figsize=(6.5, 4.5))
ax.plot(req_table["m"], req_table["time_mean"], "o-", label="CountSketch + lstsq (mean of 5)")
ax.axhline(baseline["time_b"], color="r", ls="--", label="full lstsq (a)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("m"); ax.set_ylabel("wall-clock time (s)")
ax.set_title("(c) running time vs m"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_c_time.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_c.csv, fig_c_time.png")
