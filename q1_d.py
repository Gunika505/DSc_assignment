#q1 part (d) - running time vs m
import matplotlib.pyplot as plt
from q1_helper import *

D = load_D()
base = get_baseline(D)
res = get_runs(D)

table = res.groupby("m").agg(time_mean=("time", "mean"), time_min=("time", "min"),
                             time_max=("time", "max")).reset_index()
table.to_csv(os.path.join(OUT_DIR, "table_d.csv"), index=False)
print(f"\nk-means on D (part a): {base['time_D']:.2f} s")
print("Mean time (s) of computing E + k-means on E, over 5 runs:")
print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# line plot of mean time vs m, with min/max error bars
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(M_LIST, table["time_mean"], "o-", label="project + k-means on E (mean of 5)")
ax.axhline(base["time_D"], color="r", ls="--", label="k-means on D")
ax.set_xlabel("m"); ax.set_ylabel("wall-clock time (s)")
ax.set_title("(d) running time vs m"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_d_time.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_d.csv, {OUT_DIR}/fig_d_time.png")