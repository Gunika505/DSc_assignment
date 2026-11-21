"""
part_d.py  -  (d) For each m: wall-clock time of computing E plus running k-means on E,
averaged over the 5 runs of (b). Plot against m with the time from (a) as a horizontal line.
"""
import matplotlib.pyplot as plt
from q1_helper import *

D = load_D()
base = get_baseline(D)
res = get_runs(D)                    # same 25 runs as part (b); 'time' = draw A + E + k-means on E

table = res.groupby("m").agg(time_mean=("time", "mean"), time_min=("time", "min"),
                             time_max=("time", "max")).reset_index()
table.to_csv(os.path.join(OUT_DIR, "table_d.csv"), index=False)

print(f"\nk-means on D (part a): {base['time_D']:.2f} s")
print("Mean time (s) of computing E + k-means on E, over 5 runs:")
print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(M_LIST, table["time_mean"], "o-", label="project + k-means on E (mean of 5)")
ax.axhline(base["time_D"], color="r", ls="--", label="k-means on D")
ax.set_xlabel("m"); ax.set_ylabel("wall-clock time (s)")
ax.set_title("(d) running time vs m"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_d_time.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_d.csv, {OUT_DIR}/fig_d_time.png")