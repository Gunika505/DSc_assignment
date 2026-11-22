import matplotlib.pyplot as plt
from q2_helper import *
import random
import numpy as np
random.seed(42)
np.random.seed(42)

"""
lr_d.py  for d residual of the full solution v = Db-y. For m in set 50 500 5000 we draw 100
independent countsketches and compute ||S v||^2 / ||v||^2 additionally histogram, mean and std for each m.
"""

M_D=[50, 500, 5000]
N_DRAWS=100
D, target=load_data()
n=D.shape[0]
baseline=get_baseline(D, target)
residual=D @ baseline["b"] - target
residual_sq=float(residual @ residual)

ratios={}
for m in M_D:
    vals=[]
    for t in range(N_DRAWS):
        rng=np.random.default_rng([SEED, m, 5000 + t])
        h, sigma=cs_params(n, m, rng)
        S_residual=cs_apply(residual, h, sigma, m)
        vals.append(float(S_residual @ S_residual) / residual_sq)
    ratios[m]=np.array(vals)

stats=pd.DataFrame({"m":M_D,
                    "mean":[ratios[m].mean() for m in M_D],
                    "std":[ratios[m].std(ddof=1) for m in M_D],
                    "min":[ratios[m].min() for m in M_D],
                    "max":[ratios[m].max() for m in M_D]})
stats["std*sqrt(m)"]=stats["std"] * np.sqrt(stats["m"])
stats.to_csv(os.path.join(OUT_DIR, "table_d.csv"), index=False)
print(f"||residual||^2={residual_sq:.6f}")
print(stats.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

fig, axes=plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
for ax, m in zip(axes, M_D):
    ax.hist(ratios[m], bins=20, color="tab:blue", edgecolor="white")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_title(f"m={m}: mean {ratios[m].mean():.3f}, sd {ratios[m].std(ddof=1):.3f}")
    ax.set_xlabel(r"$\|S v\|^2 / \|v\|^2$"); ax.set_ylabel("count")
fig.suptitle("(d) CountSketch of the residual, 100 draws")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_d_hist.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_d.csv, fig_d_hist.png")
