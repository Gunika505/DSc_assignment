"""
part_e.py  -  (e) For each m:
  (i)  one draw of A: ||E_i - E_l||^2 / ||D_i - D_l||^2 on 10^4 random pairs with D_i != D_l
       (same pairs for every m)
  (ii) 20 independent draws of A: cost_E(C_D) / cost_D(C_D) for the clusters C_D from part (a)
Box plots of (i) and (ii) against m in one figure.
"""
import matplotlib.pyplot as plt
from q1_helper import *

D = load_D()
n, d = D.shape
base = get_baseline(D)
labels_D = base["labels"]

# ---- 10^4 random pairs with D_i != D_l, fixed for all m ----
rng_pairs = np.random.default_rng([SEED, 999])
cand = rng_pairs.integers(0, n, size=(8 * N_PAIRS, 2))
diff = D[cand[:, 0]].astype(np.float64) - D[cand[:, 1]].astype(np.float64)
dist2_D = (diff ** 2).sum(1)
ok = (cand[:, 0] != cand[:, 1]) & (dist2_D > 0)      # many KDD rows are duplicates
assert ok.sum() >= N_PAIRS, "not enough distinct pairs; increase the oversampling factor"
pair_diff = diff[ok][:N_PAIRS].astype(np.float32)    # D_i - D_l
pair_d2 = dist2_D[ok][:N_PAIRS]

costD_CD = partition_cost(D, labels_D)               # cost_D(C_D)

pair_ratios, cost_ratios = {}, {}
for m in M_LIST:
    # (i) one draw of A
    A = draw_A(m, d, np.random.default_rng([SEED, m, 1000]))
    diffE = pair_diff @ A.astype(np.float32).T       # = E_i - E_l
    pair_ratios[m] = (diffE.astype(np.float64) ** 2).sum(1) / pair_d2

    # (ii) 20 independent draws
    vals = []
    for t in range(N_DRAWS_E):
        A = draw_A(m, d, np.random.default_rng([SEED, m, 2000 + t]))
        E = project(D, A)
        vals.append(partition_cost(E, labels_D) / costD_CD)
    cost_ratios[m] = np.array(vals)
    print(f"m={m:2d}  (i) pairs: mean={pair_ratios[m].mean():.3f} sd={pair_ratios[m].std():.3f} | "
          f"(ii) cost: mean={cost_ratios[m].mean():.3f} sd={cost_ratios[m].std():.3f}")

spread = pd.DataFrame({
    "m": M_LIST,
    "pair_mean": [pair_ratios[m].mean() for m in M_LIST],
    "pair_std": [pair_ratios[m].std() for m in M_LIST],
    "pair_IQR": [np.subtract(*np.percentile(pair_ratios[m], [75, 25])) for m in M_LIST],
    "cost_mean": [cost_ratios[m].mean() for m in M_LIST],
    "cost_std": [cost_ratios[m].std() for m in M_LIST],
    "cost_IQR": [np.subtract(*np.percentile(cost_ratios[m], [75, 25])) for m in M_LIST],
})
spread.to_csv(os.path.join(OUT_DIR, "table_e.csv"), index=False)
print("\n" + spread.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---- one figure, two box plots per m ----
fig, ax = plt.subplots(figsize=(9, 5))
pos = np.arange(len(M_LIST)) * 3.0
b1 = ax.boxplot([pair_ratios[m] for m in M_LIST], positions=pos - 0.5, widths=0.8,
                patch_artist=True, showfliers=False)
b2 = ax.boxplot([cost_ratios[m] for m in M_LIST], positions=pos + 0.5, widths=0.8,
                patch_artist=True, showfliers=True)
for patch in b1["boxes"]: patch.set_facecolor("tab:blue")
for patch in b2["boxes"]: patch.set_facecolor("tab:orange")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(pos); ax.set_xticklabels(M_LIST)
ax.set_xlabel("m"); ax.set_ylabel("ratio")
ax.set_title("(e) pairwise distance ratio (i) vs. clustering cost ratio (ii)")
ax.legend([b1["boxes"][0], b2["boxes"][0]],
          [r"(i) $\|E_i-E_l\|^2/\|D_i-D_l\|^2$, 10^4 pairs, one A",
           r"(ii) cost$_E(C_D)$/cost$_D(C_D)$, 20 draws of A"])
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_e_boxplots.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/table_e.csv, {OUT_DIR}/fig_e_boxplots.png")