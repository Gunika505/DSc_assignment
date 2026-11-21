"""
jl_common.py
Shared settings and helper functions for part_a.py ... part_e.py.
Not meant to be run directly.

Reads data/kdd_final.npz (made by data_prep.py). Intermediate results are cached in
./results so that the parts can be run separately:
  results/baseline_a.npz   k-means on D  (used by parts b, c, d, e)
  results/runs_b_c_d.csv   the 25 projected runs (shared by parts b, c, d)
Delete these two files if you change any setting below and want a fresh run.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ------------------------------------------------------------------ settings
K = 15
N_INIT = 5
M_LIST = [5, 10, 15, 20, 25]
N_REPEATS = 5          # parts (b)-(d): fresh A per repeat
N_PAIRS = 10_000       # part (e)
N_DRAWS_E = 20         # part (e)(ii)
SEED = 0
OUT_DIR = "results"
CHUNK = 50_000

os.makedirs(OUT_DIR, exist_ok=True)


def load_D():
    return np.load("data/kdd_final.npz")["D"].astype(np.float32)


# ------------------------------------------------------------------ helpers
def cost(X, B):
    """cost(X, B) = sum_i min_b ||X_i - b||^2  (chunked, float64 arithmetic)."""
    B = np.asarray(B, dtype=np.float64)
    b2 = (B ** 2).sum(1)
    total = 0.0
    for s in range(0, X.shape[0], CHUNK):
        Xc = X[s:s + CHUNK].astype(np.float64)
        d2 = (Xc ** 2).sum(1)[:, None] - 2.0 * Xc @ B.T + b2[None, :]
        total += np.maximum(d2.min(axis=1), 0.0).sum()
    return total


def cluster_means(X, labels, k=K):
    """mu_j(X) for every cluster j (float64)."""
    return np.vstack([X[labels == j].astype(np.float64).mean(axis=0) for j in range(k)])


def partition_cost(X, labels, k=K):
    """cost_X(C) = sum_j sum_{i in C_j} ||X_i - mu_j(X)||^2."""
    mu = cluster_means(X, labels, k)
    total = 0.0
    for s in range(0, X.shape[0], CHUNK):
        Xc = X[s:s + CHUNK].astype(np.float64)
        total += ((Xc - mu[labels[s:s + CHUNK]]) ** 2).sum()
    return total


def draw_A(m, d, rng):
    """A in R^{m x d}, i.i.d. N(0, 1/m)  (std = 1/sqrt(m))."""
    return rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, d))


def project(X, A):
    """E = X A^T (float32 to save memory)."""
    return X @ A.astype(np.float32).T


def run_kmeans(X, seed):
    return KMeans(n_clusters=K, n_init=N_INIT, random_state=seed).fit(X)


# ------------------------------------------------------------------ part (a) result
BASELINE_PATH = os.path.join(OUT_DIR, "baseline_a.npz")


def get_baseline(D, force=False):
    """k-means on D. Returns dict(B, labels, cost_D, time_D); cached on disk."""
    if os.path.exists(BASELINE_PATH) and not force:
        z = np.load(BASELINE_PATH)
        return dict(B=z["B"], labels=z["labels"],
                    cost_D=float(z["cost_D"]), time_D=float(z["time_D"]))
    t0 = time.perf_counter()
    km = run_kmeans(D, SEED)
    time_D = time.perf_counter() - t0
    B = km.cluster_centers_
    cost_D = cost(D, B)
    np.savez(BASELINE_PATH, B=B, labels=km.labels_, cost_D=cost_D, time_D=time_D)
    return dict(B=B, labels=km.labels_, cost_D=cost_D, time_D=time_D)


# ------------------------------------------------------------------ parts (b)-(d) runs
RUNS_PATH = os.path.join(OUT_DIR, "runs_b_c_d.csv")


def get_runs(D, force=False):
    """The 5 x len(M_LIST) projected runs, shared by parts (b), (c), (d); cached on disk."""
    if os.path.exists(RUNS_PATH) and not force:
        return pd.read_csv(RUNS_PATH)
    n, d = D.shape
    cost_D = get_baseline(D)["cost_D"]
    rows = []
    for m in M_LIST:
        for r in range(N_REPEATS):
            rng = np.random.default_rng([SEED, m, r])
            t0 = time.perf_counter()
            A = draw_A(m, d, rng)
            E = project(D, A)
            km_E = run_kmeans(E, SEED + 1000 * m + r)
            t_E = time.perf_counter() - t0        # time for (d): A, E and k-means on E

            # (b) centres in R^d: average the original rows of D in each cluster of E
            B_hat = cluster_means(D, km_E.labels_)
            cost_hat = cost(D, B_hat)

            # (c) centres in R^d via pseudo-inverse: A^+ a
            A_pinv = np.linalg.pinv(A)            # d x m
            B_plus = km_E.cluster_centers_.astype(np.float64) @ A_pinv.T   # k x d
            cost_plus = cost(D, B_plus)

            rows.append(dict(m=m, run=r, cost_hat=cost_hat, ratio_hat=cost_hat / cost_D,
                             cost_plus=cost_plus, ratio_plus=cost_plus / cost_D, time=t_E))
            print(f"m={m:2d} run={r}  ratio_hat={cost_hat / cost_D:.4f}  "
                  f"ratio_plus={cost_plus / cost_D:.4f}  time={t_E:.2f}s")
    res = pd.DataFrame(rows)
    res.to_csv(RUNS_PATH, index=False)
    return res