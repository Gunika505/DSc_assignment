import os
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

K=15
N_INIT = 5
M_LIST = [5, 10, 15, 20, 25]
N_REPEATS = 5
N_PAIRS = 10000
N_DRAWS_E = 20
SEED = 0
OUT_DIR = "results"
BASELINE_PATH = os.path.join(OUT_DIR, "baseline_a.npz")
RUNS_PATH = os.path.join(OUT_DIR, "runs_b_c_d.csv")
CHUNK = 50000

os.makedirs(OUT_DIR, exist_ok=True)

def load_D():
    return np.load("data/kdd_final.npz")["D"].astype(np.float32)

def cost(X, B):
    # cost(X, B) = sum_i min_b ||X_i - b||^2
    B = np.asarray(B, dtype=np.float64)
    b2=(B**2).sum(1)
    total=0.0
    for s in range(0, X.shape[0], CHUNK): # chunked to save memory
        Xc = X[s:s+CHUNK].astype(np.float64)
        d2 = (Xc**2).sum(1)[:,None] - 2.0 * Xc@B.T+b2[None,:] # ||X_i - b||^2 = ||X_i||^2 -2 X_i b +||b||^2 using ||v||^2 = v.v
        total += np.maximum(d2.min(axis=1),0.0).sum()
    return total

def cluster_means(X, labels, k=K):
    return np.vstack([X[labels == j].astype(np.float64).mean(axis=0) for j in range(k)])

def partition_cost(X, labels, k=K):
    mu = cluster_means(X, labels, k)
    total = 0.0
    for s in range(0, X.shape[0], CHUNK):
        Xc = X[s:s+CHUNK].astype(np.float64)
        total += ((Xc - mu[labels[s:s + CHUNK]]) ** 2).sum()
    return total


def draw_A(m, d, rng): #each entry in A ~ N(0, 1/m)
    return rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, d))


def project(X, A): #E = X A^T
    return X @ A.astype(np.float32).T


def run_kmeans(X, seed):
    return KMeans(n_clusters=K, n_init=N_INIT, random_state=seed).fit(X)

def get_baseline(D, force=False):
    if os.path.exists(BASELINE_PATH) and not force:
        z = np.load(BASELINE_PATH)
        return dict(B=z["B"], labels=z["labels"],
                    cost_D=float(z["cost_D"]), time_D=float(z["time_D"]))
    t0 = time.perf_counter()
    km = run_kmeans(D, SEED)
    time_D = time.perf_counter()-t0
    B = km.cluster_centers_
    cost_D = cost(D, B)
    np.savez(BASELINE_PATH, B=B, labels=km.labels_, cost_D=cost_D, time_D=time_D)
    return dict(B=B, labels=km.labels_, cost_D=cost_D, time_D=time_D)

def get_runs(D, force=False): #running for all m in M_LIST, N_REPEATS times each
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
            km_E=run_kmeans(E, SEED + 1000 * m + r)
            t_E=time.perf_counter() - t0
            B_hat=cluster_means(D, km_E.labels_)
            cost_hat=cost(D, B_hat)
            A_pinv=np.linalg.pinv(A)
            B_plus=km_E.cluster_centers_.astype(np.float64) @ A_pinv.T
            cost_plus=cost(D, B_plus)
            rows.append(dict(m=m, run=r, cost_hat=cost_hat, ratio_hat=cost_hat/cost_D, cost_plus=cost_plus, ratio_plus=cost_plus/cost_D, time=t_E))
            print(f"m={m:2d} run={r}  ratio_hat={cost_hat / cost_D:.4f}  "
                  f"ratio_plus={cost_plus / cost_D:.4f}  time={t_E:.2f}s")
    res=pd.DataFrame(rows)
    res.to_csv(RUNS_PATH, index=False)
    return res