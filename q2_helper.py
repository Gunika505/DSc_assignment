import os
import time
import numpy as np
import pandas as pd
"""
lr_common.py
here are shared settings and helpers for all parts of question 2 (JL / sketching for linear regression)
it reads data/kdd_final.npz made by data_prep.py (D and the label strings y)
caches results in ./results_lr:
  baseline_a.npz   full least-squares solution b  (used by b, c, d, e)
  runs_b_c.csv     the 5 x 7 sketched runs         (shared by b and c)
"""

SEED=42
OUT_DIR="results_lr"
M_LIST=[50, 100, 200, 500, 1000, 2000, 5000]
N_REPEATS=5
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    """D (float64, Fortran order so columns are contiguous) and y in {0,1}^n."""
    data_file=np.load("data/kdd_final.npz")
    D=np.asfortranarray(data_file["D"], dtype=np.float64)
    #1 is an attack and 0 is normal
    target=(data_file["y"] != "normal.").astype(np.float64)   
    return D, target


#CountSketch from here
def cs_params(n, m, rng):
    """Bucket h(i) uniform in {0..m-1}, sign sigma(i) uniform in {-1,+1}, all independent."""
    h=rng.integers(0, m, size=n)
    sigma=(2 * rng.integers(0, 2, size=n) - 1).astype(np.float64)
    return h, sigma


def cs_apply(X, h, sigma, m):
    """Compute S X without ever forming S: row i of X is added, with sign sigma(i), to row h(i).
    Same result as np.add.at(SX, h, sigma[:, None] * X), but np.bincount is much faster."""
    if X.ndim == 1:
        return np.bincount(h, weights=sigma * X, minlength=m)
    out=np.empty((m, X.shape[1]))
    for j in range(X.shape[1]):
        out[:, j]=np.bincount(h, weights=sigma * X[:, j], minlength=m)
    return out


def countsketch(D, y, m, rng):
    """Draw a CountSketch S in R^{m x n} (implicitly) and return SD, Sy."""
    h, sigma=cs_params(D.shape[0], m, rng)
    return cs_apply(D, h, sigma, m), cs_apply(y, h, sigma, m)


def loss(D, y, a):
    """Loss(D, y, a) = ||D a - y||_2^2."""
    residual=D @ a - y
    return float(residual @ residual)


def lstsq(A, b):
    """Least squares via numpy.linalg.lstsq (min-norm solution if A is rank deficient)."""
    return np.linalg.lstsq(A, b, rcond=None)[0]


#A result
BASELINE_PATH=os.path.join(OUT_DIR, "baseline_a.npz")


def get_baseline(D, y, force=False):
    """Full least squares b (timed), its loss, and rank(D). Cached on disk."""
    if os.path.exists(BASELINE_PATH) and not force:
        data_file=np.load(BASELINE_PATH)
        return dict(b=data_file["b"], loss_b=float(data_file["loss_b"]), time_b=float(data_file["time_b"]),
                    rank=int(data_file["rank"]))
    t0=time.perf_counter()
    b=lstsq(D, y)
    time_b=time.perf_counter() - t0
    loss_b=loss(D, y, b)
    rank=int(np.linalg.matrix_rank(D))
    np.savez(BASELINE_PATH, b=b, loss_b=loss_b, time_b=time_b, rank=rank)
    return dict(b=b, loss_b=loss_b, time_b=time_b, rank=rank)


#B and C runs
RUNS_PATH=os.path.join(OUT_DIR, "runs_b_c.csv")


def get_runs(D, y, force=False):
    """5 CountSketch runs per m, shared by parts (b) and (c). Cached on disk."""
    if os.path.exists(RUNS_PATH) and not force:
        return pd.read_csv(RUNS_PATH)
    loss_b=get_baseline(D, y)["loss_b"]
    n=D.shape[0]
    rows=[]
    for m in M_LIST:
        for r in range(N_REPEATS):
            rng=np.random.default_rng([SEED, m, r])
            t0=time.perf_counter()
            #drawing S and forming SD and Sy
            SD, Sy=countsketch(D, y, m, rng)    
            #sketched solution    
            a=lstsq(SD, Sy)               
            #time for c recording       
            t=time.perf_counter() - t0              
            full_loss=loss(D, y, a)
            sk=float(np.sum((SD @ a - Sy) ** 2))    # sketched loss
            rows.append(dict(m=m, run=r, loss=full_loss, ratio=full_loss / loss_b, sketched_loss=sk, time=t))
            print(f"m={m:5d} run={r}  Loss={full_loss:.4f}  ratio={full_loss / loss_b:.4f}  "
                  f"sketched={sk:.4f}  time={t:.2f}s")
    results=pd.DataFrame(rows)
    results.to_csv(RUNS_PATH, index=False)
    return results
