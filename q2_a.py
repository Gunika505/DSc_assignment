from q2_helper import *
import random
import numpy as np
random.seed(42)
np.random.seed(42)
"""
lr_a.py for a full least squares solution: b = argmin ||D a - target||^2,
computed using numpy.linalg.lstsq. Reports Loss(D, target, b), the computation
time for b, and rank(D) which is needed for b part.
"""

D, target=load_data()
n, d=D.shape
print(f"D: n={n}, d={d};  attacks: {int(target.sum())} of {n} ({100 * target.mean():.1f}%)")

baseline=get_baseline(D, target, force=True)
print(f"Loss(D, target, b)={baseline['loss_b']:.6f}")
print(f"time to get b={baseline['time_b']:.3f} s")
print(f"rank(D)={baseline['rank']}  (d = {d})")
print(f"Saved: {BASELINE_PATH}")
