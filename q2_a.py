from q2_helper import *
"""
lr_a.py - (a) Full least squares solution: b = argmin ||D a - y||^2,
computed using numpy.linalg.lstsq. Reports Loss(D, y, b), the computation
time for b, and rank(D) which is needed for b part.
"""

D, y=load_data()
n, d=D.shape
print(f"D: n={n}, d={d};  attacks: {int(y.sum())} of {n} ({100 * y.mean():.1f}%)")

base=get_baseline(D, y, force=True)
print(f"Loss(D, y, b)={base['loss_b']:.6f}")
print(f"time to get b={base['time_b']:.3f} s")
print(f"rank(D)={base['rank']}  (d = {d})")
print(f"Saved: {BASELINE_PATH}")
