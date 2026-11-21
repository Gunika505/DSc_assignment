"""
lr_a.py  -  (a) Full least squares: b = argmin ||D a - y||^2 (numpy.linalg.lstsq).
Reports Loss(D, y, b) and the time to compute b (also rank(D), needed in (b)).
"""
from q2_helper import *

D, y = load_data()
n, d = D.shape
print(f"D: n = {n}, d = {d};  attacks: {int(y.sum())} of {n} ({100 * y.mean():.1f}%)")

base = get_baseline(D, y, force=True)
print(f"Loss(D, y, b)   = {base['loss_b']:.6f}")
print(f"time to get b   = {base['time_b']:.3f} s")
print(f"rank(D)         = {base['rank']}  (d = {d})")
print(f"Saved: {BASELINE_PATH}")