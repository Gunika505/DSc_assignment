"""
part_a.py  -  (a) k-means on D (k = 15, n_init = 5): cost(D, B) and wall-clock time.
Run first (or let the other parts run it automatically).
"""
from q1_helper import *

D = load_D()
n, d = D.shape
print(f"D: n = {n}, d = {d}")

base = get_baseline(D, force=True)          # always recompute here and refresh the cache
print(f"cost(D, B)      = {base['cost_D']:.6f}")
print(f"wall-clock time = {base['time_D']:.2f} s")
print(f"Saved: {BASELINE_PATH}")