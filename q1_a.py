#q1 part (a) - compute baseline k-means on D
from q1_helper import *

D=load_D()
n,d=D.shape
print(f"D: n = {n}, d = {d}")

base = get_baseline(D, force=True)

print(f"cost(D, B)      = {base['cost_D']:.6f}")
print(f"wall-clock time = {base['time_D']:.2f} s")
print(f"Saved: {BASELINE_PATH}")