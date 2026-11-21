"""
lr_e.py  -  (e) Subset of 50,000 rows (chosen once with a fixed seed), m = 500.
Sketch (D', y') with (i) CountSketch and (ii) a dense Gaussian S with N(0, 1/m) entries.
For each: time to draw S and form SD', Sy', and Loss(D',y',a~) / Loss(D',y',b'), mean of 5 fresh sketches.
"""
from q2_helper import *

M_E = 500
N_SUB = 50_000

D, y = load_data()
n = D.shape[0]

# same subset for both methods and all repetitions
idx = np.random.default_rng(SEED + 12345).choice(n, size=N_SUB, replace=False)
Dp, yp = np.asfortranarray(D[idx]), y[idx]
bp = lstsq(Dp, yp)
loss_bp = loss(Dp, yp, bp)
print(f"Subset: {N_SUB} rows.  Loss(D', y', b') = {loss_bp:.6f}")

rows = []
for r in range(N_REPEATS):
    # (i) CountSketch  (fresh sketch each repetition)
    rng = np.random.default_rng([SEED, M_E, 7000 + r])
    t0 = time.perf_counter()
    SD, Sy = countsketch(Dp, yp, M_E, rng)
    t_form = time.perf_counter() - t0
    a = lstsq(SD, Sy)
    rows.append(dict(method="CountSketch", rep=r, time_sketch=t_form,
                     ratio=loss(Dp, yp, a) / loss_bp))

    # (ii) dense Gaussian sketch S in R^{m x 50000}, N(0, 1/m) entries
    rng = np.random.default_rng([SEED, M_E, 8000 + r])
    t0 = time.perf_counter()
    S = rng.normal(0.0, 1.0 / np.sqrt(M_E), size=(M_E, N_SUB))
    SD = S @ Dp
    Sy = S @ yp
    t_form = time.perf_counter() - t0
    a = lstsq(SD, Sy)
    rows.append(dict(method="Dense Gaussian", rep=r, time_sketch=t_form,
                     ratio=loss(Dp, yp, a) / loss_bp))
    del S

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, "runs_e.csv"), index=False)
table = res.groupby("method").agg(time_sketch_mean=("time_sketch", "mean"),
                                  ratio_mean=("ratio", "mean"),
                                  ratio_min=("ratio", "min"),
                                  ratio_max=("ratio", "max")).reset_index()
table.to_csv(os.path.join(OUT_DIR, "table_e.csv"), index=False)
print("\nAll repetitions:")
print(res.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
print("\nMean over 5 repetitions (time = draw S + form SD', Sy'):")
print(table.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
print(f"\nSaved: {OUT_DIR}/runs_e.csv, table_e.csv")