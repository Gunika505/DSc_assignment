from q2_helper import *
import random
import numpy as np
random.seed(42)
np.random.seed(42)
"""
lr_e.py  for e subset of 50,000 rows that are chosen once with a fixed seed and m = 500
then we sketch (D', y') with firstly, countSketch and then secondly with a dense gaussian S with N(0, 1/m) entries
for each m time to draw S and form SD', Sy', and Loss(D', y', a~) / Loss(D', y', b'), mean of 5 fresh sketches.
"""
M_E=500
N_SUB=50_000
D,target=load_data()
n=D.shape[0]

#same subset fixed for both methods and all reps
idx=np.random.default_rng(SEED+12345).choice(n, size=N_SUB, replace=False)
D_sub,target_sub=np.asfortranarray(D[idx]),target[idx]
b_sub=lstsq(D_sub,target_sub)
loss_b_sub=loss(D_sub,target_sub,b_sub)
print(f"Subset: {N_SUB} rows.  Loss(D', target', b')={loss_b_sub:.6f}")

rows=[]
for r in range(N_REPEATS):
    #first part CountSketch with fresh sketch each repetition
    rng=np.random.default_rng([SEED,M_E,7000+r])
    t0=time.perf_counter()
    SD,Sy=countsketch(D_sub,target_sub,M_E,rng)
    t_form=time.perf_counter()-t0
    a=lstsq(SD,Sy)
    rows.append(dict(method="CountSketch",rep=r,time_sketch=t_form,ratio=loss(D_sub,target_sub,a)/loss_b_sub))

    #second part dense Gaussian sketch S
    rng=np.random.default_rng([SEED,M_E,8000+r])
    t0=time.perf_counter()
    S=rng.normal(0.0,1.0/np.sqrt(M_E),size=(M_E,N_SUB))
    SD=S@D_sub
    Sy=S@target_sub
    t_form=time.perf_counter()-t0
    a=lstsq(SD,Sy)
    rows.append(dict(method="Dense Gaussian",rep=r,time_sketch=t_form,ratio=loss(D_sub,target_sub,a)/loss_b_sub))
    del S

results=pd.DataFrame(rows)
results.to_csv(os.path.join(OUT_DIR,"runs_e.csv"),index=False)
req_table=results.groupby("method").agg(time_sketch_mean=("time_sketch","mean"),
                                        ratio_mean=("ratio","mean"),
                                        ratio_min=("ratio","min"),
                                        ratio_max=("ratio","max")).reset_index()
req_table.to_csv(os.path.join(OUT_DIR,"table_e.csv"),index=False)
print("\nAll repetitions:")
print(results.to_string(index=False,float_format=lambda v:f"{v:.5g}"))
print("\nMean over 5 repetitions (time = draw S + form SD', Sy'):")
print(req_table.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
print(f"\nSaved: {OUT_DIR}/runs_e.csv, table_e.csv")
