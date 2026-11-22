
import os
import ssl
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
"""
here we build the data matrix D from the KDD Cup 1999 (10%) dataset.

steps are given in the assingment doc
  (i)one-hot encode protocol_type, service, flag; other 38 features -> float
  (ii)drop every column that is constant over all rows
  (iii)min-max rescale each column to [0, 1]

finally the outputs (in ./data):
  kdd_raw.csv: raw features + label, exactly as fetched here text decoded
  kdd_final.npz: D (float32, n x d), y (labels), columns (feature names)
"""

# Fix for macOS python.org installs: "CERTIFICATE_VERIFY_FAILED" when sklearn
# downloads the dataset. Use certifi's CA bundle for HTTPS requests.
try:
    import certifi
    ssl._create_default_https_context=lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    pass

OUT_DIR="data"
CAT_COLS=["protocol_type", "service", "flag"]


def decode_bytes(df):
    """sklearn returns text columns as bytes objects; turn them into str."""
    df=df.copy()
    for c in df.columns:
        if df[c].dtype==object:
            df[c]=df[c].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    #fetching the default 10% version
    kdd=fetch_kddcup99(subset=None, as_frame=True)
    X=decode_bytes(kdd.data)
    y=kdd.target.map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    y=y.rename("label")

    #saving the raw
    raw=X.copy()
    raw["label"]=y.values
    raw_path=os.path.join(OUT_DIR, "kdd_raw.csv")
    raw.to_csv(raw_path, index=False)
    print(f"Raw data saved: {raw_path}  shape={raw.shape}")

    #part 1 as mentioned up
    numeric_cols=[c for c in X.columns if c not in CAT_COLS]
    assert len(numeric_cols)==38, len(numeric_cols)
    num=X[numeric_cols].astype(np.float32)
    cat=pd.get_dummies(X[CAT_COLS], dtype=np.float32)
    feature_df=pd.concat([num, cat], axis=1)

    #part 2 as mentioned up
    keep=feature_df.nunique(dropna=False)>1
    dropped_cols=list(feature_df.columns[~keep])
    feature_df=feature_df.loc[:, keep]
    print(f"Dropped {len(dropped_cols)} constant columns: {dropped_cols}")

    #part 3 as mentioned up
    D=feature_df.to_numpy(dtype=np.float32)
    mn=D.min(axis=0)
    mx=D.max(axis=0)
    #mx>mn guaranteed after step part 2
    D=(D - mn) / (mx - mn)          
    D=D.astype(np.float32)

    n, d=D.shape
    print(f"D has n = {n} rows and d = {d} columns")
    print(f"min = {D.min():.3f}, max = {D.max():.3f}, dtype = {D.dtype}")

    #saving final
    final_path=os.path.join(OUT_DIR, "kdd_final.npz")
    np.savez_compressed(
        final_path,
        D=D,
        y=y.to_numpy().astype(str),
        columns=np.array(feature_df.columns.astype(str)),
    )
    print(f"Final data saved: {final_path}")


if __name__=="__main__":
    main()
