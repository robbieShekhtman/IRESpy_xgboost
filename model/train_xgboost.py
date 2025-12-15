"""script to train and evaluate the model"""

import os
import warnings
import numpy as np
import pandas as pd
import h5py
import glob
from typing import List
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


#bunch of globals consts that dont change
dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(dir)
data = os.path.join(root, "data", "processed")
csv = os.path.join(data, "filtered_data_with_split.csv")
kmer = os.path.join(data, "kmer_features.h5")
kmer_test = os.path.join(data, "kmer_features_test.h5")
struct = os.path.join(data, "struct_features.h5")
model_json = os.path.join(dir, "xgb_ires.json")
model_normal = os.path.join(dir, "xgb_ires.model")
random_state = 42


def load_base():
    df = pd.read_csv(csv)
    df["label_bin"] = (df["label"].str.upper() == "IRES").astype(int)
    
    return df


def load_kmer_file(path: str, ds: str = "kmer_features"):

    with h5py.File(path, "r") as f:

        X = f[ds][:]
        names_raw = f["feature_names"][:]
        ids = f["index"][:]
    
    names = []

    for i in names_raw:
        if isinstance(i, (bytes, bytearray)):
            names.append(i.decode("utf-8"))
        else:
            names.append(str(i))

    kmer = pd.DataFrame(X, index=ids.astype(int), columns=names)
    kmer.index.name = "Index"
    
    return kmer

def load_kmer():
    return load_kmer_file(kmer, "kmer_features")

def load_kmer_test():

    if not os.path.exists(kmer_test):
        return pd.DataFrame()
    
    return load_kmer_file(kmer_test, "kmer_features_test")

def load_qmfe_train():

    pattern = os.path.join(data, "q_mfe_features_batch_*.h5")
    files = sorted(glob.glob(pattern))
    indices = []
    q_vals = []

    if len(files) ==0:
        return pd.DataFrame(columns=["q_mfe"]).set_index(pd.Index([], name="Index"))
    
    for i in files:

        with h5py.File(i, "r") as f:

            if "index" not in f or "q_mfe_subset" not in f:
                continue
            ind = f["index"][:]
            q = f["q_mfe_subset"][:]
            indices.append(ind.astype(int))
            q_vals.append(q.astype(np.float32))
    
    if len(indices) == 0:
        return pd.DataFrame(columns=["q_mfe"]).set_index(pd.Index([], name="Index"))
    
    inds_all = np.concatenate(indices)
    q_all = np.concatenate(q_vals)
    df_q = pd.DataFrame({"Index": inds_all, "q_mfe": q_all})
    df_q = df_q.set_index("Index")

    if df_q.index.duplicated().any():
        df_q = df_q[~df_q.index.duplicated(keep='first')]
    

    return df_q


def load_qmfe_test():

    if not os.path.exists(struct):
        return pd.DataFrame(columns=["q_mfe"]).set_index(pd.Index([], name="Index"))
        
    with h5py.File(struct, "r") as f:
        if "q_mfe" in f and "index" in f:

            q = f["q_mfe"][:]
            ind = f["index"][:]

        elif "q_mfe_subset" in f and "index" in f:

            q = f["q_mfe_subset"][:]
            ind = f["index"][:]

        else:
            return pd.DataFrame(columns=["q_mfe"]).set_index(pd.Index([], name="Index"))
    
    df_q = pd.DataFrame({"Index": ind.astype(int), "q_mfe": q.astype(np.float32)})
    df_q = df_q.set_index("Index")
    
    return df_q


def build_features():

    df = load_base()
    kmer_tr = load_kmer()
    kmer_tst = load_kmer_test()
    
    if len(kmer_tst) > 0:
        kmer = pd.concat([kmer_tr, kmer_tst])
    else:
        kmer = kmer_tr
    
    df = df.merge(kmer, on="Index", how="inner")
    qmfe_tr = load_qmfe_train()
    qmfe_tst = load_qmfe_test()
    
    if len(qmfe_tr) > 0 and len(qmfe_tst) > 0:
        qmfe = pd.concat([qmfe_tr, qmfe_tst])

    elif len(qmfe_tr) > 0:
        qmfe = qmfe_tr

    elif len(qmfe_tst) > 0:
        qmfe = qmfe_tst

    else:
        qmfe = pd.DataFrame(columns=["q_mfe"]).set_index(pd.Index([], name="Index"))
    
    if len(qmfe) > 0:
        df = df.merge(qmfe, left_on="Index", right_index=True, how="left")
    else:
        df["q_mfe"] = np.nan
    
    cols = []
    for c in df.columns:
        if c.startswith("kmer_"):
            cols.append(c)
    if len(qmfe) > 0 and "q_mfe" in df.columns:
        cols.append("q_mfe")

    m_train = df["split"] == "train"
    m_val = df["split"] == "val"
    m_test = df["split"] == "test"
    X = df[cols].values.astype(np.float32)
    y = df["label_bin"].values
    Xtr, Ytr = X[m_train], y[m_train]
    Xval, Yval = X[m_val], y[m_val]
    Xtst, Ytst = X[m_test], y[m_test]    


    return Xtr, Ytr, Xval, Yval, Xtst, Ytst


def train(Xtr: np.ndarray, Ytr: np.ndarray, Xval: np.ndarray,Yval: np.ndarray):

    m = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.01,
        gamma=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_depth=5,
        min_child_weight=19,
        subsample=0.8,
        colsample_bytree=0.65,
        n_estimators=2000,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    
    if Xval.shape[0] > 0:
        eval_set = [(Xtr, Ytr), (Xval, Yval)]
    else:
        eval_set = [(Xtr, Ytr)]
    
    m.fit(Xtr,Ytr,eval_set=eval_set,verbose=False)
    
    return m


def evaluate(m: XGBClassifier,X: np.ndarray,y: np.ndarray,split: str):

    if X.shape[0] == 0:
        return
    
    proba = m.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    
    print("\n")
    print(f"Metrics on {split} split:")
    print("\n")
    print(f"AUC:{auc}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print("\n")


def main():

    Xtr, Ytr, Xval, Yval, Xtst, Ytst = build_features()
    m = train(Xtr, Ytr, Xval, Yval)
    evaluate(m, Xtst, Ytst, "test")
    m.save_model(model_json)
    #save both formats just incase
    m.save_model(model_normal)


if __name__ == "__main__":
    main()

