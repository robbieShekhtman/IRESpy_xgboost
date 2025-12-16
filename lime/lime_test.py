import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train_xgboost import (load_kmer,load_kmer_test,load_qmfe_train,load_qmfe_test,build_features)
from lime.lime import LIME


def load_model():
    'loads trained xgboost model'
    m = XGBClassifier()
    m.load_model(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "xgb_ires.model"))
    return m


def get_features():
    "returns list of feature column names"
    kt = load_kmer()
    ks = load_kmer_test()
    
    if len(ks) > 0:
        k = pd.concat([kt, ks])
    else:
        k = kt
    
    qt = load_qmfe_train()
    qs = load_qmfe_test()
    q = pd.concat([qt, qs])
    
    c = []
    for i in k.columns:
        if i.startswith("kmer_"):
            c.append(i)
    if len(q) > 0 and "q_mfe" in q.columns:
        c.append("q_mfe")
    
    return c


def get_samples(ni, nn, rs):
    "builds samples of positive and negative instances"
    xt, yt, xv, yv, xs, ys = build_features()

    xa = np.vstack([xt, xv, xs])
    ya = np.concatenate([yt, yv, ys])
    np.random.seed(rs)
    ii = np.where(ya == 1)[0]
    ni2 = np.where(ya == 0)[0]
    is1 = np.random.choice(ii, size=min(ni, len(ii)), replace=False)
    ns = np.random.choice(ni2, size=min(nn, len(ni2)), replace=False)



    il = []
    for i in is1:
        il.append((i, xa[i], ya[i]))
    
    nl = []
    for i in ns:
        nl.append((i, xa[i], ya[i]))
    
    ins = {'ires': il, 'non_ires': nl}
    
    return ins


def print_details(e, lt, tk, idx):
    "prints explanation details for a sample"
    print("\n")
    print(f"type: {lt} (Index: {idx})")
    print("\n")
    print(f"prediction: {e['prediction']:}")
    print(f"intercept: {e['intercept']:}")
    print(f"features used: {e['features_used']}")
    
    pt = 0
    for _, i in e['positive_features']:
        pt += i
    nt = 0
    for _, i in e['negative_features']:
        nt += i
    print(f"positive contribution: {pt:}")
    print(f"negative contribution: {nt:}")
    print(f"net contribution: {pt + nt:}")
    
    nba = e.get('negative_abs', e['negative_features'])
    if nba:
        print("\n")
        print("negative features (by absolute value):")
        for i, j in nba[:20]:
            print(f"  {i}: {j:} (abs: {abs(j):})")
    
    print("\n")
    print(f"top positive features (coefficient %):")
    for i, j in e['positive_features'][:tk]:
        if pt > 0:
            p = (j / pt * 100)
        else:
            p = 0
        print(f"  {i}: {j:} ({p:}%)")
    
    ns = []
    for i in nba:
        if abs(i[1]) > 0.00001:
            ns.append(i)
    if ns:
        print("\n")
        print(f"top negative features (coefficient %):")
        for i, j in ns[:tk]:
            if nt < 0:
                p = (abs(j) / abs(nt) * 100)
            else:
                p = 0
            print(f"  {i}: {j:} ({p:}%)")


def main():    
    "runs lime explanations for sampled instances"
    ins = get_samples(1, 1, None)
    exp = LIME(load_model(), 4000, 0.7, 42)
    
    for i in range(1, len(ins['ires']) + 1):
        j, k, _ = ins['ires'][i - 1]
        print_details(exp.explain(k, get_features(), 0.01), "IRES", 15, j)
    print("\n")
    for i in range(1, len(ins['non_ires']) + 1):
        j, k, _ = ins['non_ires'][i - 1]
        print_details(exp.explain(k, get_features(), 0.01), "non-IRES", 15, j)


if __name__ == "__main__":
    main()

