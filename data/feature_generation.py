"""kmer feature generations script"""

import numpy as np
from itertools import product
import pandas as pd
import h5py
import os

def gen_kmers(k):
    res = []

    for i in product('ACGU', repeat=k):
        kmer = ''.join(i)
        res.append(kmer)

    return res

def gen(csv, out, k_values=[1, 2, 3, 4]):

    
    df = pd.read_csv(csv)
    df2 = df[df['split'] == 'train'].copy()
    sequences = df2['Sequence'].tolist()
    
    num_feats = 0
    for i in k_values:
        num_feats += 4**i
    num_seqs = len(sequences)
    names = []



    for i in k_values:
        kmers = gen_kmers(i)
        for j in kmers:
            names.append(f'kmer_{i}_{j}')
    
    with h5py.File(out, 'w') as hf:
        main_dset = hf.create_dataset('kmer_features', shape=(num_seqs, num_feats), dtype=np.float32)
        x = []
        for name in names:
            x.append(name.encode())
        hf.create_dataset('feature_names', data=x)
        hf.create_dataset('index', data=df2['Index'].values.astype(int))
        
        for i in range(len(sequences)):
            s = sequences[i]
            print(f"Processing sequence {i}")
            
            vec = np.zeros(num_feats, dtype=np.float32)
            slen = len(s)
            feat_id = 0
            
            for j in k_values:
                kmers = gen_kmers(j)
                
                k_counts = {}
                for k in range(0, len(s) - j + 1):
                    kmer = s[k:k+j]
                    k_counts[kmer] = k_counts.get(kmer, 0) + 1
                

                for k in kmers:
                    count = k_counts.get(k, 0)

                    if slen > 0:
                        freq = count / slen 
                    else:
                        freq = 0


                    vec[feat_id] = freq
                    feat_id += 1
            
            main_dset[i] = vec
    
    return out





script = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(script)
processed = os.path.join(root, 'data', 'processed')
csv = os.path.join(processed, 'filtered_data_with_split.csv')

if __name__ == "__main__":
    h5_output = gen(csv, os.path.join(processed, "kmer_features.h5"))