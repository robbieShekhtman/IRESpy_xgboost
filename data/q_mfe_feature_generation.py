import numpy as np
import pandas as pd
import h5py
import subprocess
import os
import sys
import re
from typing import List, Optional
import ushuffle as ushuffle_pkg



def run_rnafold(seq):
    "runs rnafold on sequence and returns mfe value"
    p = subprocess.run( ['RNAfold', '--noPS'], input=seq, capture_output=True, text=True, timeout=30, check=False)
        
    if p.returncode != 0:
            return None
        
    lines = p.stdout.strip().split('\n')
    if len(lines) < 2:
            return None
        
    line = lines[-1]
    m = re.search(r'\((-?\d+\.?\d*)\)', line)
    if m:
        return float(m.group(1))
    return None

def generate_shuffled_sequences(seq , n):
    "generates n shuffled sequences from input sequence"
    b = seq.encode('utf-8')
    shuffled = []
    for _ in range(n):
        result = ushuffle_pkg.shuffle(b, 2)
        if result:
            out = result.decode('utf-8') if isinstance(result, bytes) else str(result)
            shuffled.append(out)

            
    return shuffled
    


def compute_q_mfe_for_sequence(seq, mfe, n):
    "computes q_mfe score for a sequence using n shuffled sequences"
    out = {'q_mfe': np.nan,'native_mfe': mfe,'shuffled_mfes': [],'n_shuffles_used': 0,'success': False}
    
    if mfe is None:
        m = run_rnafold(seq)
        if m is None:
            return out
        mfe = m
        out['native_mfe'] = mfe
    
    shuffled = generate_shuffled_sequences(seq, n)
    if len(shuffled) == 0: # type: ignore
        return out
    
    mfes = []
    for s in shuffled: # type: ignore
        m = run_rnafold(s)
        if m is not None:
            mfes.append(m)
    
    if len(mfes) == 0:
        return out
    
    out['shuffled_mfes'] = mfes
    out['n_shuffles_used'] = len(mfes)
    count = 0
    for m in mfes:
        if m <= mfe:
            count += 1
    total = len(mfes)
    out['q_mfe'] = count / (total + 1)
    out['success'] = True
    
    return out


def compute_q_mfe_for_indices(df, indices, n):
    "computes q_mfe for multiple indices from dataframe"
    q = []
    inds = []
    mfe = []
    processed = 0
    last = None

    for ind in indices:
        print(ind)
        
        row = df[df['Index'] == ind]
        if len(row) == 0:
            processed += 1
            last = ind
            continue
        
        r = row.iloc[0]
        seq = r['Sequence']
        mfe = r['mfe']
        res = compute_q_mfe_for_sequence(seq,mfe,n)
        
        if res['success']:
            q.append(res['q_mfe'])
            inds.append(ind)
            mfe.append(res['native_mfe'])
        else:
            q.append(np.nan)
            inds.append(ind)

            if res['native_mfe'] is not None:
                mfe.append(res['native_mfe'])
            else:
                mfe.append(np.nan)
        
        last = ind
        processed += 1
    
    q = np.array(q, dtype=np.float32)
    inds = np.array(inds, dtype=int)
    mfe = np.array(mfe, dtype=np.float32)
    
    return q, inds, mfe, last




def save_q_mfe_features_to_hdf5(qvals, inds, mfes, out):
    "saves q_mfe features to hdf5 file"
    with h5py.File(out, 'w') as f:
        f.create_dataset('index', data=inds.astype(int), dtype=int)
        f.create_dataset('native_mfe_subset', data=mfes.astype(np.float32), dtype=np.float32)
        f.create_dataset('q_mfe_subset', data=qvals.astype(np.float32), dtype=np.float32)



def load_progress(pf):
    "loads last processed index from progress file"
    if not os.path.exists(pf):
        return None
    
    with open(pf, 'r') as f:
        ind = int(f.read().strip())
    return ind


def save_progress(file, ind):
    "saves last processed index to progress file"
    with open(file, 'w') as f:
        f.write(str(ind))


def get_next_batch_number(dir, base):
    "gets next available batch number for output file"
    num = 0
    while True:
        bf = os.path.join(dir, f"{base}_batch_{num:04d}.h5")
        if not os.path.exists(bf):
            return num
        num += 1


def main():
    "main function to compute and save q_mfe features in batches"
    sdir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(sdir)
    processed_dir = os.path.join(root, 'data', 'processed')
    csv = os.path.join(processed_dir, 'filtered_data_with_split.csv')
    progress = os.path.join(processed_dir, 'q_mfe_progress.txt')
    df = pd.read_csv(csv)
    train = df[df['split'] == 'train'].copy()
    inds = train['Index'].tolist()

    
    last = load_progress(progress)
    if last is not None:
        try:
            pos = inds.index(last)
            unprocessed = inds[pos + 1:]
        except ValueError:
            unprocessed = inds
    else:
        unprocessed = inds
        
    max_ent = 9500
    unprocessed = unprocessed[:max_ent]
    
    if len(unprocessed) == 0:
        print("nothing left")
        sys.exit(0)
    
    qvals, inds, mfes, last = compute_q_mfe_for_indices(df, unprocessed, 100)

    bnum = get_next_batch_number(processed_dir, 'q_mfe_features')
    out = os.path.join(processed_dir, f'q_mfe_features_batch_{bnum}.h5')
    save_q_mfe_features_to_hdf5(qvals, inds, mfes,out )
        
    if last is not None:
        save_progress(progress, last)


if __name__ == '__main__':
    main()
