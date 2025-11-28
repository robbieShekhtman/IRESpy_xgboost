import numpy as np
from itertools import product
import pandas as pd
import h5py
import os

def generate_all_kmers(k, alphabet='ACGU'):
    """Generate all possible k-mers of length k"""
    return [''.join(comb) for comb in product(alphabet, repeat=k)]

def save_kmer_features_to_hdf5(csv_file_path, output_h5_path, k_values=[1, 2, 3, 4]):
    """
    Save k-mer features to HDF5 format for efficient storage and access
    """
    # Read all sequences
    df = pd.read_csv(csv_file_path)
    sequences = df['Sequence'].tolist()  # Adjust column name

    # Add an explicit Index column that matches the sequence index used for k-mer features
    # This ensures the CSV contains a column named 'Index' matching the HDF5 'index'
    if 'Index' not in df.columns:
        df['Index'] = df.index.values.astype(int)
    
    total_features = sum(4**k for k in k_values)
    n_sequences = len(sequences)
    
    # Generate feature names
    feature_names = []
    for k in k_values:
        kmers = generate_all_kmers(k)
        feature_names.extend([f'kmer_{k}_{kmer}' for kmer in kmers])
    
    # Create HDF5 file
    with h5py.File(output_h5_path, 'w') as hf:
        # Create dataset for features
        features_dset = hf.create_dataset('kmer_features', 
                                         shape=(n_sequences, total_features),
                                         dtype=np.float32)
        
        # Store feature names
        hf.create_dataset('feature_names', data=[name.encode() for name in feature_names])
        
        # Store sequence identifiers (as 'index' to match CSV column name)
        hf.create_dataset('index', data=df['Index'].values.astype(int))
        
        # Process and store features
        for seq_idx, sequence in enumerate(sequences):
            if seq_idx % 1000 == 0:
                print(f"Processing sequence {seq_idx}/{n_sequences}")
            
            feature_vector = np.zeros(total_features, dtype=np.float32)
            seq_len = len(sequence)
            feature_idx = 0
            
            for k in k_values:
                kmers = generate_all_kmers(k)
                
                # Count k-mers
                kmer_counts = {}
                for i in range(0, len(sequence) - k + 1):
                    kmer = sequence[i:i+k]
                    kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
                
                # Calculate frequencies
                for kmer in kmers:
                    count = kmer_counts.get(kmer, 0)
                    frequency = count / seq_len if seq_len > 0 else 0
                    feature_vector[feature_idx] = frequency
                    feature_idx += 1
            
            # Store in HDF5
            features_dset[seq_idx] = feature_vector
    
    print(f"K-mer features saved to {output_h5_path}")
    return output_h5_path

# Usage
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
processed_dir = os.path.join(project_root, 'data', 'processed')
csv_file_path = os.path.join(processed_dir, 'filtered_data_with_split.csv')



if __name__ == "__main__":
    # Example usage 
    h5_output = save_kmer_features_to_hdf5(csv_file_path, os.path.join(processed_dir, "kmer_features.h5"))