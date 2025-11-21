import numpy as np
from itertools import product

def generate_all_kmers(k, alphabet='ACGU'):
    """Generate all possible k-mers of length k"""
    return [''.join(comb) for comb in product(alphabet, repeat=k)]

def create_combined_kmer_vector(sequences, k_values=[1, 2, 3, 4]):
    """
    Create a combined k-mer feature matrix using NumPy
    """
    # Calculate total number of features
    total_features = sum(4**k for k in k_values)  # 4 + 16 + 64 + 256 = 340
    
    # Initialize feature matrix
    feature_matrix = np.zeros((len(sequences), total_features))
    
    # Feature names for reference
    feature_names = []
    
    for k in k_values:
        kmers = generate_all_kmers(k)
        feature_names.extend([f'kmer_{k}_{kmer}' for kmer in kmers])
    
    # Fill the feature matrix
    for seq_idx, sequence in enumerate(sequences):
        feature_idx = 0
        seq_len = len(sequence)
        
        for k in k_values:
            kmers = generate_all_kmers(k)
            
            # Count k-mers
            kmer_counts = {}
            for i in range(0, len(sequence) - k + 1):
                kmer = sequence[i:i+k]
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
            
            # Add to feature matrix
            for kmer in kmers:
                count = kmer_counts.get(kmer, 0)
                frequency = count / seq_len if seq_len > 0 else 0
                feature_matrix[seq_idx, feature_idx] = frequency
                feature_idx += 1
    
    return feature_matrix, feature_names


if __name__ == "__main__":
    # Example usage
    sequences = ["AUCGUAUCG", "GGGAAACCC", "AUGCGAUGC"]
    feature_matrix, feature_names = create_combined_kmer_vector(sequences)
    print(f"Feature matrix shape: {feature_matrix.shape}")  # (3, 340)
    print(f"Number of features: {len(feature_names)}")  # 340