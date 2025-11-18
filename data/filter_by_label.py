""" script should filter out entries by label and save the output to a csv in /processed/"""

import os
import sys
import pandas as pd


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    input_file = os.path.join(raw_data_dir, '55k_oligos_sequence_and_expression_measurements.tab.txt')
    output_file = os.path.join(processed_dir, 'filtered_data.csv')
    
    
    valid_labels = [
        'CDS_screen',
        'Genome_Wide_Sceen_Elements',
        'High_Priority_Genes_Blocks',
        'High_Priority_Viruses_Blocks',
        'Human_5UTR_Screen',
        'IRESite_blocks',
        'Viral_5UTR_Screen',
        'rRNA_Matching_5UTRs'
    ]
    
    try:
        df = pd.read_csv(input_file, sep='\t', low_memory=False, encoding='latin-1')
    except Exception as e:
        sys.exit(1)
    
    if 'Oligo_name' in df.columns:
        pattern = '|'.join(valid_labels)
        mask = df['Oligo_name'].astype(str).str.contains(pattern, na=False, regex=True)
        df = df[mask]
    else:
        print("column not found. Skipping label filter.")
    
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
