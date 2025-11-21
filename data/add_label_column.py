""" add ires vs non ires label column"""

import os
import sys
import pandas as pd


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    processed_dir = os.path.join(project_root, 'data', 'processed')
    input_file = os.path.join(processed_dir, 'filtered_data.csv')
    output_file = os.path.join(processed_dir, 'filtered_data.csv')
    
    
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        sys.exit(1)
    
    if 'ires_activity' in df.columns:
        df['ires_activity'] = pd.to_numeric(df['ires_activity'], errors='coerce')
        df['label'] = df['ires_activity'].apply(lambda x: 'IRES' if x > 600 else 'non-IRES')
        
        label_counts = df['label'].value_counts()
        print(f"IRES entries: {label_counts.get('IRES', 0)}")
        print(f"non-IRES entries: {label_counts.get('non-IRES', 0)}")
    else:
        print("column not found. Cannot add label.")
    
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()

