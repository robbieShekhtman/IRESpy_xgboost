""" script should filter out entries by splicing score and promoter activity and save the output to a csv in /processed/"""

import os
import sys
import pandas as pd

def main():
    "our script we used to filter entires by splicing score and promoter activity described in base paper"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    processed_dir = os.path.join(project_root, 'data', 'processed')
    input_file = os.path.join(processed_dir, 'filtered_data.csv')
    output_file = os.path.join(processed_dir, 'filtered_data.csv')
    
    
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        sys.exit(1)
    
    if 'splicing_score' in df.columns:
        df['splicing_score'] = pd.to_numeric(df['splicing_score'], errors='coerce')
        mask = (df['splicing_score'] > -2.5) | (df['splicing_score'].isna())
        df = df[mask]
    else:
        print("column not found. Skipping splicing score filter.")
    
    if 'promoter_activity' in df.columns:
        df['promoter_activity'] = pd.to_numeric(df['promoter_activity'], errors='coerce')
        mask = (df['promoter_activity'] < 0.2) | (df['promoter_activity'].isna())
        df = df[mask]
    else:
        print("column not found. Skipping promoter activity filter.")
    
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()

