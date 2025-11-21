""" create 80/10/10 train/test/validation split and add split column"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    processed_dir = os.path.join(project_root, 'data', 'processed')
    input_file = os.path.join(processed_dir, 'filtered_data.csv')
    output_file = os.path.join(processed_dir, 'filtered_data_with_split.csv')
    
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        sys.exit(1)
    
    if 'label' not in df.columns:
        sys.exit(1)
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    train_df = X_train.copy()
    train_df['label'] = y_train.values
    train_df['split'] = 'train'
    test_df = X_test.copy()
    test_df['label'] = y_test.values
    test_df['split'] = 'test'
    val_df = X_val.copy()
    val_df['label'] = y_val.values
    val_df['split'] = 'val'
    
    df_with_split = pd.concat([train_df, test_df, val_df], ignore_index=True)
    original_cols = list(df.columns)
    df_with_split = df_with_split[original_cols + ['split']]
    df_with_split.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()

