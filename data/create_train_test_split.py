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
    Y = df['label']

    Xtr, Xtmp, Ytr, Ytmp = train_test_split( X, Y, test_size=0.2, stratify=Y, random_state=42)
    Xtst, Xval, Ytst, Yval = train_test_split(Xtmp, Ytmp, test_size=0.5, stratify=Ytmp, random_state=42)

    train = Xtr.copy()
    train['label'] = Ytr.values
    train['split'] = 'train'


    test = Xtst.copy()
    test['label'] = Ytst.values
    test['split'] = 'test'


    validate = Xval.copy()
    validate['label'] = Yval.values
    validate['split'] = 'val'
    
    wpslit = pd.concat([train, test, validate], ignore_index=True)
    origninal = list(df.columns)

    
    wpslit = wpslit[origninal + ['split']]
    wpslit.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()

