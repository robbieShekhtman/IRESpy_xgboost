"""
filter_data_template.py

Template script to filter tab-separated data files (.tab.txt) using pandas.

Features:
- Select columns to keep
- Apply simple filters like `col>5`, `col>=3.2`, `col==value`, `col!=value`, `col contains substring`
- Drop rows with NA
- Process large files in chunks to save memory

Example usage:
python data/filter_data_template.py -i data/55k_oligos_sequence_and_expression_measurements.tab.txt -o out.filtered.tab.txt \
    -c sequence,expression -f "expression>5" --dropna

Requirements:
- pandas (see `requirements.txt`)
"""

import argparse
import os
import sys
import re

try:
    import pandas as pd
except Exception:
    print("Error: pandas is required. Install with `pip install pandas`.")
    sys.exit(1)


FILTER_RE = re.compile(r"^(?P<col>[^!=<>\s]+)\s*(?P<op>>=|<=|==|!=|>|<|:contains:)\s*(?P<val>.+)$")


def parse_filter(expr):
    m = FILTER_RE.match(expr)
    if not m:
        raise ValueError(f"Invalid filter expression: {expr}")
    col = m.group('col')
    op = m.group('op')
    val = m.group('val')
    if op != ':contains:':
        # try to convert to number
        try:
            if '.' in val:
                val = float(val)
            else:
                val = int(val)
        except Exception:
            # leave as string
            pass
    return col, op, val


def apply_filters(df, filters):
    if not filters:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    for f in filters:
        col, op, val = parse_filter(f)
        if col not in df.columns:
            # treat missing column as all False
            mask &= False
            continue
        if op == ':contains:':
            mask &= df[col].astype(str).str.contains(str(val), na=False)
        elif op == '==':
            mask &= df[col] == val
        elif op == '!=':
            mask &= df[col] != val
        elif op == '>':
            mask &= pd.to_numeric(df[col], errors='coerce') > float(val)
        elif op == '>=':
            mask &= pd.to_numeric(df[col], errors='coerce') >= float(val)
        elif op == '<':
            mask &= pd.to_numeric(df[col], errors='coerce') < float(val)
        elif op == '<=':
            mask &= pd.to_numeric(df[col], errors='coerce') <= float(val)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return df[mask]


def main():
    parser = argparse.ArgumentParser(description="Filter tab-separated data files (TSV) using simple CLI filters.")
    parser.add_argument('-i', '--input', required=True, help='Input .tab.txt (TSV) file')
    parser.add_argument('-o', '--output', required=False, help='Output TSV file (default: <input>.filtered.tab.txt)')
    parser.add_argument('-c', '--columns', help='Comma-separated list of columns to keep (order preserved)')
    parser.add_argument('-f', '--filter', action='append', help='Filter expression, e.g. "expression>5" or "sequence :contains: ATG". Can be provided multiple times.')
    parser.add_argument('--dropna', action='store_true', help='Drop rows that contain any NA values after filtering/selecting columns')
    parser.add_argument('--chunksize', type=int, default=100000, help='Number of rows per chunk when reading large files')
    args = parser.parse_args()

    inp = args.input
    if not os.path.isfile(inp):
        print(f"Input file not found: {inp}")
        sys.exit(2)

    out = args.output or (inp + '.filtered.tab.txt')
    cols = [c.strip() for c in args.columns.split(',')] if args.columns else None
    filters = args.filter or []
    chunksize = args.chunksize

    write_header = True
    try:
        reader = pd.read_csv(inp, sep='\t', chunksize=chunksize, dtype=str)
    except Exception as e:
        # try without chunks
        try:
            df = pd.read_csv(inp, sep='\t', dtype=str)
            dfs = [df]
        except Exception as e2:
            print(f"Failed to read input file: {e2}")
            sys.exit(3)
    else:
        dfs = reader

    for chunk in dfs:
        # convert columns types as needed are left as strings to avoid conversion errors
        if cols:
            missing = [c for c in cols if c not in chunk.columns]
            if missing:
                print(f"Warning: columns not found and will be filled with NA: {missing}")
                for c in missing:
                    chunk[c] = pd.NA
            chunk = chunk[cols]
        chunk = apply_filters(chunk, filters)
        if args.dropna:
            chunk = chunk.dropna()
        if chunk.empty:
            continue
        mode = 'w' if write_header else 'a'
        header = write_header
        chunk.to_csv(out, sep='\t', index=False, mode=mode, header=header)
        write_header = False

    print(f"Filtered output written to: {out}")


if __name__ == '__main__':
    main()
