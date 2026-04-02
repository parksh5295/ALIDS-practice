"""
Convert NSL-KDD.txt (KDDTrain+, KDDTest+) to .csv files for the project.
KDDTrain+ is split into KDDTrain.csv and KDDVal.csv at a 9:1 ratio.
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# 프로젝트 루트를 path에 넣어 data._HEADERS 사용
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import _HEADERS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Convert NSL-KDD .txt to KDDTrain/KDDVal/KDDTest .csv')
    parser.add_argument(
        '--train-txt',
        default=os.path.join(_ROOT, 'data', 'KDDTrain+.txt'),
        help='Path to KDDTrain+.txt',
    )
    parser.add_argument(
        '--test-txt',
        default=os.path.join(_ROOT, 'data', 'KDDTest+.txt'),
        help='Path to KDDTest+.txt',
    )
    parser.add_argument(
        '--out-dir',
        default=os.path.join(_ROOT, 'data'),
        help='Directory for KDDTrain.csv, KDDVal.csv, KDDTest.csv',
    )
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation fraction from training (default 0.1 = 9:1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    args = parser.parse_args()

    if not os.path.isfile(args.train_txt):
        print(f'Error: train file not found: {args.train_txt}', file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.test_txt):
        print(f'Error: test file not found: {args.test_txt}', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    train_full = pd.read_csv(args.train_txt, header=None, names=_HEADERS, low_memory=False)
    test_df = pd.read_csv(args.test_txt, header=None, names=_HEADERS, low_memory=False)

    stratify = None
    if 'class' in train_full.columns:
        counts = train_full['class'].value_counts()
        if (counts >= 2).all():
            stratify = train_full['class']

    try:
        train_df, val_df = train_test_split(
            train_full,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=stratify,
        )
    except ValueError:
        train_df, val_df = train_test_split(
            train_full,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=None,
        )

    out_train = os.path.join(args.out_dir, 'KDDTrain.csv')
    out_val = os.path.join(args.out_dir, 'KDDVal.csv')
    out_test = os.path.join(args.out_dir, 'KDDTest.csv')

    train_df.to_csv(out_train, header=False, index=False)
    val_df.to_csv(out_val, header=False, index=False)
    test_df.to_csv(out_test, header=False, index=False)

    print(f'Wrote {out_train}  ({len(train_df)} rows)')
    print(f'Wrote {out_val}    ({len(val_df)} rows)')
    print(f'Wrote {out_test}   ({len(test_df)} rows)')


if __name__ == '__main__':
    main()
