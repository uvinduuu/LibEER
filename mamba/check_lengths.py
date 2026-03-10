"""Quick script to show trial length distribution in 1000-sample bins."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_seediv_clips

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--sessions', nargs='+', type=int, default=None)
args = parser.parse_args()

trials, labels, _, _ = load_seediv_clips(args.dataset_path, sessions=args.sessions)

lengths = np.array([t.shape[1] for t in trials])

print(f"\n{'='*50}")
print(f"Trial Length Distribution ({len(lengths)} trials)")
print(f"  Min: {lengths.min()}, Max: {lengths.max()}, Mean: {lengths.mean():.0f}")
print(f"  Median: {np.median(lengths):.0f}")
print(f"  Std: {lengths.std():.0f}")
print(f"{'='*50}\n")

# Bin by 1000s
bin_edges = range(int(lengths.min() // 1000) * 1000,
                  int(lengths.max() // 1000 + 2) * 1000, 1000)

print(f"{'Range':>16}  {'Count':>5}  {'%':>6}  Bar")
print(f"{'-'*16}  {'-'*5}  {'-'*6}  {'-'*30}")
for lo in bin_edges:
    hi = lo + 1000
    count = np.sum((lengths >= lo) & (lengths < hi))
    if count > 0:
        pct = 100 * count / len(lengths)
        bar = '█' * int(pct)
        print(f"{lo:>7}-{hi:>7}  {count:>5}  {pct:>5.1f}%  {bar}")

# Percentiles
print(f"\nPercentiles:")
for p in [50, 75, 90, 95, 99, 100]:
    val = np.percentile(lengths, p)
    print(f"  {p:>3}th: {val:.0f} samples ({val/200:.1f}s)")

# How much padding at different fixed_lengths
print(f"\nPadding waste at different fixed_length choices:")
for fl in [20000, 25000, 30000, 35000, 40000, 43400, 51800]:
    clipped = np.minimum(lengths, fl)
    total_samples = fl * len(lengths)
    real_samples = clipped.sum()
    waste = 100 * (1 - real_samples / total_samples)
    n_cropped = np.sum(lengths > fl)
    print(f"  fixed_length={fl:>5}: {waste:>5.1f}% zeros, "
          f"{n_cropped:>3} trials cropped ({100*n_cropped/len(lengths):.1f}%)")
