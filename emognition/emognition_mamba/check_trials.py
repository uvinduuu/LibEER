"""Quick sanity check: show Emognition trials shorter than the window size."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from emognition_loader import load_emognition_trials, FS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--window_sec', type=float, default=10.0)
args = parser.parse_args()

window_size = int(args.window_sec * FS)

trials, labels, subject_ids, lab2id, id2lab = load_emognition_trials(args.data_root)

lengths = np.array([t.shape[1] for t in trials])

print(f"\n{'='*60}")
print(f"SANITY CHECK — Emognition Trial Lengths")
print(f"  Window size: {window_size} samples ({args.window_sec}s at {FS}Hz)")
print(f"  Total trials: {len(trials)}")
print(f"{'='*60}\n")

# Trials shorter than window
short = [(i, lengths[i], subject_ids[i], id2lab[labels[i]])
         for i in range(len(trials)) if lengths[i] < window_size]

if short:
    print(f"  ⚠ {len(short)} trials SHORTER than {args.window_sec}s window:")
    print(f"  {'Idx':>5}  {'Length':>8}  {'Seconds':>8}  {'Subject':>8}  {'Emotion'}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
    for idx, length, subj, emotion in short:
        print(f"  {idx:>5}  {length:>8}  {length/FS:>7.1f}s  {subj:>8}  {emotion}")
    print(f"\n  These trials will be SKIPPED (cannot produce a full {args.window_sec}s window).")
else:
    print(f"  ✓ All {len(trials)} trials are >= {args.window_sec}s. No trials will be skipped.")

# Length distribution in buckets
print(f"\n  Length distribution (seconds):")
buckets = [0, 5, 10, 15, 30, 60, 120, 300, 600, 9999]
for i in range(len(buckets) - 1):
    lo_s, hi_s = buckets[i], buckets[i+1]
    lo, hi = lo_s * FS, hi_s * FS
    count = np.sum((lengths >= lo) & (lengths < hi))
    if count > 0:
        label = f"  {lo_s:>4}-{hi_s:>4}s" if hi_s < 9999 else f"  {lo_s:>4}s+   "
        bar = '█' * int(count * 40 / len(trials))
        print(f"  {label}: {count:>4} trials ({100*count/len(trials):>5.1f}%)  {bar}")

# Windows per trial
n_windows = [max(0, l // window_size) for l in lengths]
print(f"\n  Windows per trial (at {args.window_sec}s):")
print(f"    Total windows possible: {sum(n_windows)}")
print(f"    Trials producing 0 windows: {sum(1 for w in n_windows if w == 0)}")
print(f"    Mean windows/trial: {np.mean(n_windows):.1f}")
print(f"    Max windows/trial: {max(n_windows)} ({max(lengths)/FS:.0f}s trial)")
