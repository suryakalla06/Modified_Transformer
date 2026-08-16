"""
Seed-variance analysis for the bAbI ΔW experiments.

Why this exists
---------------
The headline numbers in the READMEs are 5-seed means, and several of the
largest apparent wins sit on top of a standard deviation that is comparable to
the effect itself. Mean → ΔW gains +8.14 pp on qa1, but with sd = 9.93 pp; CLS
→ ΔW gains +4.44 pp on qa15 with sd = 12.99 pp. Reported as a bare mean, both
read as results. Reported with their spread, neither is yet distinguishable
from the choice of random seed.

This script does that second reading mechanically, so no one has to eyeball it.
For every (task, variant-pair) it reports the observed difference together with
a paired bootstrap interval over seeds, and labels the comparison SEPARATED
only when that interval excludes zero.

A caveat the output repeats, and which you should not skip: with n = 5 seeds
the interval is itself estimated from five numbers. It is a guard against
over-claiming, not a significance test. "NOT SEPARATED" means the experiment
cannot tell the two apart — not that they are equal.

Usage
-----
    python analysis/seed_variance.py runs/standard_vs_posdelta/comparison_results.csv

Input is the per-seed CSV written by standard_vs_posdelta.py, with columns:
    task, variant, seed, accuracy
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

# Fixed so two runs of this script on the same CSV agree. The bootstrap is a
# reporting tool; it must not itself become a source of run-to-run variation.
BOOTSTRAP_SEED = 0
N_RESAMPLES = 10_000
CI = 95.0


def load(path):
    """-> {task: {variant: {seed: accuracy}}}"""
    table = defaultdict(lambda: defaultdict(dict))

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            table[row["task"]][row["variant"]][int(row["seed"])] = float(
                row["accuracy"]
            )

    if not table:
        sys.exit(f"no rows read from {path}")

    return table


def paired_bootstrap(a, b, rng):
    """
    Resample SEEDS, not runs.

    Both variants are evaluated on the same seed set, so a seed that happens to
    be easy inflates both. Resampling the seed index preserves that pairing;
    resampling each variant independently would discard it and widen the
    interval for no reason.
    """
    n = len(a)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)

    half = (100.0 - CI) / 2.0
    return np.percentile(diffs, [half, 100.0 - half])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="per-seed results CSV")
    ap.add_argument("--baseline", default=None,
                    help="variant to compare others against "
                         "(default: the alphabetically first variant)")
    args = ap.parse_args()

    table = load(args.csv_path)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    print(f"\nPer-variant accuracy ({CI:.0f}% paired-bootstrap intervals, "
          f"{N_RESAMPLES:,} resamples)\n")
    print(f"{'task':<14}{'variant':<14}{'mean':>8}{'sd':>8}{'n':>4}")
    print("-" * 48)

    for task in sorted(table):
        for variant in sorted(table[task]):
            vals = np.array(list(table[task][variant].values()))
            print(f"{task:<14}{variant:<14}{vals.mean():>8.2f}"
                  f"{vals.std(ddof=0):>8.2f}{len(vals):>4}")

    print(f"\n\nPairwise differences vs baseline\n")
    print(f"{'task':<14}{'comparison':<26}{'diff':>8}{'  ' + str(int(CI)) + '% interval':>20}   verdict")
    print("-" * 82)

    n_separated = 0
    n_total = 0

    for task in sorted(table):
        variants = sorted(table[task])
        base = args.baseline or variants[0]

        if base not in variants:
            print(f"{task:<14}baseline '{base}' absent — skipped")
            continue

        # Compare only on seeds both variants actually completed.
        for variant in variants:
            if variant == base:
                continue

            shared = sorted(set(table[task][base]) & set(table[task][variant]))
            if len(shared) < 2:
                print(f"{task:<14}{variant + ' vs ' + base:<26}"
                      f"{'—':>8}{'':>20}   too few shared seeds")
                continue

            a = np.array([table[task][variant][s] for s in shared])
            b = np.array([table[task][base][s] for s in shared])

            diff = a.mean() - b.mean()
            lo, hi = paired_bootstrap(a, b, rng)

            separated = lo > 0 or hi < 0
            n_separated += separated
            n_total += 1

            verdict = "SEPARATED" if separated else "not separated"
            print(f"{task:<14}{variant + ' vs ' + base:<26}"
                  f"{diff:>+8.2f}   [{lo:>+6.2f}, {hi:>+6.2f}]   {verdict}")

    print("-" * 82)
    print(f"\n{n_separated}/{n_total} comparisons separated from seed noise.")

    if n_total and n_separated < n_total:
        print(
            "\nIntervals are estimated from the seed set alone. With 5 seeds this\n"
            "is a guard against over-claiming, not a significance test: "
            "'not separated'\nmeans the experiment cannot distinguish the "
            "variants, not that they match."
        )


if __name__ == "__main__":
    main()
