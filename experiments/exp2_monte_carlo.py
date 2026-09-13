"""Section 5.3: Monte-Carlo check of unbiasedness and the second moment.
All three ratios should be ~1.00; noise scales as 1/sqrt(hits).  ~10 min."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from localcount.graphs import gen
from localcount.sampler import check

JOBS = [("K3", "sparse",   400, 1_500_000),
        ("K3", "sparse",   800, 1_500_000),
        ("K3", "skewed",   600,   600_000),
        ("K3", "separate", 800,   600_000),
        ("C5", "ba",       200, 2_000_000),
        ("C5", "ba",       400, 2_000_000),
        # K5 is the only pattern here whose decomposition mixes an odd cycle
        # with a star, so it is the one that exercises the part-combining step
        # of Theorem 3.4.  It needs a dense graph to get a usable hit rate.
        ("K5", "dense6",    50, 12_000_000)]

hdr = f"{'H':>4}{'graph':>11}{'m':>8}{'#H':>9}{'hits':>9}{'E[Y]':>9}{'E[Y^2]':>10}{'k':>9}"
print(hdr); print("-" * len(hdr)); print(f"{'':>41}{'ratios, should be ~1.00'}")
for pat, fam, n, N in JOBS:
    r = check(gen(fam, n, 1), pat, N)
    print(f"{pat:>4}{fam:>11}{r['m']:>8}{r['nH']:>9}{r['hits']:>9}"
          f"{r['unbiased_ratio']:>9.4f}{r['second_moment_ratio']:>10.4f}{r['k_ratio']:>9.4f}")
