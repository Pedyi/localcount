"""Table 7: LocalCount on the separation family of [PredCount26, Thm 11].
The cost does not grow with m, and no predictor is used.  ~1 min."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from localcount.graphs import gen
from localcount.cost import localcount_cost

hdr = f"{'m':>9}{'#K3':>9}{'#K3/m':>8}{'kappa':>7}{'D_H':>7}{'k_FGP':>11}{'k_LocalCount':>15}{'gain':>9}"
print(hdr); print("-" * len(hdr))
for n in [500, 1000, 2000, 4000, 8000, 16000, 32000]:
    r = localcount_cost(gen("separate", n, 1), "K3")
    print(f"{r['m']:>9}{r['nH']:>9}{r['nH']/r['m']:>8.2f}{r['kappa']:>7}{r['D_H']:>7.1f}"
          f"{r['k_fgp']:>11.1f}{r['k_local']:>15.2f}{r['speedup']:>9.2f}")
