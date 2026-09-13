"""Table 3 (space bound) + growth exponents.  Verifies Theorem 3.4 and
Proposition 3.7: patterns with alpha = 0 gain exactly 1.  Runtime: ~1 min."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from localcount.graphs import gen
from localcount.cost import localcount_cost
from localcount.patterns import PATTERNS

JOBS = [("K3", "sparse", [400, 800, 1600, 3200]),
        ("C5", "ba",     [200, 400, 800]),
        ("K5", "cliquey",[400, 800, 1600, 3200]),   # decomposition = C3 + S1
        ("C4", "sparse", [400, 800, 1600]),
        ("K4", "cliquey",[400, 800, 1600])]

rows = []
hdr = f"{'H':>4}{'family':>9}{'m':>8}{'#H':>9}{'a':>3}{'D_H':>8}{'k_FGP':>13}{'k_LC':>13}{'speedup':>10}{'predicted':>11}"
print(hdr); print("-" * len(hdr))
for pat, fam, sizes in JOBS:
    for n in sizes:
        G = gen(fam, n, 1)
        r = localcount_cost(G, pat)
        if r is None: continue
        P = PATTERNS[pat]
        pred = (2.0 * r["m"]) ** (P["alpha"] / 2.0) / r["D_H"]
        r.update(pattern=pat); rows.append(r)
        print(f"{pat:>4}{fam:>9}{r['m']:>8}{r['nH']:>9}{P['alpha']:>3}{r['D_H']:>8.2f}"
              f"{r['k_fgp']:>13.1f}{r['k_local']:>13.2f}{r['speedup']:>10.4f}{pred:>11.4f}")
    print()

print(f"{'H':>4}{'measured b':>13}{'D_H ~ m^a':>12}{'predicted b':>13}")
print("-" * 42)
for pat in ["K3", "C5", "K5", "C4", "K4"]:
    sub = [r for r in rows if r["pattern"] == pat]
    if len(sub) < 2: continue
    ms = np.log([r["m"] for r in sub])
    b = np.polyfit(ms, np.log([r["speedup"] for r in sub]), 1)[0]
    a = np.polyfit(ms, np.log([r["D_H"] for r in sub]), 1)[0]
    print(f"{pat:>4}{b:>13.3f}{a:>12.3f}{PATTERNS[pat]['alpha']/2.0 - a:>13.3f}")
