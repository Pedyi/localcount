"""Table 4: LocalCount on eleven SNAP networks, no predictor.
maxL must be <= 1 everywhere (Lemma 3.2).  ~3 min + downloads."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from localcount.graphs import load_snap, SNAP_URLS
from localcount.cost import localcount_cost

hdr = (f"{'graph':>13}{'m':>9}{'#K3':>10}{'sqrt2m':>8}{'kap':>5}{'high%':>8}"
       f"{'D_1branch':>11}{'D_H':>8}{'D_H/kap':>9}{'speedup':>10}{'maxL':>7}{'sec':>6}")
print(hdr); print("-" * len(hdr))
rows = []
for name in SNAP_URLS:
    try:
        t0 = time.time(); G = load_snap(name)
        r2 = localcount_cost(G, "K3", two_branch=True)
        r1 = localcount_cost(G, "K3", two_branch=False)
        r2["name"] = name; rows.append(r2)
        print(f"{name:>13}{r2['m']:>9}{r2['nH']:>10}{np.sqrt(2*r2['m']):>8.0f}"
              f"{r2['kappa']:>5}{r2['high_frac']:>8.2%}{r1['D_H']:>11.1f}{r2['D_H']:>8.1f}"
              f"{r2['D_H']/r2['kappa']:>9.2f}{r2['speedup']:>10.2f}{r2['maxL']:>7.3f}"
              f"{time.time()-t0:>6.0f}")
    except Exception as e:
        print(f"{name:>13}   FAILED: {type(e).__name__}: {e}")

bad = [r["name"] for r in rows if r["maxL"] > 1 + 1e-9]
print("\nmaxL <= 1 on all graphs:", "YES" if not bad else f"NO -> {bad}")
if rows:
    s = np.array([r["speedup"] for r in rows]); dk = np.array([r["D_H"]/r["kappa"] for r in rows])
    print(f"  speedup  mean {s.mean():.1f}x  range [{s.min():.1f}x, {s.max():.1f}x]")
    print(f"  D_H/kappa mean {dk.mean():.2f}  range [{dk.min():.2f}, {dk.max():.2f}]")
