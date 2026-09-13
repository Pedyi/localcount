"""Table 6: a second pattern on real networks.

Section 5 reports K3 on eleven networks; this adds C5 on the four where exact
enumeration is feasible.  Copy counts reach 7e7, so the moments are
accumulated by streaming_costs in two passes rather than by materialising the
copies -- memory, not time, is the binding constraint here.

The headline entry is that the exact predictor is WORSE than no predictor on
all four networks, and on ca-GrQc worse than the baseline itself.  A C5 copy
is found through two base-edge draws against the global normaliser, so
weighting sharpens the per-copy distribution twice where a triangle sharpens
it once; U rises and U/p_succ loses.  ~45 min + downloads."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from localcount.graphs import load_snap
from localcount.cost import localcount_cost, streaming_costs

NAMES = ["ca-HepTh", "oregon1", "ca-GrQc", "as-caida"]   # ascending by #C5

hdr = (f"{'graph':>11}{'m':>9}{'#C5':>12}{'D_H':>8}{'kappa':>7}{'D_H/kap':>9}"
       f"{'high%':>8}{'LocalCount':>12}{'exact pred':>12}{'maxL':>7}{'min':>6}")
print(hdr); print("-" * len(hdr))
rows = []
for name in NAMES:
    try:
        t0 = time.time(); G = load_snap(name)
        r = localcount_cost(G, "C5")
        if r is None:
            print(f"{name:>11}   no 5-cycles"); continue
        base = streaming_costs(G, "C5", None)["var_cost"]
        s_exact = base / streaming_costs(G, "C5", "exact")["var_cost"]
        r["name"] = name; r["exact"] = s_exact; rows.append(r)
        print(f"{name:>11}{r['m']:>9}{r['nH']:>12}{r['D_H']:>8.1f}{r['kappa']:>7}"
              f"{r['D_H']/r['kappa']:>9.2f}{r['high_frac']:>8.2%}{r['speedup']:>12.2f}"
              f"{s_exact:>12.2f}{r['maxL']:>7.3f}{(time.time()-t0)/60:>6.1f}")
    except Exception as e:
        print(f"{name:>11}   FAILED: {type(e).__name__}: {e}")

if rows:
    bad = [r["name"] for r in rows if r["maxL"] > 1 + 1e-9]
    print("\nmaxL <= 1 on all graphs:", "YES" if not bad else f"NO -> {bad}")
    lc = np.array([r["speedup"] for r in rows]); ex = np.array([r["exact"] for r in rows])
    hurt = [r["name"] for r in rows if r["exact"] < r["speedup"]]
    print(f"  LocalCount       : mean {lc.mean():6.2f}x  range [{lc.min():.2f}x, {lc.max():.2f}x]")
    print(f"  exact predictor  : mean {ex.mean():6.2f}x  range [{ex.min():.2f}x, {ex.max():.2f}x]")
    print(f"  exact/LocalCount : mean {np.mean(ex/lc):6.2f}  (< 1 means it HURTS)")
    print(f"  predictor worse than none on {len(hurt)}/{len(rows)}: {hurt}")
