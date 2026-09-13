"""Table 5: the three-way decomposition (localization / distributional /
informational) on eleven SNAP networks.  ~10 min + downloads."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from localcount.graphs import load_snap, SNAP_URLS
from localcount.cost import costs, localcount_cost
from localcount.predictors import (exact_weights, constant_weights,
                                   permuted_weights, mindeg_weights)

N_PERM = 5
hdr = (f"{'graph':>13}{'D_H/kap':>9}{'constant':>10}{'permuted':>10}{'exact':>9}"
       f"{'mindeg':>9}{'loc%':>8}{'dist%':>8}{'info%':>8}{'sec':>6}")
print(hdr); print("-" * len(hdr))
rows = []
for name in SNAP_URLS:
    try:
        t0 = time.time(); G = load_snap(name)
        rng = np.random.default_rng(0)
        base = costs(G, "K3", None)["var_cost"]
        w = exact_weights(G, "K3")
        S = lambda wt: base / costs(G, "K3", wt)["var_cost"]
        s_const = S(constant_weights()); s_exact = S(w); s_md = S(mindeg_weights(G, w))
        s_perm = float(np.mean([S(permuted_weights(G, w, rng)) for _ in range(N_PERM)]))
        g = s_exact - 1.0
        lc = localcount_cost(G, "K3")
        rows.append(dict(name=name, loc=(s_const-1)/g, dist=(s_perm-s_const)/g,
                         info=(s_exact-s_perm)/g, const=s_const, perm=s_perm,
                         exact=s_exact, mindeg=s_md))
        print(f"{name:>13}{lc['D_H']/lc['kappa']:>9.2f}{s_const:>10.2f}{s_perm:>10.2f}"
              f"{s_exact:>9.2f}{s_md:>9.2f}{(s_const-1)/g:>8.1%}{(s_perm-s_const)/g:>8.1%}"
              f"{(s_exact-s_perm)/g:>8.1%}{time.time()-t0:>6.0f}")
    except Exception as e:
        print(f"{name:>13}   FAILED: {type(e).__name__}: {e}")

if rows:
    for k in ["loc", "dist", "info"]:
        v = np.array([r[k] for r in rows])
        print(f"  {k+'%':>6} mean {v.mean():7.1%}  range [{v.min():.1%}, {v.max():.1%}]")
    md = sum(1 for r in rows if r["mindeg"] >= r["exact"])
    print(f"  min-degree matches or beats exact on {md} of {len(rows)} networks")
