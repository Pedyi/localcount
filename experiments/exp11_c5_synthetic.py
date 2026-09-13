"""Synthetic control for the C5 finding (Table 2).

On the four real networks the exact predictor is worse than no predictor for
C5.  This script asks whether synthetic families reproduce that, and finds
they do not: on preferential-attachment-style graphs the exact predictor still
helps by a wide margin, and on dense G(n,p) the ratio only approaches 1 from
above and dips just below it.  The reversal is therefore a property of those
real networks rather than of the pattern C5 as such, and the manuscript says
so.  ~5 min, no network."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import networkx as nx
from localcount.cost import localcount_cost, streaming_costs

CASES = [("PLC(3000, 3, 0.5)", lambda: nx.powerlaw_cluster_graph(3000, 3, 0.5, seed=1)),
         ("PLC(5000, 3, 0.5)", lambda: nx.powerlaw_cluster_graph(5000, 3, 0.5, seed=1)),
         ("PLC(5000, 5, 0.5)", lambda: nx.powerlaw_cluster_graph(5000, 5, 0.5, seed=1)),
         ("PLC(4000, 8, 0.5)", lambda: nx.powerlaw_cluster_graph(4000, 8, 0.5, seed=1)),
         ("G(200, 0.10)",      lambda: nx.gnp_random_graph(200, 0.10, seed=1)),
         ("G(250, 0.12)",      lambda: nx.gnp_random_graph(250, 0.12, seed=1))]

hdr = (f"{'family':>20}{'m':>8}{'#C5':>11}{'#C5/m':>8}{'kappa':>7}{'D_H/kap':>9}"
       f"{'LocalCount':>12}{'exact pred':>12}{'exact/LC':>10}")
print(hdr); print("-" * len(hdr))
for name, make in CASES:
    G = make()
    r = localcount_cost(G, "C5")
    if r is None:
        print(f"{name:>20}   no 5-cycles"); continue
    base = streaming_costs(G, "C5", None)["var_cost"]
    ex = base / streaming_costs(G, "C5", "exact")["var_cost"]
    print(f"{name:>20}{r['m']:>8}{r['nH']:>11}{r['nH']/r['m']:>8.0f}{r['kappa']:>7}"
          f"{r['D_H']/r['kappa']:>9.2f}{r['speedup']:>12.2f}{ex:>12.2f}"
          f"{ex/r['speedup']:>10.2f}")
print("\nexact/LC < 1 means the predictor costs more than using none.")
