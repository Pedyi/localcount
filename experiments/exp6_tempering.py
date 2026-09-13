"""Table 2: the tempering sweep.  U is U-shaped in theta on real networks, so
the true speedup is concave with an interior maximum (Proposition 4.2).
~3 min + downloads."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from localcount.graphs import load_snap, prep, edge_list
from localcount.cost import costs
from localcount.predictors import exact_weights, tempered_weights, learned_weights

THETAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
NAMES = ["ca-GrQc", "ca-HepTh", "oregon1", "as-caida"]

def sweep(G, base):
    adj, _ = prep(G); edges = edge_list(adj)
    b_ps = costs(G, "K3", None)["psucc_cost"]; b_v = costs(G, "K3", None)["var_cost"]
    out = []
    for th in THETAS:
        c = costs(G, "K3", tempered_weights(base, edges, th))
        out.append(dict(theta=th, psucc=b_ps / c["psucc_cost"],
                        U=c["U"], true=b_v / c["var_cost"]))
    return out

for weights_kind in ["exact", "learned"]:
    print(f"\n=========== base weights: {weights_kind} ===========")
    for i, name in enumerate(NAMES):
        G = load_snap(name)
        if weights_kind == "exact":
            base = exact_weights(G, "K3"); note = ""
        else:
            train = NAMES[(i + 1) % len(NAMES)]
            base, corr = learned_weights(load_snap(train), G)
            note = f" (trained on {train}, corr {corr:.2f})"
        out = sweep(G, base)
        best = max(out, key=lambda r: r["true"])
        one = [r for r in out if r["theta"] == 1.0][0]
        print(f"--- {name}{note} ---")
        print("  theta :", " ".join(f"{r['theta']:>7.2f}" for r in out))
        print("  psucc :", " ".join(f"{r['psucc']:>7.2f}" for r in out))
        print("  U     :", " ".join(f"{r['U']:>7.3f}" for r in out))
        print("  TRUE  :", " ".join(f"{r['true']:>7.2f}" for r in out))
        print(f"  best theta = {best['theta']:.2f} -> {best['true']:.2f}x   "
              f"(theta=1: {one['true']:.2f}x, gap {best['true']/one['true']:.2f}x)")
