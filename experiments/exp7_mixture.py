"""Table 1 and Section 5.7: the mixture ceiling and inverse-variance pooling.
Runs on synthetic graphs only, so it needs no network access.  ~1 min."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from localcount.graphs import gen, prep, edge_list
from localcount.pooling import branch_moments, budgets
from localcount.predictors import exact_weights, constant_weights, permuted_weights

G = gen("ba", 2000, 1)
adj, deg = prep(G); edges = edge_list(adj)
rng = np.random.default_rng(0)
w = exact_weights(G, "K3")
vals = np.array([w.get(e, 0.0) for e in edges])
adversarial = {e: vals.max() - w.get(e, 0.0) for e in edges}

print(f"m = {G.number_of_edges()}, lambda = 1/2\n")
hdr = (f"{'predictor':>13}{'uniform':>11}{'weighted':>11}{'plain mean':>12}"
       f"{'pooled':>11}{'plain gain':>12}{'pooled gain':>13}")
print(hdr); print("-" * len(hdr))
base = None
for name, wt in [("constant", constant_weights()), ("exact", w),
                 ("permuted", permuted_weights(G, w, rng)),
                 ("adversarial", adversarial)]:
    mu, Vu, Vw = branch_moments(G, "K3", wt)
    b = budgets(mu, Vu, Vw)
    if base is None: base = b["uniform_only"]
    print(f"{name:>13}{b['uniform_only']:>11.1f}{b['weighted_only']:>11.1f}"
          f"{b['plain_mixture']:>12.1f}{b['inverse_variance']:>11.1f}"
          f"{base/b['plain_mixture']:>12.2f}{base/b['inverse_variance']:>13.2f}")
    assert b["inverse_variance"] <= 2.0 * min(b["uniform_only"], b["weighted_only"]) + 1e-9

print("\nlambda sweep, exact predictor (Theorem 2.6: gain <= 1/lambda)")
mu, Vu, Vw = branch_moments(G, "K3", w)
print(f"{'lambda':>10}{'plain gain':>13}{'1/lambda':>11}")
for lam in [0.5, 0.25, 0.1, 0.05, 0.02]:
    b = budgets(mu, Vu, Vw, lam)
    print(f"{lam:>10.2f}{b['uniform_only']/b['plain_mixture']:>13.2f}{1/lam:>11.2f}")
b0 = budgets(mu, Vu, Vw, 0.0)
print(f"{0.0:>10.2f}{b0['uniform_only']/b0['plain_mixture']:>13.2f}{'inf':>11}")
