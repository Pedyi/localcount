"""Literal Monte-Carlo implementation of LocalCount (Algorithm 3.1).

No closed form is used anywhere in this file.  It exists to check the
derivation of Theorem 3.4 rather than to re-evaluate it: the analytic code in
cost.py evaluates the same formulas the theorem derives, so it cannot detect
an error in the derivation itself.
"""
import math
import numpy as np

from .graphs import prep, edge_list
from .patterns import PATTERNS, canonical


def _directed_edge(edges, rng):
    """A uniformly random directed edge: probability 1/(2m) each."""
    e = edges[rng.integers(0, len(edges))]
    return e if rng.integers(0, 2) == 0 else (e[1], e[0])


def run(G, pattern, n_instances, seed=0):
    """Return the array of per-instance outputs Y (0 when no copy is found)."""
    rng = np.random.default_rng(seed)
    adj, deg = prep(G)
    m = int(deg.sum() // 2); twom = 2.0 * m
    edges = edge_list(adj)
    nbrs = {v: sorted(adj[v]) for v in range(len(adj))}
    Y = np.zeros(n_instances)

    for t in range(n_instances):
        if pattern == "K3":
            x, y = _directed_edge(edges, rng)          # base edge
            Nx = nbrs[x]
            w = Nx[rng.integers(0, len(Nx))]           # LOCAL draw, 1/deg(x)
            if w in (x, y) or w not in adj[y]:
                continue
            can = canonical((x, y, w), "K3", deg)
            if can["base"] != [(x, y)] or can["cyc"] != [(x, w)]:
                continue
            Y[t] = deg[x] / math.sqrt(twom)            # likelihood ratio L(C)
        elif pattern == "C5":
            p, q = _directed_edge(edges, rng)
            r, s = _directed_edge(edges, rng)
            if len({p, q, r, s}) != 4 or r not in adj[q]:
                continue
            Np = nbrs[p]
            w = Np[rng.integers(0, len(Np))]
            if w in (p, q, r, s) or w not in adj[s]:
                continue
            can = canonical((p, q, r, s, w), "C5", deg)
            if can["base"] != [(p, q), (r, s)] or can["cyc"] != [(p, w)]:
                continue
            Y[t] = deg[p] / math.sqrt(twom)
        elif pattern == "K5":
            # one base edge for the odd cycle, one for the star: this is the
            # only pattern we simulate whose decomposition mixes the two.
            p_, q = _directed_edge(edges, rng)
            r, s_ = _directed_edge(edges, rng)
            if len({p_, q, r, s_}) != 4:
                continue
            Np = nbrs[p_]
            w = Np[rng.integers(0, len(Np))]
            if w in (p_, q, r, s_):
                continue
            V = (p_, q, w, r, s_)
            if any(y not in adj[x] for i, x in enumerate(V) for y in V[i+1:]):
                continue                                   # not a K5
            can = canonical(V, "K5", deg)
            if can["base"] != [(p_, q), (r, s_)] or can["cyc"] != [(p_, w)]:
                continue
            Y[t] = deg[p_] / math.sqrt(twom)
        else:
            raise ValueError("Monte-Carlo is implemented for the alpha=1 "
                             "patterns K3, C5 and K5; alpha=0 patterns "
                             "coincide with the baseline (Proposition 3.7).")
    return Y


def check(G, pattern, n_instances, seed=0):
    """Compare the simulation against Theorem 3.4."""
    from .cost import localcount_cost
    adj, deg = prep(G)
    m = int(deg.sum() // 2); twom = 2.0 * m
    P = PATTERNS[pattern]
    Pu = twom ** (-P["rho"])
    ref = localcount_cost(G, pattern)
    Y = run(G, pattern, n_instances, seed)
    EY2_emp = float((Y ** 2).mean())
    EY2_the = twom ** (-P["bH"] - P["alpha"]) * ref["nH"] * ref["D_H"]
    k_emp = EY2_emp / (Y.mean() ** 2) if Y.mean() > 0 else float("nan")
    return dict(m=m, nH=ref["nH"], D_H=ref["D_H"], hits=int((Y > 0).sum()),
                unbiased_ratio=float(Y.mean() / Pu / ref["nH"]),
                second_moment_ratio=EY2_emp / EY2_the,
                k_ratio=k_emp / ref["k_local"])
