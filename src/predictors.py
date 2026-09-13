"""The weightings compared in the paper.

  constant  : w_hat = 0 everywhere.  No information, no distributional shape.
              By Proposition 4.1 this is exactly LocalCount.
  exact     : w(e) = number of copies whose canonical path uses e.
  permuted  : the exact (or learned) multiset, reassigned uniformly at random.
  mindeg    : the same multiset, reassigned by rank of min(deg u, deg v).
  learned   : ridge regression on eight degree/core features, trained on a
              different graph (strict leave-one-graph-out).
  tempered  : (w_hat + 1) = (w + 1)**theta, the family of Section 4.
"""
import numpy as np
import networkx as nx

from .graphs import prep, edge_key, edge_list
from .patterns import PATTERNS, canonical


def exact_weights(G, pattern):
    adj, deg = prep(G)
    w = {}
    for c in PATTERNS[pattern]["enum"](adj, deg):
        can = canonical(c, pattern, deg)
        for e in can["base"]:
            k = edge_key(*e); w[k] = w.get(k, 0.0) + 1.0
        for (piv, close) in can["cyc"]:
            k = edge_key(piv, close); w[k] = w.get(k, 0.0) + 1.0
    return w


def triangle_counts(G):
    """Exact per-edge triangle counts t(e); the regression target."""
    adj, deg = prep(G)
    t = {}
    for (u, a, b) in PATTERNS["K3"]["enum"](adj, deg):
        for e in (edge_key(u, a), edge_key(u, b), edge_key(a, b)):
            t[e] = t.get(e, 0.0) + 1.0
    return t


def constant_weights():
    return {}


def permuted_weights(G, base, rng):
    adj, _ = prep(G)
    edges = edge_list(adj)
    vals = np.array([base.get(e, 0.0) for e in edges])
    return dict(zip(edges, rng.permutation(vals)))


def rank_matched_weights(G, base, key_fn):
    """Same weight multiset, reassigned by the rank of ``key_fn(u, v)``."""
    adj, deg = prep(G)
    edges = edge_list(adj)
    vals = np.sort(np.array([base.get(e, 0.0) for e in edges]))
    order = np.argsort(np.array([key_fn(deg[u], deg[v]) for (u, v) in edges]),
                       kind="stable")
    return {edges[i]: vals[r] for r, i in enumerate(order)}


def mindeg_weights(G, base):
    return rank_matched_weights(G, base, lambda du, dv: min(du, dv))


def tempered_weights(base, edges, theta):
    return {e: (base.get(e, 0.0) + 1.0) ** theta - 1.0 for e in edges}


def _features(edges, deg, core):
    L = np.log1p
    X = []
    for (u, v) in edges:
        du, dv = float(deg[u]), float(deg[v])
        cu, cv = float(core[u]), float(core[v])
        X.append([L(min(du, dv)), L(max(du, dv)), L(du + dv), L(abs(du - dv)),
                  L(min(cu, cv)), L(max(cu, cv)), L(cu + cv), 1.0])
    return np.array(X)


def _core_numbers(adj):
    H = nx.Graph(edge_list(adj))
    H.add_nodes_from(range(len(adj)))
    return nx.core_number(H)


def learned_weights(G_train, G_test, lam=1.0):
    """Ridge on log(1+t(e)); train on one graph, predict on another.

    Returns (weights, correlation of predicted vs true log-heaviness).
    """
    aT, dT = prep(G_train); eT = edge_list(aT)
    a, d = prep(G_test);    e = edge_list(a)
    tT, t = triangle_counts(G_train), triangle_counts(G_test)
    X = _features(eT, dT, _core_numbers(aT))
    y = np.log1p([tT.get(x, 0.0) for x in eT])
    beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    yhat = _features(e, d, _core_numbers(a)) @ beta
    ytrue = np.log1p([t.get(x, 0.0) for x in e])
    corr = float(np.corrcoef(yhat, ytrue)[0, 1])
    return {x: max(float(np.expm1(yhat[i])), 0.0) for i, x in enumerate(e)}, corr
