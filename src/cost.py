"""Exact sample-complexity computations.

Two cost measures are reported throughout the paper (Section 5.1):

  psucc_cost = 1 / p_succ            the proxy used in the prior literature
  var_cost   = E[Y^2] / E[Y]^2       the number of instances actually needed

Theorem 2.2 relates them by  var_cost = U * psucc_cost  with U >= 1, and U = 1
exactly for the prediction-free baseline.
"""
import math
import numpy as np
import networkx as nx

from .graphs import prep, edge_key, edge_list
from .patterns import PATTERNS, canonical


def per_copy_probabilities(adj, deg, m, pattern, weights):
    """Per-copy discovery probability of the full-path weighted sampler.

    ``weights`` maps an undirected edge to w_hat(e); a missing edge has
    w_hat = 0.  ``weights = {}`` is the constant predictor, which by
    Proposition 4.1 is exactly LocalCount.  ``weights = None`` is the
    prediction-free FGP baseline.
    """
    n = len(adj); twom = 2.0 * m
    P = PATTERNS[pattern]
    copies = list(P["enum"](adj, deg))
    nH = len(copies)
    Pu = twom ** (-P["rho"])
    if nH == 0:
        return copies, None, Pu, 0

    if weights is None:
        return copies, np.full(nH, Pu), Pu, nH

    edges = edge_list(adj)
    W = sum(weights.get(e, 0.0) + 1.0 for e in edges)
    wsum = np.zeros(n)
    for u in range(n):
        wsum[u] = sum(weights.get(edge_key(u, v), 0.0) + 1.0 for v in adj[u])

    Pw = np.empty(nH)
    for i, c in enumerate(copies):
        can = canonical(c, pattern, deg)
        p = 1.0
        for e in can["base"]:                        # directed edge: 1/(2m) when uniform
            p *= (weights.get(edge_key(*e), 0.0) + 1.0) / (2.0 * W)
        for (piv, close) in can["cyc"]:
            wc = weights.get(edge_key(piv, close), 0.0) + 1.0
            p *= wc / wsum[piv] if wsum[piv] > 0 else 0.0
        Pw[i] = p
    return copies, Pw, Pu, nH


def costs(G, pattern, weights):
    """Return dict(psucc_cost, var_cost, U, nH, m)."""
    adj, deg = prep(G)
    m = int(deg.sum() // 2)
    copies, Pw, Pu, nH = per_copy_probabilities(adj, deg, m, pattern, weights)
    if nH == 0:
        return None
    psucc = float(Pw.sum())
    EY = nH * Pu
    EY2 = float(np.sum(np.where(Pw > 0, Pu * Pu / np.maximum(Pw, 1e-300), 0.0)))
    var_cost = EY2 / EY ** 2
    return dict(m=m, nH=nH, psucc_cost=1.0 / psucc, var_cost=var_cost,
                U=var_cost * psucc)


def baseline_cost(G, pattern):
    return costs(G, pattern, None)


def localcount_cost(G, pattern, two_branch=True):
    """Cost of Algorithm 3.1 computed from the closed form of Theorem 3.4.

    With ``two_branch`` the high-degree branch of the FGP sampler is handled
    exactly, which is what makes L(C) <= 1 hold pointwise (Lemma 3.2).  With
    ``two_branch=False`` only the low-degree branch is used; that understates
    the speedup on graphs with high-degree pivots and can produce L > 1.
    """
    adj, deg = prep(G)
    m = int(deg.sum() // 2); twom = 2.0 * m; s = math.sqrt(twom)
    P = PATTERNS[pattern]
    L, nhigh, nH = [], 0, 0
    for c in P["enum"](adj, deg):
        nH += 1
        val = 1.0
        for (piv, close) in canonical(c, pattern, deg)["cyc"]:
            dp = float(deg[piv])
            if dp <= s or not two_branch:
                val *= dp / s
            else:
                nhigh += 1
                val *= s / float(deg[close])
        L.append(val)
    if nH == 0:
        return None
    L = np.array(L)
    Lbar = float(L.mean())
    k_fgp = twom ** P["rho"] / nH
    return dict(m=m, nH=nH, Lbar=Lbar, maxL=float(L.max()),
                D_H=(twom ** (P["alpha"] / 2.0)) * Lbar,
                high_frac=nhigh / nH,
                k_fgp=k_fgp, k_local=Lbar * k_fgp, speedup=1.0 / Lbar,
                kappa=max(nx.core_number(G).values()))


def streaming_costs(G, pattern, weights, two_pass_exact=False):
    """Same as ``costs`` but never materialises the list of copies.

    Needed for patterns whose copy count runs into tens of millions: the
    eleven-network C5 study has up to 7e7 five-cycles, where holding the
    copies in memory is the binding constraint rather than time.

    ``weights`` may be a dict as in ``costs``, or the string ``"exact"``,
    in which case the per-edge canonical-path counts are accumulated in a
    first pass and the moments computed in a second.
    """
    adj, deg = prep(G)
    m = int(deg.sum() // 2); twom = 2.0 * m; n = len(adj)
    P = PATTERNS[pattern]
    Pu = twom ** (-P["rho"])

    if weights == "exact":                       # pass 1: accumulate w(e)
        w = {}
        for c in P["enum"](adj, deg):
            can = canonical(c, pattern, deg)
            for e in can["base"]:
                k = edge_key(*e); w[k] = w.get(k, 0.0) + 1.0
            for (piv, close) in can["cyc"]:
                k = edge_key(piv, close); w[k] = w.get(k, 0.0) + 1.0
        weights = w

    if weights is not None:
        W = sum(weights.get(e, 0.0) + 1.0 for e in edge_list(adj))
        wsum = np.zeros(n)
        for u in range(n):
            wsum[u] = sum(weights.get(edge_key(u, v), 0.0) + 1.0 for v in adj[u])

    nH = 0; psucc = 0.0; EY2 = 0.0
    for c in P["enum"](adj, deg):                # pass 2: moments, streaming
        nH += 1
        if weights is None:
            p = Pu
        else:
            can = canonical(c, pattern, deg)
            p = 1.0
            for e in can["base"]:
                p *= (weights.get(edge_key(*e), 0.0) + 1.0) / (2.0 * W)
            for (piv, close) in can["cyc"]:
                wc = weights.get(edge_key(piv, close), 0.0) + 1.0
                p *= wc / wsum[piv] if wsum[piv] > 0 else 0.0
        psucc += p
        if p > 0:
            EY2 += Pu * Pu / p
    if nH == 0:
        return None
    EY = nH * Pu
    var_cost = EY2 / EY ** 2
    return dict(m=m, nH=nH, psucc_cost=1.0 / psucc, var_cost=var_cost,
                U=var_cost * psucc)
