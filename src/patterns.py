"""Copy enumeration and canonical sampling paths.

Every copy of H is enumerated exactly once.  ``canonical`` returns the data the
Fichtenberger-Peng sampler uses: the base edges it draws and, for each
odd-cycle part, the (pivot, closing) pair.  The analytic cost code and the
Monte-Carlo sampler both call this function, so they cannot disagree about
what the canonical path is.
"""
import itertools


def copies_K3(adj, deg):
    for u in range(len(adj)):
        Nu = sorted(w for w in adj[u] if w > u)
        for i in range(len(Nu)):
            Aa = adj[Nu[i]]
            for j in range(i + 1, len(Nu)):
                if Nu[j] in Aa:
                    yield (u, Nu[i], Nu[j])


def copies_C4(adj, deg):
    for u in range(len(adj)):
        seen = {}
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w <= u or w == u:
                    continue
                seen.setdefault(w, []).append(v)
        for w, mids in seen.items():
            for v, x in itertools.combinations(sorted(set(mids)), 2):
                yield (u, v, w, x)


def copies_K4(adj, deg):
    for (u, a, b) in copies_K3(adj, deg):
        for c in adj[u] & adj[a] & adj[b]:
            if c > b:
                yield (u, a, b, c)


def copies_C5(adj, deg):
    for u in range(len(adj)):
        for a in adj[u]:
            if a <= u:
                continue
            for b in adj[a]:
                if b <= u or b == a:
                    continue
                for c in adj[b]:
                    if c <= u or c in (a, b):
                        continue
                    for d in adj[c]:
                        if d <= u or d in (a, b, c):
                            continue
                        if u in adj[d] and a < d:
                            yield (u, a, b, c, d)


def copies_K5(adj, deg):
    for (u, a, b, c) in copies_K4(adj, deg):
        for e in adj[u] & adj[a] & adj[b] & adj[c]:
            if e > c:
                yield (u, a, b, c, e)


def canonical(copy, pattern, deg):
    """{'base': [edges], 'cyc': [(pivot, closing)]} for the canonical path."""
    key = lambda v: (deg[v], v)
    if pattern == "K3":
        x, y, z = sorted(copy, key=key)
        return {"base": [(x, y)], "cyc": [(x, z)]}
    if pattern == "C5":
        n = len(copy)
        i = min(range(n), key=lambda t: key(copy[t]))
        fwd = tuple(copy[(i + t) % n] for t in range(n))
        bwd = (fwd[0],) + tuple(reversed(fwd[1:]))
        r = fwd if key(fwd[-1]) < key(fwd[1]) else bwd
        return {"base": [(r[0], r[1]), (r[2], r[3])], "cyc": [(r[0], r[4])]}
    if pattern == "C4":                       # two disjoint S_1 stars: alpha = 0
        u, v, w, x = copy
        return {"base": [(u, v), (w, x)], "cyc": []}
    if pattern == "K4":                       # two disjoint S_1 stars: alpha = 0
        u, a, b, c = copy
        return {"base": [(u, a), (b, c)], "cyc": []}
    if pattern == "K5":
        # rho(K5) = 5/2 = rho(C_3) + rho(S_1), so the decomposition is one odd
        # cycle plus one star.  K5 is vertex-transitive, so which three vertices
        # carry the cycle has to be fixed by convention: we take the three
        # <-smallest, which makes the cycle part agree with the K3 rule above,
        # and the two <-largest as the star.  This is the only place a choice
        # is made, and both the enumeration and the sampler read it from here.
        v = sorted(copy, key=key)
        return {"base": [(v[0], v[1]), (v[3], v[4])], "cyc": [(v[0], v[2])]}
    raise ValueError(pattern)


# rho = b_H + alpha/2 ; beta = integral edge-cover number (Proposition 3.7)
PATTERNS = {
    "K3": dict(rho=1.5, bH=1, alpha=1, beta=2, enum=copies_K3),
    "C5": dict(rho=2.5, bH=2, alpha=1, beta=3, enum=copies_C5),
    "C4": dict(rho=2.0, bH=2, alpha=0, beta=2, enum=copies_C4),
    "K4": dict(rho=2.0, bH=2, alpha=0, beta=2, enum=copies_K4),
    "K5": dict(rho=2.5, bH=2, alpha=1, beta=3, enum=copies_K5),
}
for _k, _v in PATTERNS.items():
    assert abs(_v["rho"] - (_v["bH"] + _v["alpha"] / 2)) < 1e-9, _k
    assert (_v["alpha"] == 0) == (abs(_v["rho"] - _v["beta"]) < 1e-9), _k
