"""Graph families and SNAP loaders."""
import gzip, itertools, urllib.request
import networkx as nx
import numpy as np

SNAP_URLS = {
    "ca-GrQc":      "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "oregon1":      "https://snap.stanford.edu/data/oregon1_010331.txt.gz",
    "ca-HepTh":     "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    "as-caida":     "https://snap.stanford.edu/data/as-caida20071105.txt.gz",
    "facebook":     "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "ca-CondMat":   "https://snap.stanford.edu/data/ca-CondMat.txt.gz",
    "email-Enron":  "https://snap.stanford.edu/data/email-Enron.txt.gz",
    "ca-AstroPh":   "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
    "email-EuAll":  "https://snap.stanford.edu/data/email-EuAll.txt.gz",
    "soc-Slashdot": "https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz",
    "amazon0302":   "https://snap.stanford.edu/data/amazon0302.txt.gz",
}

_cache = {}


def load_snap(name, url=None, verbose=True):
    """Download (once per session) and parse a SNAP edge list."""
    if name in _cache:
        return _cache[name]
    url = url or SNAP_URLS[name]
    if verbose:
        print(f"downloading {name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    txt = gzip.decompress(urllib.request.urlopen(req, timeout=300).read()).decode("utf-8", "ignore")
    G = nx.Graph()
    for line in txt.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 2 and p[0] != p[1]:
            G.add_edge(int(p[0]), int(p[1]))
    G.remove_edges_from(nx.selfloop_edges(G))
    _cache[name] = G
    return G


def gen(family, n, seed=1):
    """Synthetic families used in the paper.

    sparse   : G(n, 3n); constant average degree, so D_H does not grow with m
    ba       : Barabasi-Albert(n, 4)
    skewed   : friendship graph
    separate : the separation instance of [PredCount26, Thm 11] --
               friendship graph beside a dense triangle-free K_{d,d} decoy
    cliquey  : sparse graph with planted 5-cliques, so K_4 copies exist
    dense    : G(n, 0.3)
    """
    if family == "sparse":
        return nx.gnm_random_graph(n, 3 * n, seed=seed)
    if family == "ba":
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    if family == "skewed":
        G = nx.Graph(); node = 1
        for _ in range(n // 2):
            a, b = node, node + 1
            G.add_edge(0, a); G.add_edge(0, b); G.add_edge(a, b); node += 2
        return G
    if family == "separate":
        G = nx.Graph(); node = 1
        s = max(100, n // 8)
        for _ in range(s):
            a, b = node, node + 1
            G.add_edge(0, a); G.add_edge(0, b); G.add_edge(a, b); node += 2
        d = int(s ** 0.5)
        L = list(range(node, node + d)); R = list(range(node + d, node + 2 * d))
        for u in L:
            for v in R:
                G.add_edge(u, v)
        return G
    if family == "cliquey":
        G = nx.gnm_random_graph(n, 2 * n, seed=seed)
        for i in range(0, min(n, 400), 10):
            for a, b in itertools.combinations(range(i, min(i + 5, n)), 2):
                G.add_edge(a, b)
        return G
    if family == "dense":
        return nx.gnp_random_graph(n, 0.3, seed=seed)
    if family == "dense6":
        # G(n, 0.6): dense enough that K5 copies are plentiful, which is what a
        # Monte-Carlo check of the K5 path needs.  The speedup here is small
        # (D_H is close to its ceiling sqrt(2m)); the point is the identity,
        # not the gain.
        return nx.gnp_random_graph(n, 0.6, seed=seed)
    if family == "book":
        # K_{d,d} plus a perfect matching on the left side, d = n.
        # m = d^2 + d/2, #K3 = d^2/2 = Theta(m), and every triangle's
        # minimum-degree vertex is on the right, so D_H = d = Theta(sqrt m).
        # This is the family on which the bound of Theorem 3.4 is tight.
        d = n
        G = nx.Graph()
        L, R = list(range(d)), list(range(d, 2 * d))
        for u in L:
            for v in R:
                G.add_edge(u, v)
        for i in range(0, d - 1, 2):
            G.add_edge(L[i], L[i + 1])
        return G
    raise ValueError(family)


def prep(G):
    """Integer-indexed adjacency sets plus a degree array."""
    idx = {v: i for i, v in enumerate(G.nodes())}
    adj = [set() for _ in idx]
    for u, v in G.edges():
        iu, iv = idx[u], idx[v]
        if iu != iv:
            adj[iu].add(iv); adj[iv].add(iu)
    return adj, np.array([len(a) for a in adj], dtype=np.int64)


def edge_key(u, v):
    return (u, v) if u < v else (v, u)


def edge_list(adj):
    return [(u, v) for u in range(len(adj)) for v in adj[u] if u < v]
