"""Table 6: the permutation control, located.

Runs the leave-one-graph-out transfer study under BOTH cost measures and adds
the constant-weight column.  The decisive number is const/permuted, which is
>= 1 everywhere: a predictor with no distributional shape at all matches or
beats the permuted one, so what the permutation control fails to remove cannot
be distributional.  ~10 min + downloads."""
import sys, os, itertools; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from localcount.graphs import load_snap, gen
from localcount.cost import costs
from localcount.predictors import (exact_weights, constant_weights,
                                   permuted_weights, learned_weights)

# The four graphs of the transfer study in [PredCount26] are synthetic
# powerlaw-cluster instances; pass --synthetic to reproduce that setting.
SYNTHETIC = "--synthetic" in sys.argv
if SYNTHETIC:
    import networkx as nx
    GRAPHS = {"collab-A": nx.powerlaw_cluster_graph(1500, 3, 0.40, seed=1),
              "collab-B": nx.powerlaw_cluster_graph(1200, 4, 0.35, seed=2),
              "social-C": nx.powerlaw_cluster_graph(900,  8, 0.50, seed=3),
              "social-D": nx.powerlaw_cluster_graph(1100, 6, 0.45, seed=4)}
else:
    GRAPHS = {n: load_snap(n) for n in
              ["ca-GrQc", "ca-HepTh", "facebook", "oregon1"]}

N_PERM = 5
hdr = (f"{'train':>11}{'test':>11}{'corr':>7}{'measure':>9}{'exact':>9}{'learned':>9}"
       f"{'permL':>8}{'const':>8}{'struct%':>9}{'loc%':>8}")
print(hdr); print("-" * len(hdr))
rows = []
for A, B in itertools.permutations(GRAPHS, 2):
    GA, GB = GRAPHS[A], GRAPHS[B]
    wl, corr = learned_weights(GA, GB)
    we = exact_weights(GB, "K3")
    rng = np.random.default_rng(0)
    perms = [permuted_weights(GB, wl, rng) for _ in range(N_PERM)]
    for measure in ["psucc_cost", "var_cost"]:
        base = costs(GB, "K3", None)[measure]
        S = lambda wt: base / costs(GB, "K3", wt)[measure]
        e, l, c = S(we), S(wl), S(constant_weights())
        p = float(np.mean([S(x) for x in perms]))
        struct = ((l - 1) - (p - 1)) / (l - 1) if l > 1 else float("nan")
        loc = (c - 1) / (l - 1) if l > 1 else float("nan")
        rows.append(dict(measure=measure, struct=struct, loc=loc, ratio=c / p,
                         exact=e, learned=l))
        tag = "1/psucc" if measure == "psucc_cost" else "true"
        print(f"{A:>11}{B:>11}{corr:>7.2f}{tag:>9}{e:>9.2f}{l:>9.2f}{p:>8.2f}"
              f"{c:>8.2f}{struct:>9.1%}{loc:>8.1%}")

print()
for measure in ["psucc_cost", "var_cost"]:
    sub = [r for r in rows if r["measure"] == measure]
    st = np.array([r["struct"] for r in sub]); lo = np.array([r["loc"] for r in sub])
    ra = np.array([r["ratio"] for r in sub])
    wins = sum(1 for r in sub if r["learned"] > r["exact"])
    tag = "1/psucc" if measure == "psucc_cost" else "true cost"
    print(f"{tag:>10}: struct% {st.mean():6.1%}  loc% {lo.mean():6.1%}  "
          f"const/permL {ra.mean():5.3f}  learned beats exact on {wins}/{len(sub)}")
