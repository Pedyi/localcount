# LocalCount

Code for **"Localization, Not Prediction: Where the Speedup Comes From in
Streaming Subgraph Counting."**

The paper asks what the reported speedups of prediction-augmented subgraph
counters are made of, and finds that a large, identifiable part of them is not
attributable to prediction. This repository reproduces every table.

## What is here

`LocalCount` is the Fichtenberger–Peng sampler with the closing-vertex
rejection step deleted and the output corrected by a likelihood ratio. It uses
**no predictor**, asks no query the baseline does not already ask, and runs in
three passes over a turnstile stream. Its likelihood ratio is bounded by 1
pointwise, so it is never worse than the baseline on any graph.

The repository also contains the machinery for the paper's second claim: that
`1/p_succ` — the measure used to report speedups in this literature — is a
lower bound on the sample complexity rather than the sample complexity, the
gap being the non-uniformity `U` of the sampler's distribution over copies.

## Install

    pip install -r requirements.txt

Python 3.9+. Only `numpy` and `networkx` are required; nothing is compiled.
The real-graph experiments download SNAP datasets over the network on first
use and cache them in memory for the session.

## Reproducing the paper

Each script prints one table. Times are for a laptop.

| script | produces | needs network | time |
|---|---|---|---|
| `experiments/exp1_space_bound.py` | Table 8 and the growth exponents | no | ~1 min |
| `experiments/exp2_monte_carlo.py` | Section 5.3, the Monte-Carlo check | no | ~10 min |
| `experiments/exp3_real_graphs.py` | Table 1, eleven SNAP networks | yes | ~3 min |
| `experiments/exp4_decomposition.py` | Table 3, the three-way split | yes | ~10 min |
| `experiments/exp5_permutation_control.py` | Table 4, the control located | yes | ~10 min |
| `experiments/exp6_tempering.py` | Table 7, the theta sweep | yes | ~3 min |
| `experiments/exp7_mixture.py` | Table 6 and the lambda sweep | no | ~1 min |
| `experiments/exp8_separation.py` | Table 5, the separation family | no | ~1 min |
| `experiments/exp9_tightness.py` | Table 9, the bound is tight | no | ~1 min |
| `experiments/exp10_c5_real.py` | Table 2, `C5` on real networks | yes | ~45 min |
| `experiments/exp11_c5_synthetic.py` | Table 2's synthetic control | no | ~5 min |

Table numbers refer to the published manuscript. Tables 6-9 are in Appendix A.

Run everything that needs no network with:

    bash run_all_offline.sh

`exp5` also reproduces the transfer study of the prior work, whose four graphs
are synthetic `powerlaw_cluster` instances rather than real networks:

    python experiments/exp5_permutation_control.py --synthetic

## Two cost measures

Every experiment reports both, and the difference between them is one of the
paper's findings.

* `psucc_cost = 1 / p_succ` — the expected number of parallel instances before
  one discovers a copy. This is what the prior literature reports.
* `var_cost = E[Y^2] / E[Y]^2` — the number of instances a
  `(1 ± eps)`-approximation actually needs.

Theorem 2.2 gives `var_cost = U * psucc_cost` with `U >= 1`, and `U = 1`
exactly for the prediction-free baseline. So the two agree on the baseline and
diverge for every weighted sampler, which is precisely the comparison the
literature makes.

## Layout

    src/localcount/
      graphs.py      synthetic families, SNAP loaders, adjacency prep
      patterns.py    copy enumeration and canonical sampling paths
      cost.py        exact per-copy probabilities, both cost measures, U
      predictors.py  constant / exact / permuted / min-degree / learned / tempered
      sampler.py     literal Monte-Carlo LocalCount (no closed forms)
      pooling.py     plain mixture vs inverse-variance pooling
    experiments/     one script per table

`patterns.canonical` is shared by the analytic code and the Monte-Carlo
sampler, so the two cannot disagree about what the canonical path is. This
matters: `cost.py` evaluates the same formulas the theorems derive, so on its
own it could not detect an error in a derivation. `sampler.py` runs the
sampler for real and uses no closed form anywhere; `exp2` compares them.

## Notes on conventions

Base edges are drawn uniformly from the `2m` **directed** edges, so the
baseline's per-copy probability is exactly `(2m)^(-rho(H))`. This differs by a
pattern-dependent constant from the undirected convention used in the prior
work; ratios are unaffected, absolute values of `1/p_succ` differ by
`2^(b_H)`.

`localcount_cost(..., two_branch=True)` handles the high-degree branch of the
sampler exactly. This is not cosmetic: without it the likelihood ratio can
exceed 1 and the "never worse than the baseline" guarantee fails. On the
eleven SNAP networks the branch fires on 0–2.7% of triangles, but where it
fires it changes the speedup by up to 24%.

## Patterns

The improvement is `(2m)^(alpha/2) / D_H`, where `alpha` is the number of odd
cycles in the decomposition of `H`. It is exactly 1 when `rho(H) = beta(H)`,
and hence for every bipartite pattern.

| H | rho | beta | alpha | improvement |
|---|---|---|---|---|
| K3 | 3/2 | 2 | 1 | `sqrt(2m)/D_H` |
| C5 | 5/2 | 3 | 1 | `sqrt(2m)/D_H` |
| C4 | 2 | 2 | 0 | 1 |
| K4 | 2 | 2 | 0 | 1 |

## License

MIT.
