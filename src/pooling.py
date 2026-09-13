"""Combining the uniform and weighted branches (Section 2.3).

Plain mean (Theorem 2.6)  : Var = lam*Var_u + (1-lam)*Var_w
                            -> the gain can never exceed 1/lam.
Inverse-variance (Thm 2.7): 1/Var = lam/Var_u + (1-lam)/Var_w
                            -> within 1/min(lam,1-lam) of the better branch.
"""
import numpy as np

from .cost import per_copy_probabilities
from .graphs import prep
from .patterns import PATTERNS


def branch_moments(G, pattern, weights):
    """Return (mu, Var_uniform, Var_weighted)."""
    adj, deg = prep(G)
    m = int(deg.sum() // 2)
    _, Pw, Pu, nH = per_copy_probabilities(adj, deg, m, pattern, weights)
    mu = nH * Pu
    M_w = float(np.sum(np.where(Pw > 0, Pu * Pu / np.maximum(Pw, 1e-300), 0.0)))
    return mu, mu - mu * mu, M_w - mu * mu     # Y_u in {0,1} so E[Y_u^2] = mu


def budgets(mu, Var_u, Var_w, lam=0.5):
    """Instances required, with eps^2 and log n factored out."""
    d = mu * mu
    return dict(uniform_only=Var_u / d,
                weighted_only=Var_w / d,
                plain_mixture=(lam * Var_u + (1 - lam) * Var_w) / d,
                inverse_variance=1.0 / ((lam / Var_u + (1 - lam) / Var_w) * d))
