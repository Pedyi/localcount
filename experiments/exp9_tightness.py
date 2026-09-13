"""Table 8: the bound of Theorem 3.4 is tight.

On K_{d,d} plus a perfect matching on one side we have m = d^2 + d/2,
#K3 = d^2/2 = Theta(m), and every triangle's minimum-degree vertex sits on the
right side with degree d, so D_H = d = Theta(sqrt m).  LocalCount then needs
4d + 2 instances, and the Bera-Chakrabarti worst-case bound at these
parameters is Theta(sqrt(2) d): the two agree up to the constant 2 sqrt 2.

Contrast with exp8, where the same bound gives O(1).  D_H therefore spans the
whole range Theta(1) .. Theta(sqrt m) and is correct at both ends.
~1 min, no network."""
import sys, os, math; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from localcount.graphs import gen
from localcount.cost import localcount_cost

hdr = (f"{'d':>6}{'m':>9}{'#K3':>9}{'#K3/m':>8}{'D_H':>7}{'D_H/sqrt(m)':>13}"
       f"{'k_local':>10}{'4d+2':>8}{'BC bound':>10}{'ratio':>8}")
print(hdr); print("-" * len(hdr))
for d in [20, 40, 80, 160, 320]:
    G = gen("book", d)
    r = localcount_cost(G, "K3")
    m, T, DH, k = r["m"], r["nH"], r["D_H"], r["k_local"]
    bc = min(m ** 1.5 / T, m / math.sqrt(T))
    print(f"{d:>6}{m:>9}{T:>9}{T/m:>8.2f}{DH:>7.1f}{DH/math.sqrt(m):>13.3f}"
          f"{k:>10.1f}{4*d+2:>8}{bc:>10.1f}{k/bc:>8.4f}")
    assert abs(k - (4 * d + 2)) < 1e-6, "closed form for k disagrees"
    assert abs(T - d * d // 2) < 1e-9, "closed form for #K3 disagrees"
print(f"\nratio -> 4/sqrt(2) = {4/math.sqrt(2):.4f}")
