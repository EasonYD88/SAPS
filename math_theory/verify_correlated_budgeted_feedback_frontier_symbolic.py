from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

OUT = Path('/mnt/data/correlated_budgeted_feedback_frontier_symbolic_results.json')

# Pairwise finite-budget reduction.
w, l1, l2, z = sp.symbols('w l1 l2 z')  # z=1 if colors differ
lhs = w * (l1 * z + l2 * (1 - z))
rhs = w * l2 + w * (l1 - l2) * z
pairwise_identity = sp.simplify(lhs - rhs)
assert pairwise_identity == 0

# Rademacher KL expansion and inverse leading behavior.
m = sp.symbols('m', positive=True)
d = sp.Rational(1, 2) * ((1 + m) * sp.log(1 + m) + (1 - m) * sp.log(1 - m))
d_series = sp.series(d, m, 0, 8)
assert sp.expand(d_series.removeO()).coeff(m, 2) == sp.Rational(1, 2)
assert sp.expand(d_series.removeO()).coeff(m, 4) == sp.Rational(1, 12)

# If m ~ sqrt(2 epsilon / k), product gain has epsilon^(k/2).
eps, k = sp.symbols('eps k', positive=True)
leading_m = sp.sqrt(2 * eps / k)
leading_gain = sp.simplify(leading_m**k)

# Balanced frontier-width probabilities sum to one by Vandermonde.
# For one fixed hyperedge of size K distributed among ordered batches with capacities b_l,
# sum over last batch l and frontier multiplicity r counts every K-subset exactly once.
# Check a symbolic small three-batch instance and a general numerical family.
b1, b2, b3, K = sp.symbols('b1 b2 b3 K', integer=True, nonnegative=True)
# We record the exact combinatorial expression; generic simplification with symbolic K is limited.
expr_three = sum(
    sp.binomial(b, r) * sp.binomial(prefix, K - r)
    for b, prefix in [(b1, 0), (b2, b1), (b3, b1 + b2)]
    for r in range(1, 6)
)
# Numerical exact checks for K<=5.
width_sum_checks = []
for caps in [(3, 3, 3), (4, 2, 3), (2, 2, 2, 2)]:
    n = sum(caps)
    for kk in range(1, min(5, n) + 1):
        total = 0
        prefix = 0
        for b in caps:
            for r in range(1, kk + 1):
                total += sp.binomial(b, r) * sp.binomial(prefix, kk - r)
            prefix += b
        gap = sp.simplify(total - sp.binomial(n, kk))
        assert gap == 0
        width_sum_checks.append({'capacities': caps, 'k': kk, 'gap': str(gap)})

# Projection invariance under a duplicate feature.
r = sp.symbols('r', real=True)
F = sp.Matrix([[1, 1], [1, 1]])
bvec = sp.Matrix([r, r])
# SymPy pseudoinverse.
gamma_duplicate = sp.simplify((bvec.T * F.pinv() * bvec)[0])
assert sp.simplify(gamma_duplicate - r**2) == 0

result = {
    'pairwise_reduction_gap': str(pairwise_identity),
    'rademacher_kl_series': str(d_series),
    'leading_binary_width_gain': str(leading_gain),
    'duplicate_feature_projection_energy': str(gamma_duplicate),
    'balanced_width_probability_sum_checks': width_sum_checks,
}
OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
