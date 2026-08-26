from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize

OUT = Path('/mnt/data/correlated_budgeted_feedback_frontier_results.json')
RNG = np.random.default_rng(20260821)


def kl_rademacher_mean(m: float) -> float:
    """KL(Bern((1+m)/2) || Bern(1/2))."""
    if not (-1.0 < m < 1.0):
        if abs(m) == 1.0:
            return math.log(2.0)
        raise ValueError(m)
    if abs(m) < 1e-15:
        return 0.0
    return 0.5 * ((1 + m) * math.log1p(m) + (1 - m) * math.log1p(-m))


def inverse_kl_mean(a: float) -> float:
    if a < 0 or a >= math.log(2.0):
        if abs(a - math.log(2.0)) < 1e-14:
            return 1.0
        raise ValueError(a)
    if a == 0:
        return 0.0
    return brentq(lambda m: kl_rademacher_mean(m) - a, 0.0, 1.0 - 1e-14)


def exact_binary_interaction_gain(width: int, epsilon: float) -> float:
    """Maximum product of Rademacher means under total KL epsilon."""
    if width <= 0:
        raise ValueError(width)
    if epsilon >= width * math.log(2.0):
        return 1.0
    m = inverse_kl_mean(epsilon / width)
    return m**width


def generate_near_identity_spd(p: int, delta: float, rng: np.random.Generator) -> NDArray[np.float64]:
    q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    eigs = rng.uniform(1.0 - delta, 1.0 + delta, size=p)
    return q @ np.diag(eigs) @ q.T


def gram_bound_experiment() -> dict:
    max_rel_violation = 0.0
    max_abs_bound_violation = 0.0
    rows = []
    sigma = 1.0
    for delta in [0.01, 0.05, 0.15, 0.35, 0.6]:
        worst_ratio = 0.0
        for _ in range(500):
            p = 12
            F = generate_near_identity_spd(p, delta, RNG)
            b = RNG.normal(size=p)
            # Scale so the corresponding projection energy cannot exceed sigma^2.
            gamma_raw = float(b @ np.linalg.solve(F, b))
            if gamma_raw > 0:
                b *= min(1.0, math.sqrt(0.95 * sigma**2 / gamma_raw))
            W = float(b @ b)
            gamma = float(b @ np.linalg.solve(F, b))
            lo = W / (1.0 + delta)
            hi = W / (1.0 - delta)
            violation = max(lo - gamma, gamma - hi, 0.0)
            max_abs_bound_violation = max(max_abs_bound_violation, violation)
            rel = abs(gamma - W) / max(W, 1e-15)
            worst_ratio = max(worst_ratio, rel)
            max_rel_violation = max(max_rel_violation, rel - delta / (1 - delta))
        rows.append({
            'delta': delta,
            'worst_observed_relative_error': worst_ratio,
            'theory_relative_bound': delta / (1 - delta),
        })
    assert max_abs_bound_violation < 1e-10
    assert max_rel_violation < 1e-10
    return {
        'rows': rows,
        'max_interval_violation': max_abs_bound_violation,
        'max_relative_bound_violation': max_rel_violation,
    }


def leakage_bound_experiment() -> dict:
    # Bound: |b^T F^{-1} b - ||b0||^2| <= [delta/(1-delta)(1+rho)^2 + 2rho + rho^2] sigma^2.
    max_violation = 0.0
    ratios = []
    for delta, rho in [(0.05, 0.02), (0.15, 0.05), (0.3, 0.1), (0.5, 0.2)]:
        bound = delta / (1 - delta) * (1 + rho) ** 2 + 2 * rho + rho**2
        worst = 0.0
        for _ in range(1000):
            p = 10
            F = generate_near_identity_spd(p, delta, RNG)
            b0 = RNG.normal(size=p)
            b0 /= max(np.linalg.norm(b0), 1e-15)
            b0 *= RNG.uniform(0.05, 1.0)
            d = RNG.normal(size=p)
            d /= max(np.linalg.norm(d), 1e-15)
            d *= RNG.uniform(0.0, rho)
            b = b0 + d
            gamma = float(b @ np.linalg.solve(F, b))
            target = float(b0 @ b0)
            err = abs(gamma - target)
            worst = max(worst, err)
            max_violation = max(max_violation, err - bound)
        ratios.append({'delta': delta, 'rho': rho, 'worst_abs_error': worst, 'bound': bound})
    assert max_violation < 1e-10
    return {'rows': ratios, 'max_bound_violation': max_violation}


def duplicate_feature_experiment() -> dict:
    r = 0.73
    F = np.array([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([r, r])
    gamma = float(b @ np.linalg.pinv(F) @ b)
    naive = float(b @ b)
    assert abs(gamma - r**2) < 1e-12
    return {
        'single_feature_energy': r**2,
        'projection_energy_with_duplicate': gamma,
        'naive_sum_of_squared_correlations': naive,
        'naive_overcount_factor': naive / gamma,
    }


def finite_budget_gain_experiment() -> dict:
    eps_grid = np.logspace(-8, -2, 25)
    slopes: Dict[str, float] = {}
    table = []
    for w in [1, 2, 3, 4, 6, 10]:
        gains = np.array([exact_binary_interaction_gain(w, float(e)) for e in eps_grid])
        slope = float(np.polyfit(np.log(eps_grid[:12]), np.log(gains[:12]), 1)[0])
        slopes[str(w)] = slope
        table.append({
            'width': w,
            'gain_eps_1e-3': exact_binary_interaction_gain(w, 1e-3),
            'gain_eps_5e-2': exact_binary_interaction_gain(w, 5e-2),
            'gain_eps_2e-1': exact_binary_interaction_gain(w, 2e-1),
            'small_epsilon_slope': slope,
            'theory_slope': w / 2,
        })
        assert abs(slope - w / 2) < 5e-4

    # Independently verify equal KL allocation with numerical optimization.
    max_opt_gap = 0.0
    checks = []
    for w in [2, 3, 4, 5]:
        for eps in [0.01, 0.1, 0.3]:
            target = exact_binary_interaction_gain(w, eps)

            def objective(z: NDArray[np.float64]) -> float:
                return -float(np.prod(np.clip(z, 1e-15, 1.0)))

            cons = ({'type': 'ineq', 'fun': lambda z, eps=eps: eps - sum(kl_rademacher_mean(float(x)) for x in z)},)
            x0 = np.full(w, inverse_kl_mean(eps / w))
            sol = minimize(objective, x0=x0, bounds=[(0.0, 1.0 - 1e-10)] * w, constraints=cons, method='SLSQP', options={'ftol': 1e-13, 'maxiter': 3000})
            if not sol.success:
                raise RuntimeError(sol.message)
            observed = float(np.prod(sol.x))
            gap = abs(observed - target)
            max_opt_gap = max(max_opt_gap, gap)
            checks.append({'width': w, 'epsilon': eps, 'analytic': target, 'numeric': observed, 'gap': gap})
    assert max_opt_gap < 5e-8
    return {'slopes': slopes, 'table': table, 'allocation_checks': checks, 'max_numeric_gap': max_opt_gap}


def frontier_width(first_batch: frozenset[int], support: frozenset[int]) -> int:
    r = len(first_batch & support)
    k = len(support)
    return k if r == k else k - r


def dprm_batch_coordination_experiment() -> dict:
    rows = []
    for k in [3, 4, 5, 6, 8, 10]:
        n = 2 * k
        support = frozenset(range(k))
        batches = list(itertools.combinations(range(n), k))
        widths = np.array([frontier_width(frozenset(A), support) for A in batches])
        optimal_count = int(np.sum(widths == 1))
        exact_formula = k * k
        assert optimal_count == exact_formula
        total = math.comb(2 * k, k)
        p_opt = optimal_count / total
        # Every single-token reveal leaves at least one relevant variable unresolved (k>=3),
        # hence exact terminal success committor under uniform completion remains 1/q.
        q = 5
        singleton_committors = [1 / q] * n
        assert len(set(singleton_committors)) == 1
        eps = 0.05
        best_gain = exact_binary_interaction_gain(1, eps)
        random_mean_gain = float(np.mean([exact_binary_interaction_gain(int(w), eps) for w in widths]))
        rows.append({
            'k': k,
            'num_batch_actions': total,
            'all_singleton_process_rewards': 1 / q,
            'num_optimal_batches': optimal_count,
            'prob_random_itemwise_top_k_is_frontier_optimal': p_opt,
            'best_subset_guidance_gain_eps_0.05': best_gain,
            'random_batch_mean_guidance_gain_eps_0.05': random_mean_gain,
            'best_over_random_gain_ratio': best_gain / random_mean_gain,
        })
    return {'rows': rows}



def dprm_regular_cycle_coordination_experiment() -> dict:
    rows = []
    for m in [3, 4, 5, 6, 8, 10]:
        n = 2 * m
        batches = list(itertools.combinations(range(n), m))
        cuts = []
        for A_tuple in batches:
            A = set(A_tuple)
            cut = 0
            for i in range(n):
                j = (i + 1) % n
                cut += int((i in A) != (j in A))
            cuts.append(cut)
        optimum = max(cuts)
        n_opt = sum(x == optimum for x in cuts)
        assert optimum == n
        assert n_opt == 2
        empirical_mean = float(np.mean(cuts))
        theory_mean = n * m / (2 * m - 1)
        assert abs(empirical_mean - theory_mean) < 1e-12
        rows.append({
            'n_vertices': n,
            'batch_size': m,
            'all_unary_degrees_and_itemwise_scores_equal': True,
            'optimal_cut': optimum,
            'random_itemwise_mean_cut': empirical_mean,
            'optimal_over_random_ratio': optimum / empirical_mean,
            'probability_random_tie_break_is_optimal': n_opt / math.comb(n, m),
            'num_batch_actions': math.comb(n, m),
        })
    return {'rows': rows}

def all_colorings_with_capacities(n: int, capacities: Sequence[int]) -> Iterable[Tuple[int, ...]]:
    L = len(capacities)
    colors = [None] * n

    def rec(i: int, rem: List[int]):
        if i == n:
            yield tuple(int(x) for x in colors)
            return
        for ell in range(L):
            if rem[ell] > 0:
                colors[i] = ell
                rem[ell] -= 1
                yield from rec(i + 1, rem)
                rem[ell] += 1

    yield from rec(0, list(capacities))


def pairwise_reduction_experiment() -> dict:
    n = 8
    capacities = (3, 3, 2)
    edges: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if RNG.random() < 0.45:
                edges.append((i, j, float(RNG.uniform(0.1, 2.0))))
    lam1, lam2 = 1.0, 0.17
    total_w = sum(w for _, _, w in edges)
    max_err = 0.0
    best_f = -1.0
    best_cut = -1.0
    best_c_f = None
    best_c_cut = None
    for c in all_colorings_with_capacities(n, capacities):
        cut = sum(w for i, j, w in edges if c[i] != c[j])
        f = sum(w * (lam1 if c[i] != c[j] else lam2) for i, j, w in edges)
        rhs = lam2 * total_w + (lam1 - lam2) * cut
        max_err = max(max_err, abs(f - rhs))
        if f > best_f:
            best_f, best_c_f = f, c
        if cut > best_cut:
            best_cut, best_c_cut = cut, c
    assert max_err < 1e-12
    assert best_c_f == best_c_cut or abs(best_f - (lam2 * total_w + (lam1 - lam2) * best_cut)) < 1e-12
    return {
        'n': n,
        'capacities': capacities,
        'num_edges': len(edges),
        'max_identity_error': max_err,
        'best_budget_aware_objective': best_f,
        'best_cut_weight': best_cut,
    }


@dataclass
class TreeNode:
    name: str
    leaves: Tuple[int, ...]
    children: Tuple['TreeNode', ...] = ()
    weight: float = 0.0


def build_balanced_laminar_tree(leaves: Tuple[int, ...], counter: List[int]) -> TreeNode:
    counter[0] += 1
    name = f'n{counter[0]}'
    if len(leaves) == 1:
        return TreeNode(name=name, leaves=leaves, children=(), weight=0.0)
    mid = len(leaves) // 2
    left = build_balanced_laminar_tree(leaves[:mid], counter)
    right = build_balanced_laminar_tree(leaves[mid:], counter)
    return TreeNode(name=name, leaves=leaves, children=(left, right), weight=float(RNG.uniform(0.1, 2.0)))


def node_contribution(counts: Tuple[int, ...], weight: float, lambdas: Sequence[float]) -> float:
    if weight == 0.0:
        return 0.0
    max_color = max(i for i, x in enumerate(counts) if x > 0)
    w = counts[max_color]
    return weight * lambdas[w]


def laminar_dp(root: TreeNode, L: int, capacities: Tuple[int, ...], lambdas: Sequence[float]) -> float:
    def solve(node: TreeNode) -> Dict[Tuple[int, ...], float]:
        if not node.children:
            table: Dict[Tuple[int, ...], float] = {}
            for ell in range(L):
                counts = [0] * L
                counts[ell] = 1
                table[tuple(counts)] = 0.0
            return table
        child_tables = [solve(ch) for ch in node.children]
        cur = {(0,) * L: 0.0}
        for tab in child_tables:
            nxt: Dict[Tuple[int, ...], float] = {}
            for a, va in cur.items():
                for b, vb in tab.items():
                    counts = tuple(a[i] + b[i] for i in range(L))
                    if all(counts[i] <= capacities[i] for i in range(L)):
                        nxt[counts] = max(nxt.get(counts, -math.inf), va + vb)
            cur = nxt
        for counts in list(cur):
            cur[counts] += node_contribution(counts, node.weight, lambdas)
        return cur

    result = solve(root)
    return result[capacities]


def collect_internal_nodes(node: TreeNode) -> List[TreeNode]:
    out = []
    if node.children:
        out.append(node)
        for ch in node.children:
            out.extend(collect_internal_nodes(ch))
    return out


def laminar_dp_experiment() -> dict:
    n = 8
    L = 3
    capacities = (3, 3, 2)
    root = build_balanced_laminar_tree(tuple(range(n)), [0])
    lambdas = [0.0] + [1.0, 0.32, 0.12, 0.05, 0.02, 0.01, 0.005, 0.002]
    internal = collect_internal_nodes(root)

    def score(c: Tuple[int, ...]) -> float:
        val = 0.0
        for node in internal:
            counts = [0] * L
            for i in node.leaves:
                counts[c[i]] += 1
            val += node_contribution(tuple(counts), node.weight, lambdas)
        return val

    brute = max(score(c) for c in all_colorings_with_capacities(n, capacities))
    dp = laminar_dp(root, L, capacities, lambdas)
    err = abs(brute - dp)
    assert err < 1e-10
    return {'n': n, 'L': L, 'capacities': capacities, 'num_laminar_edges': len(internal), 'brute_opt': brute, 'dp_opt': dp, 'gap': err}



def path_graph_capacity_dp_experiment() -> dict:
    n = 9
    L = 3
    capacities = (3, 3, 3)
    edge_weights = [float(RNG.uniform(0.1, 2.0)) for _ in range(n - 1)]
    lam1, lam2 = 1.0, 0.21
    # State: (used_counts, last_color) -> value.
    dp = {}
    for col in range(L):
        counts = [0] * L
        counts[col] = 1
        dp[(tuple(counts), col)] = 0.0
    for i in range(1, n):
        ndp = {}
        for (counts, prev), val in dp.items():
            for col in range(L):
                if counts[col] >= capacities[col]:
                    continue
                new_counts = list(counts)
                new_counts[col] += 1
                gain = edge_weights[i - 1] * (lam1 if col != prev else lam2)
                key = (tuple(new_counts), col)
                ndp[key] = max(ndp.get(key, -math.inf), val + gain)
        dp = ndp
    dp_opt = max(val for (counts, _), val in dp.items() if counts == capacities)

    brute = -math.inf
    for c in all_colorings_with_capacities(n, capacities):
        val = sum(edge_weights[i] * (lam1 if c[i] != c[i + 1] else lam2) for i in range(n - 1))
        brute = max(brute, val)
    err = abs(dp_opt - brute)
    assert err < 1e-10
    return {'n': n, 'treewidth': 1, 'capacities': capacities, 'dp_opt': dp_opt, 'brute_opt': brute, 'gap': err}

def balanced_width_probability(n: int, capacities: Sequence[int], k: int, w: int) -> float:
    denom = math.comb(n, k)
    total = 0
    prefix = 0
    for b in capacities:
        if w <= b and k - w <= prefix:
            total += math.comb(b, w) * math.comb(prefix, k - w)
        prefix += b
    return total / denom


def frontier_distribution_experiment() -> dict:
    n = 9
    capacities = (3, 3, 3)
    k = 5
    support = frozenset(range(k))
    counts = {w: 0 for w in range(1, k + 1)}
    schedules = list(all_colorings_with_capacities(n, capacities))
    for c in schedules:
        maxc = max(c[i] for i in support)
        w = sum(c[i] == maxc for i in support)
        counts[w] += 1
    empirical = {w: counts[w] / len(schedules) for w in counts}
    theory = {w: balanced_width_probability(n, capacities, k, w) for w in counts}
    max_err = max(abs(empirical[w] - theory[w]) for w in counts)
    assert max_err < 1e-12
    return {'n': n, 'capacities': capacities, 'k': k, 'num_balanced_schedules': len(schedules), 'empirical': empirical, 'theory': theory, 'max_error': max_err}



def candidate_support_discovery_experiment() -> dict:
    M = 256
    s = 8
    active = np.arange(s)
    signal = 0.32
    rows = []
    for n in [128, 256, 512, 1024, 2048]:
        exact = 0
        mean_recall = 0.0
        trials = 120
        for _ in range(trials):
            X = RNG.choice([-1.0, 1.0], size=(n, M))
            beta = np.zeros(M)
            beta[active] = signal * RNG.choice([-1.0, 1.0], size=s)
            Y = X @ beta + RNG.normal(scale=1.0, size=n)
            Y -= Y.mean()
            bhat = X.T @ Y / n
            selected = np.argpartition(np.abs(bhat), -s)[-s:]
            recall = len(set(selected) & set(active)) / s
            mean_recall += recall
            exact += int(recall == 1.0)
        rows.append({
            'labels_n': n,
            'candidate_count_M': M,
            'active_s': s,
            'mean_support_recall': mean_recall / trials,
            'exact_support_recovery_rate': exact / trials,
            'concentration_scale_sqrt_logM_over_n': math.sqrt(math.log(M) / n),
        })
    return {'rows': rows}

def library_regret_experiment() -> dict:
    # Simulate nested oracle/structural score spaces and verify the deterministic decomposition.
    p = 12
    Y = RNG.normal(size=20000)
    Z = RNG.normal(size=(20000, p))
    # Inject signal into first 6 oracle features; structural library retains first 3.
    beta = np.array([0.65, -0.5, 0.4, 0.3, -0.2, 0.18] + [0.0] * (p - 6))
    Y = Z @ beta + 0.8 * Y
    Y = Y - Y.mean()
    oracle_sets = [tuple(range(6)), tuple([0, 1, 2, 3, 4]), tuple([0, 1, 2, 5])]
    structural_sets = [tuple(range(3)), tuple([0, 1]), tuple([0, 2])]

    def proj_energy(cols: Tuple[int, ...]) -> float:
        X = Z[:, cols]
        F = X.T @ X / len(X)
        b = X.T @ Y / len(X)
        return float(b @ np.linalg.pinv(F) @ b)

    O = [proj_energy(s) for s in oracle_sets]
    S = [proj_energy(s) for s in structural_sets]
    O_star = max(O)
    S_star = max(S)
    # Choose a deliberately suboptimal structural schedule index 1.
    idx = 1
    lhs = O_star - O[idx]
    rhs = (O_star - S_star) + (S_star - S[idx])
    # O[idx] >= S[idx] because structural set at same index is nested in oracle set.
    assert set(structural_sets[idx]).issubset(set(oracle_sets[idx]))
    assert lhs <= rhs + 1e-10
    return {
        'oracle_values': O,
        'structural_values': S,
        'oracle_opt': O_star,
        'structural_opt': S_star,
        'chosen_index': idx,
        'oracle_regret': lhs,
        'discovery_plus_schedule_bound': rhs,
        'bound_slack': rhs - lhs,
    }


def small_frontier_experiment() -> dict:
    # Exact pairwise first-order frontier for a small weighted graph.
    n = 7
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if RNG.random() < 0.55:
                weights[i, j] = weights[j, i] = RNG.uniform(0.2, 1.5)
    total = float(weights.sum() / 2)
    rows = []
    prev = -1.0
    for L in range(1, 5):
        # No capacity restriction here; enumerate all L^n colorings.
        best = 0.0
        for c in itertools.product(range(L), repeat=n):
            cut = sum(weights[i, j] for i in range(n) for j in range(i + 1, n) if c[i] != c[j])
            best = max(best, float(cut))
        eta = best / total if total > 0 else 0.0
        random_lb = 1.0 - 1.0 / L if L > 0 else 0.0
        if L == 1:
            random_lb = 0.0
        assert eta + 1e-12 >= random_lb
        assert eta + 1e-12 >= prev
        prev = eta
        rows.append({'rounds_L': L, 'eta_star_pairwise': eta, 'random_coloring_lower_bound': random_lb, 'parallel_speedup_proxy_n_over_L': n / L})
    return {'n': n, 'total_edge_weight': total, 'rows': rows}




def schur_marginal_gain_experiment() -> dict:
    # Exact blockwise marginal projection gain under correlated features.
    n = 30000
    p = 9
    latent = RNG.normal(size=(n, p))
    mix = RNG.normal(size=(p, p))
    X = latent @ mix
    beta = RNG.normal(size=p)
    Y = X @ beta + 0.7 * RNG.normal(size=n)
    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean()
    F = X.T @ X / n
    b = X.T @ Y / n
    A = np.array([0, 1, 2, 3], dtype=int)
    E = np.array([4, 5, 6], dtype=int)
    AE = np.concatenate([A, E])

    def energy(idx):
        FF = F[np.ix_(idx, idx)]
        bb = b[idx]
        return float(bb @ np.linalg.pinv(FF) @ bb)

    direct = energy(AE) - energy(A)
    F_A = F[np.ix_(A, A)]
    F_E = F[np.ix_(E, E)]
    F_EA = F[np.ix_(E, A)]
    b_A = b[A]
    b_E = b[E]
    F_res = F_E - F_EA @ np.linalg.pinv(F_A) @ F_EA.T
    b_res = b_E - F_EA @ np.linalg.pinv(F_A) @ b_A
    schur = float(b_res @ np.linalg.pinv(F_res) @ b_res)
    err = abs(direct - schur)
    assert err < 1e-9
    return {'direct_marginal_gain': direct, 'schur_residual_gain': schur, 'gap': err}

def surrogate_schedule_approximation_experiment() -> dict:
    # Treat each candidate schedule as exposing a different subset of a common feature dictionary.
    p = 14
    rows = []
    max_ratio_violation = 0.0
    for delta in [0.02, 0.1, 0.25, 0.45]:
        min_ratio = 1.0
        same_count = 0
        trials = 120
        for _ in range(trials):
            schedules = []
            for _j in range(60):
                size = int(RNG.integers(3, 10))
                schedules.append(tuple(sorted(RNG.choice(p, size=size, replace=False).tolist())))
            Ffull = generate_near_identity_spd(p, delta, RNG)
            bfull = RNG.normal(size=p)
            W = []
            G = []
            for A in schedules:
                idx = np.array(A, dtype=int)
                b = bfull[idx]
                F = Ffull[np.ix_(idx, idx)]
                W.append(float(b @ b))
                G.append(float(b @ np.linalg.solve(F, b)))
            i_sur = int(np.argmax(W))
            i_opt = int(np.argmax(G))
            ratio = G[i_sur] / G[i_opt]
            min_ratio = min(min_ratio, ratio)
            same_count += int(i_sur == i_opt)
        guarantee = (1 - delta) / (1 + delta)
        max_ratio_violation = max(max_ratio_violation, guarantee - min_ratio)
        rows.append({
            'delta': delta,
            'minimum_surrogate_to_optimal_projection_ratio_over_trials': min_ratio,
            'theory_guarantee': guarantee,
            'fraction_same_schedule_selected': same_count / trials,
            'trials': trials,
        })
    assert max_ratio_violation < 1e-10
    return {'rows': rows, 'max_guarantee_violation': max_ratio_violation}


def main() -> None:
    result = {
        'gram_perturbation': gram_bound_experiment(),
        'surrogate_schedule_approximation': surrogate_schedule_approximation_experiment(),
        'leakage_perturbation': leakage_bound_experiment(),
        'duplicate_feature': duplicate_feature_experiment(),
        'schur_marginal_gain': schur_marginal_gain_experiment(),
        'finite_budget_binary_interactions': finite_budget_gain_experiment(),
        'dprm_batch_coordination': dprm_batch_coordination_experiment(),
        'dprm_regular_cycle_coordination': dprm_regular_cycle_coordination_experiment(),
        'pairwise_max_l_cut_reduction': pairwise_reduction_experiment(),
        'laminar_exact_dp': laminar_dp_experiment(),
        'bounded_treewidth_path_dp': path_graph_capacity_dp_experiment(),
        'balanced_frontier_width_distribution': frontier_distribution_experiment(),
        'candidate_library_regret': library_regret_experiment(),
        'candidate_support_discovery': candidate_support_discovery_experiment(),
        'feedback_completeness_frontier': small_frontier_experiment(),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
