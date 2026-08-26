# Correlated, Budget-Aware Feedback Frontiers for Parallel Generative Control

## Mathematical reconstruction, algorithmic consequences, and verified experiments

**Date:** 2026-08-21  
**Research status:** substantially strengthened, but still conditional on frozen-DLM validation  
**Recommended paper framing:** *Guidance-Safe Parallel Decoding under Correlated Scores*  

---

## 0. Executive verdict

The six objections are all valid. They do not kill the feedback-frontier direction, but they force a substantial change in the mathematical object, the DPRM comparison, the scheduling objective, and the claimed algorithmic contribution.

The revised central claim is:

> **A parallel decoding schedule should be evaluated by the reward-relevant control subspace and finite-budget response that it preserves, not by a sum of nominal interaction weights. Under weak score correlations, the weighted unique-maximum objective is a controlled approximation; under strong correlations, the correct object is a Fisher-projected or directly measured budget-aware response.**

The revised theory has four levels.

1. **General correlated-score theory.** For a schedule-dependent score vector \(\Psi_c\), first-order accessible reward energy is
   \[
   \Gamma_1^2(c)=b_c^\top F_c^\dagger b_c,
   \qquad
   b_c=\mathbb E[(R-\mathbb ER)\Psi_c],
   \qquad
   F_c=\mathbb E[\Psi_c\Psi_c^\top].
   \]
   This is invariant to reparameterization, handles redundant and overlapping groups, and cannot double count identical directions.

2. **Approximate unique-maximum theorem.** If the normalized score Gram matrix satisfies
   \[
   \|F_c-I\|_{\mathrm{op}}\le\delta<1,
   \]
   then the orthogonal weighted objective \(W(c)=\|b_c\|^2\) approximates the true projection energy multiplicatively. A scheduler maximizing \(W\) obtains at least
   \[
   \frac{1-\delta}{1+\delta}
   \]
   of the optimal projection energy. A second term quantifies low-order leakage away from the product reference theorem.

3. **Finite-budget response.** The first-order indicator \(\mathbf1\{w_c(e)=1\}\) is only the \(\varepsilon\to0\) limit. At the actual control budget, the appropriate orthogonal surrogate is
   \[
   F_\varepsilon(c)
   =
   \sum_e W_e\lambda_{w_c(e)}(\varepsilon),
   \]
   where \(\lambda_w\) is calibrated from the finite-KL response spectrum. For a binary pure interaction,
   \[
   g_w(\varepsilon)
   =
   \left[d^{-1}\left(\frac{\varepsilon}{w}\right)\right]^w,
   \qquad
   \lambda_w(\varepsilon)=g_w(\varepsilon)^2,
   \]
   is exact.

4. **Controllability-latency frontier.** Rather than optimizing one fixed round count, define
   \[
   \eta^*(L)
   =
   \max_{c:\,\text{at most }L\text{ rounds}}
   \frac{\Gamma_1^2(c)}{\operatorname{Var}(R)},
   \]
   and its finite-budget analogue. This produces the full Pareto frontier between feedback rounds, NFE/latency, and preserved control response.

The DPRM conclusion also changes:

- It is not defensible to claim that an omniscient multi-step DPRM with the complete batch-subset action space cannot discover feedback-frontier structure. Such a planner contains the same Bellman information by definition.
- The valid distinction is computational and representational: current DPRM scores candidate items and selects a top-\(m\) set. Batch value is generally non-additive. SAPS is best framed as a structural approximation to an exponentially large subset-action reward-aware planner, not as a phenomenon that exact reward-aware planning cannot capture.

The strongest surviving “aha” is therefore:

\[
\boxed{
\textbf{Dependency-safe parallelism, itemwise reward-aware parallelism, and guidance-safe parallelism are three different notions.}
}
\]

---

# 1. Correlated score geometry

## 1.1 Schedule-dependent path score class

Let \(P\) be the frozen base path law, and let

\[
Y=R-\mathbb E_PR
\]

be the centered terminal reward. A schedule \(c\) determines a finite-dimensional admissible local guidance class represented by a centered path-score vector

\[
\Psi_c=(\psi_{c,1},\ldots,\psi_{c,p_c})^\top,
\qquad
\mathbb E_P\Psi_c=0.
\]

A smooth local controlled family satisfies

\[
\log\frac{dQ_{c,u}}{dP}
=
 u^\top\Psi_c
-
\frac12u^\top F_cu
+
 o(\|u\|^2),
\]

where

\[
F_c=\mathbb E_P[\Psi_c\Psi_c^\top]
\]

is the path-score Gram/Fisher matrix. The first-order reward coefficient is

\[
b_c=\mathbb E_P[Y\Psi_c].
\]

Therefore

\[
\mathbb E_{Q_{c,u}}R-\mathbb E_PR
=
 b_c^\top u+o(\|u\|),
\]

and

\[
D_{\mathrm{KL}}(Q_{c,u}\|P)
=
\frac12u^\top F_cu+o(\|u\|^2).
\]

## 1.2 Theorem 1: correlated-score local control value

**Theorem 1.** If \(F_c\succeq0\), then \(b_c\in\operatorname{Range}(F_c)\), and

\[
\sup_{D_{\mathrm{KL}}(Q_{c,u}\|P)\le\varepsilon}
\left(
\mathbb E_{Q_{c,u}}R-\mathbb E_PR
\right)
=
\sqrt{2\varepsilon\,\Gamma_1^2(c)}
+o(\sqrt\varepsilon),
\]

where

\[
\boxed{
\Gamma_1^2(c)
=
 b_c^\top F_c^\dagger b_c.
}
\]

### Proof

If \(a^\top F_ca=0\), then \(a^\top\Psi_c=0\) almost surely, hence \(a^\top b_c=0\). Thus \(b_c\) lies in the range of \(F_c\).

Ignoring lower-order remainders, the optimization is

\[
\max_u b_c^\top u
\quad\text{subject to}\quad
u^\top F_cu\le2\varepsilon.
\]

Whiten on \(\operatorname{Range}(F_c)\): let \(z=F_c^{1/2}u\). Then

\[
b_c^\top u
=
(F_c^{\dagger/2}b_c)^\top z.
\]

Cauchy-Schwarz yields the optimum

\[
\sqrt{2\varepsilon}
\|F_c^{\dagger/2}b_c\|
=
\sqrt{2\varepsilon\,b_c^\top F_c^\dagger b_c}.
\]

The local remainder follows by smoothness. \(\square\)

## 1.3 Projection interpretation

Let

\[
\mathcal H_c
=
\operatorname{span}\{\psi_{c,1},\ldots,\psi_{c,p_c}\}
\subset L_0^2(P).
\]

The normal equations for the least-squares projection of \(Y\) onto \(\mathcal H_c\) give

\[
\boxed{
\Gamma_1^2(c)
=
\|\Pi_{\mathcal H_c}Y\|_2^2.
}
\]

This formulation resolves all four non-orthogonality concerns:

- redundant directions do not add energy;
- overlapping groups are handled through the Gram matrix;
- cancellation is retained in \(b_c\);
- the result is invariant under invertible feature reparameterization.

### Duplicate-feature example

Take two identical standardized features \(\psi_1=\psi_2\), each with reward covariance \(r\). Then

\[
F=
\begin{pmatrix}
1&1\\
1&1
\end{pmatrix},
\qquad
b=(r,r)^\top.
\]

The naive sum gives

\[
\|b\|^2=2r^2,
\]

but

\[
b^\top F^\dagger b=r^2.
\]

The numerical verification used \(r=0.73\) and obtained exactly this factor-of-two overcount.

---

# 2. Approximate bridge from neural scores to weighted unique-maximum scheduling

## 2.1 Near-orthogonal Gram theorem

Assume score coordinates have been standardized and

\[
F_c=I+E_c,
\qquad
\|E_c\|_{\mathrm{op}}\le\delta<1.
\]

Define the orthogonal surrogate

\[
W(c)=\|b_c\|^2.
\]

**Theorem 2.**

\[
\boxed{
\frac{W(c)}{1+\delta}
\le
\Gamma_1^2(c)
\le
\frac{W(c)}{1-\delta}.
}
\]

Consequently,

\[
\boxed{
\left|\Gamma_1^2(c)-W(c)\right|
\le
\frac{\delta}{1-\delta}W(c).
}
\]

### Proof

The spectral assumption implies

\[
\frac1{1+\delta}I
\preceq
F_c^{-1}
\preceq
\frac1{1-\delta}I.
\]

Sandwiching \(b_c^\top F_c^{-1}b_c\) gives the claim. \(\square\)

Because \(\Gamma_1^2(c)\le\operatorname{Var}(R)\), the lower sandwich implies

\[
W(c)\le(1+\delta)\operatorname{Var}(R),
\]

so a reward-scale absolute bound is

\[
\left|\Gamma_1^2(c)-W(c)\right|
\le
\frac{\delta(1+\delta)}{1-\delta}
\operatorname{Var}(R).
\]

## 2.2 Leakage away from the ideal product theorem

Let \(b_c^{(0)}\) be the coefficient vector predicted by the ideal product/orthogonal theorem and let

\[
\|b_c-b_c^{(0)}\|
\le
\rho\,\sigma,
\qquad
\sigma^2=\operatorname{Var}(R),
\qquad
\|b_c^{(0)}\|\le\sigma.
\]

Then

\[
\boxed{
\begin{aligned}
\left|
\Gamma_1^2(c)-\|b_c^{(0)}\|^2
\right|
\le
\Bigg[
&\frac{\delta}{1-\delta}(1+\rho)^2\\
&+2\rho+\rho^2
\Bigg]\sigma^2.
\end{aligned}
}
\]

### Proof

Insert and subtract \(\|b_c\|^2\):

\[
\left|b_c^\top F_c^{-1}b_c-\|b_c^{(0)}\|^2\right|
\le
\left|b_c^\top(F_c^{-1}-I)b_c\right|
+
\left|\|b_c\|^2-\|b_c^{(0)}\|^2\right|.
\]

The first term is at most

\[
\frac{\delta}{1-\delta}\|b_c\|^2
\le
\frac{\delta}{1-\delta}(1+\rho)^2\sigma^2.
\]

The second is at most

\[
(\|b_c\|+\|b_c^{(0)}\|)
\|b_c-b_c^{(0)}\|
\le
(2+\rho)\rho\sigma^2.
\]

Adding them proves the result. \(\square\)

This is the requested approximate theorem. It explicitly separates:

- score correlation, measured by \(\delta\);
- low-order/model leakage, measured by \(\rho\).

## 2.3 Approximation guarantee for the weighted scheduler

Suppose the bound \(\|F_c-I\|_{\mathrm{op}}\le\delta\) holds for every feasible schedule. Let

\[
c_W\in\arg\max_c W(c),
\qquad
c_*\in\arg\max_c\Gamma_1^2(c).
\]

Then

\[
\boxed{
\Gamma_1^2(c_W)
\ge
\frac{1-\delta}{1+\delta}
\Gamma_1^2(c_*).
}
\]

### Proof

\[
\Gamma_1^2(c_W)
\ge
\frac{W(c_W)}{1+\delta}
\ge
\frac{W(c_*)}{1+\delta}
\ge
\frac{1-\delta}{1+\delta}
\Gamma_1^2(c_*).
\]

Thus weighted unique-maximum scheduling remains principled under weakly correlated neural scores; it is not the exact objective when score correlations are strong.

Across 120 random instances per \(\delta\), the observed minimum surrogate/optimal projection ratios were:

| \(\delta\) | theoretical guarantee | worst observed ratio |
|---:|---:|---:|
| 0.02 | 0.9608 | 0.9971 |
| 0.10 | 0.8182 | 0.9807 |
| 0.25 | 0.6000 | 0.9391 |
| 0.45 | 0.3793 | 0.8229 |

No bound violation occurred.

---

# 3. Overlapping groups: residualized interaction value

A raw group weight is not meaningful when candidate groups overlap. The correct marginal value of a new score block is a partial-\(R^2\) quantity.

Partition an existing feature set \(A\) and a proposed block \(E\):

\[
F=
\begin{pmatrix}
F_{AA}&F_{AE}\\
F_{EA}&F_{EE}
\end{pmatrix},
\qquad
b=
\begin{pmatrix}
b_A\\b_E
\end{pmatrix}.
\]

Define residualized block covariance and Gram matrix

\[
\widetilde b_{E\mid A}
=
 b_E-F_{EA}F_{AA}^\dagger b_A,
\]

\[
\widetilde F_{E\mid A}
=
 F_{EE}-F_{EA}F_{AA}^\dagger F_{AE}.
\]

**Theorem 3.**

\[
\boxed{
\Gamma_1^2(A\cup E)-\Gamma_1^2(A)
=
\widetilde b_{E\mid A}^\top
\widetilde F_{E\mid A}^\dagger
\widetilde b_{E\mid A}.
}
\]

This follows from block Gaussian elimination/Schur complement inversion, or equivalently from projecting both the reward and new block onto the orthogonal complement of the current feature span.

Consequences:

- a duplicate block has zero marginal value;
- two individually weak blocks can have high joint value through suppressor effects;
- candidate discovery should rank residual gains, not raw \(W_e\);
- the correlated objective is the classical squared multiple-correlation/feature-selection objective and is generally non-modular.

The numerical verification matched the direct refit and the Schur-complement formula to

\[
2.84\times10^{-14}.
\]

Under restricted sparse-eigenvalue conditions, this projection objective is weakly submodular, providing standard greedy approximation guarantees. That connection is useful algorithmically, but it is existing subset-selection theory rather than headline novelty.

---

# 4. Finite-budget response rather than a binary first-order indicator

## 4.1 Why \(\mathbf1\{w=1\}\) is insufficient

The first-order weighted unique-maximum objective

\[
F_0(c)
=
\sum_eW_e\mathbf1\{w_c(e)=1\}
\]

is a singular small-budget limit. It treats widths 2 and 10 identically even though their usable response at a practical KL budget can differ by many orders of magnitude.

A robust schedule objective should be tied to the actual operating budget.

## 4.2 Direct budget-aware response power

For schedule \(c\), draw a random Fisher-whitened admissible control direction \(V\), calibrated so that its controlled path law \(Q_{c,V,\varepsilon}\) has path KL \(\varepsilon\). Define

\[
G_c(V,\varepsilon)
=
\mathbb E_{Q_{c,V,\varepsilon}}Y,
\]

and

\[
\boxed{
\mathcal S_\varepsilon(c)
=
\mathbb E_V[G_c(V,\varepsilon)^2].
}
\]

This quantity is defined for a correlated neural generator without an ANOVA decomposition. It can be estimated by paired terminal rollouts and automatically includes:

- score correlation;
- low-order leakage;
- finite perturbation effects;
- cancellation among response directions.

If \(V\) is uniform on the unit sphere in \(p_c\) Fisher-whitened dimensions, then

\[
\boxed{
\mathcal S_\varepsilon(c)
=
\frac{2\varepsilon}{p_c}
\Gamma_1^2(c)
+o(\varepsilon).
}
\]

Thus the projection objective is the infinitesimal limit of the finite-budget response-power diagnostic.

## 4.3 Exact binary finite-budget response

Consider a pure interaction involving \(w\) unresolved independent Rademacher variables. Let their controlled means be \(m_1,\ldots,m_w\). The reward response is

\[
\prod_{j=1}^wm_j,
\]

and the total KL is

\[
\sum_{j=1}^wd(m_j),
\]

where

\[
 d(m)
 =
 \frac12
 \left[
 (1+m)\log(1+m)
 +(1-m)\log(1-m)
 \right].
\]

**Theorem 4.** For \(0<\varepsilon<w\log2\),

\[
\boxed{
 g_w(\varepsilon)
 =
 \max_{\sum_jd(m_j)\le\varepsilon}
 \prod_{j=1}^wm_j
 =
 \left[
 d^{-1}\left(\frac\varepsilon w\right)
 \right]^w.
}
\]

### Proof

At an interior optimum, the Lagrangian first-order condition is

\[
\frac1{m_j}
=
\lambda d'(m_j)
=
\lambda\operatorname{arctanh}(m_j).
\]

Therefore \(m_j\operatorname{arctanh}(m_j)\) is equal for all \(j\). This function is strictly increasing on \((0,1)\), so all \(m_j\) are equal. The KL constraint is active, yielding \(d(m)=\varepsilon/w\). Boundary solutions have zero product and are suboptimal. \(\square\)

Since

\[
d(m)=\frac{m^2}{2}+\frac{m^4}{12}+O(m^6),
\]

we recover

\[
 g_w(\varepsilon)
 =
 \left(\frac{2\varepsilon}{w}\right)^{w/2}
 (1+O(\varepsilon)).
\]

Independent constrained numerical optimization matched the analytic solution within

\[
2.9\times10^{-18}.
\]

Selected exact values are:

| width \(w\) | \(g_w(10^{-3})\) | \(g_w(0.05)\) | \(g_w(0.2)\) |
|---:|---:|---:|---:|
| 1 | \(4.47\times10^{-2}\) | 0.3136 | 0.6103 |
| 2 | \(1.00\times10^{-3}\) | 0.04958 | 0.1932 |
| 3 | \(1.72\times10^{-5}\) | 0.006035 | 0.04706 |
| 4 | \(2.50\times10^{-7}\) | \(6.20\times10^{-4}\) | 0.009667 |
| 6 | \(3.70\times10^{-11}\) | \(4.59\times10^{-6}\) | \(2.86\times10^{-4}\) |
| 10 | \(3.20\times10^{-19}\) | \(9.92\times10^{-11}\) | \(9.90\times10^{-8}\) |

This makes the practical inadequacy of a binary visible/blind objective explicit.

## 4.4 Budget-aware orthogonal scheduling objective

In the product/orthogonal theorem, random sign-symmetric perturbations remove cross terms. The finite-budget response power takes the exact form

\[
\boxed{
F_\varepsilon^{\mathrm{orth}}(c)
=
\sum_eW_e\lambda_{w_c(e)}(\varepsilon),
}
\]

where \(\lambda_w\) is the squared response power of a width-\(w\) channel. For the binary pure-interaction model,

\[
\lambda_w(\varepsilon)=g_w(\varepsilon)^2.
\]

For a neural DLM, \(\lambda_w\) should not be specified by hand. It should be calibrated at the target KL regime using controlled perturbation spectroscopy, after which monotone smoothing or isotonic regression can be used if estimation noise violates the expected order

\[
\lambda_1\ge\lambda_2\ge\cdots.
\]

The general neural objective remains the directly measured \(\mathcal S_\varepsilon(c)\); the weighted spectrum is a tractable structural surrogate.

---

# 5. Precise relationship to DPRM

## 5.1 What exact DPRM already contains

DPRM defines an action-conditioned process reward

\[
R_t^\star(a;s)
=
\frac1\beta
\log
\mathbb E
\left[
\exp(\beta R(X_T))
\mid
S_t=s,S_{t+1}=s^a
\right],
\]

and a reward-tilted Gibbs rule over candidate actions. The practical algorithm assigns scores to shortlisted candidate items, ranks them, and selects the top \(m_t\) items.

If the action space is expanded to all batch subsets and the exact process value includes the same downstream guidance controller and all future feedback, then an exact dynamic planner will select the optimal batch schedule by Bellman optimality.

Therefore:

\[
\boxed{
\text{There is no honest expressivity separation between SAPS and omniscient exact subset-DPRM.}
}
\]

Any paper claim to the contrary would be false.

## 5.2 What standard itemwise DPRM does not represent directly

The practical DPRM action score is itemwise, followed by top-\(m\) selection. This implicitly uses an additive approximation to batch value. Feedback-frontier value is generally non-additive because the value of revealing token \(i\) depends on which other tokens are committed in the same batch.

### Controlled regular-cycle construction

Take an even cycle \(C_{2m}\). Pairwise reward interactions are the cycle edges. Every vertex has:

- identical local confidence;
- identical entropy;
- identical degree;
- identical unary/itemwise structural score;
- by symmetry, identical one-item process value.

Choose a first batch of exactly \(m\) vertices. A pairwise interaction is first-order preserved exactly when its endpoints receive different batch colors. Thus selecting the first batch is balanced Max-Cut on the cycle.

The two alternating batches cut all \(2m\) edges. An itemwise score followed by symmetric tie-breaking selects a uniformly random balanced subset. Its probability of selecting an optimal alternating batch is

\[
\boxed{
\frac{2}{\binom{2m}{m}},
}
\]

which decays exponentially. Its expected cut is

\[
2m\cdot\frac{m}{2m-1}
=
\frac{2m^2}{2m-1},
\]

so the coordinated optimum/random ratio approaches 2:

\[
\frac{2m}{2m^2/(2m-1)}
=2-\frac1m.
\]

For \(2m=20\), the exact enumeration gave:

- optimality probability under random itemwise tie-breaking:
  \[
  1.08\times10^{-5};
  \]
- optimum/random mean objective ratio:
  \[
  1.9.
  \]

This separates itemwise scoring from batch-level coordination without relying on different unary information.

### Syndrome construction

For \(n=2k\) independent \(q\)-ary variables, let reward depend on a syndrome over a relevant support \(S\) of size \(k\). As long as at least one relevant variable remains unresolved,

\[
P(R=1\mid\text{partial state})=\frac1q.
\]

Thus all single-token current committor/process rewards are identical. With two equal batches, the first-order-optimal first batch contains exactly \(k-1\) relevant variables and one distractor. A uniform itemwise tie-break reaches such a batch with probability

\[
\boxed{
\frac{k^2}{\binom{2k}{k}}.
}
\]

At \(k=10\), this probability is \(5.41\times10^{-4}\). At finite KL \(\varepsilon=0.05\), exact enumeration produced a 232-fold ratio between the best frontier-preserving batch gain and the random-batch mean gain.

## 5.3 Correct novelty statement

The correct framing is:

> **Exact subset-action reward-aware planning subsumes feedback-frontier scheduling but has a combinatorial action space. Itemwise DPRM is a scalable additive approximation that can miss batch synergies. SAPS uses an interaction library and structured combinatorial optimization to approximate the subset-action value efficiently.**

The required experimental baselines are therefore:

1. **Myopic/itemwise DPRM:** current candidate-wise score and top-\(m\) rule.
2. **Rollout itemwise DPRM:** better candidate-wise process rewards but still additive top-\(m\).
3. **Exact subset-rollout DPRM:** only on small instances; upper bound with \(\binom{n}{m}\) actions.
4. **Oracle-E SAPS:** true interaction candidates; theorem ceiling.
5. **Structural-E SAPS:** objective-independent AST/contact/dataflow candidates.
6. **Random matched batches.**

If exact subset-DPRM wins, that is expected. The empirical question is whether Structural-E SAPS approaches it with far fewer rollout/action evaluations.

---

# 6. Candidate interaction discovery

## 6.1 Three libraries must be separated

### Oracle-E

The true reward-relevant interaction/score blocks are supplied. This isolates scheduling theory from candidate discovery and defines an upper bound.

### Structural-E

Candidates come from objective-independent structure:

- AST and dataflow groups;
- protein contact graph neighborhoods;
- motifs/domains defined without reward labels;
- architecture-derived or representation-derived sparse groups.

This is the practical main method.

### Learned-E

Interaction candidates are discovered from terminal labels and model representations. This is a separate statistical/computational problem and need not be completely solved in the first paper.

## 6.2 Projection-library discovery gap

Let \(f_{\mathcal O}(c)\) be the projection or finite-budget objective using the oracle library and \(f_{\mathcal S}(c)\) the objective using a nested structural library. Define

\[
f_{\mathcal O}^*=\max_cf_{\mathcal O}(c),
\qquad
f_{\mathcal S}^*=\max_cf_{\mathcal S}(c),
\]

and

\[
\boxed{
\Delta_{\mathrm{disc}}
=
f_{\mathcal O}^*-f_{\mathcal S}^*.
}
\]

For any structural schedule \(\widehat c\), nestedness implies

\[
f_{\mathcal O}(\widehat c)
\ge
f_{\mathcal S}(\widehat c).
\]

Therefore

\[
\boxed{
 f_{\mathcal O}^*-f_{\mathcal O}(\widehat c)
 \le
 \Delta_{\mathrm{disc}}
 +
 \left[f_{\mathcal S}^*-f_{\mathcal S}(\widehat c)\right].
}
\]

This decomposes total regret into:

1. candidate-library/discovery error;
2. scheduling, estimation, and optimization error within the supplied library.

If an algorithm is an \(\alpha\)-approximation for an estimated structural objective \(\widehat f\), and

\[
\sup_c|\widehat f(c)-f_{\mathcal S}(c)|\le\xi,
\]

then

\[
f_{\mathcal S}(\widehat c)
\ge
\alpha f_{\mathcal S}^*-(1+\alpha)\xi,
\]

hence

\[
\boxed{
 f_{\mathcal O}^*-f_{\mathcal O}(\widehat c)
 \le
 \Delta_{\mathrm{disc}}
 +(1-\alpha)f_{\mathcal S}^*
 +(1+\alpha)\xi.
}
\]

This is the correct way to prevent candidate discovery from being hidden inside a scheduling claim.

## 6.3 Finite-label screening in a fixed candidate library

For scalar residualized candidate features \(\phi_e\), let

\[
b_e=\mathbb E[Y\phi_e],
\qquad
|Y\phi_e|\le B.
\]

With \(M\) candidates and \(n\) terminal-labeled trajectories,

\[
\widehat b_e
=
\frac1n\sum_{j=1}^nY_j\phi_e(X_j).
\]

A union Hoeffding bound gives, with probability at least \(1-\delta\),

\[
\boxed{
\max_{e\le M}|\widehat b_e-b_e|
\le
B\sqrt{
\frac{2\log(2M/\delta)}{n}
}.
}
\]

Consequently,

\[
|\widehat b_e^2-b_e^2|
\le
2|b_e|a+a^2,
\]

where \(a\) is the bound above. For group features, replace \(M\) by the total number of tested coordinates and use the residualized Schur-complement energy.

A simulation with \(M=256\), eight active candidates, and fixed signal showed exact support recovery rising from 0 at 128 labels to 87.5% at 512 and 100% at 1,024 labels. This is not a universal guarantee; it illustrates the \(\sqrt{\log M/n}\) regime for a finite structural dictionary.

If all \(k\)-subsets are allowed, \(M=\sum_{j\le k}\binom dj\) is prohibitive. Under noisy parity-like rewards, candidate discovery is closely related to computationally hard sparse interaction/parity learning. Thus Structural-E is not merely an engineering convenience; it is the realistic scope boundary.

---

# 7. Stronger scheduling algorithms for structured interaction classes

## 7.1 Pairwise interactions: exact reduction to capacitated Max-\(L\)-Cut

For a pairwise interaction \(e=\{i,j\}\),

\[
w_c(e)=
\begin{cases}
1,&c(i)\ne c(j),\\
2,&c(i)=c(j).
\end{cases}
\]

Under a finite-budget spectrum \(\lambda_1>\lambda_2\),

\[
W_{ij}\lambda_{w_c(e)}
=
W_{ij}\lambda_2
+
W_{ij}(\lambda_1-\lambda_2)
\mathbf1\{c(i)\ne c(j)\}.
\]

Therefore

\[
\boxed{
F_\lambda(c)
=
\lambda_2\sum_{ij}W_{ij}
+
(\lambda_1-\lambda_2)
\operatorname{Cut}_L(c).
}
\]

Maximizing the budget-aware objective is exactly weighted Max-\(L\)-Cut, with balance/capacity constraints if batch sizes are fixed.

Consequences:

- \(L=2\): use the Goemans-Williamson SDP family rather than a generic local heuristic;
- \(L>2\): use the Frieze-Jerrum Max-\(k\)-Cut relaxation/rounding family;
- small balanced instances: solve by MILP or branch-and-bound to obtain exact upper bounds;
- correlated scores: the reduction is an approximation controlled by the Gram perturbation theorem.

The identity was verified by exhaustive balanced-coloring enumeration to machine precision.

## 7.2 Laminar interactions: exact dynamic programming

Suppose candidate interactions form a laminar family represented by a rooted tree whose leaves are tokens and whose internal nodes are interaction groups, as in an AST hierarchy.

Let each group \(u\) have weight \(W_u\). For a subtree \(u\), define the count vector

\[
\mathbf n=(n_1,\ldots,n_L),
\]

where \(n_\ell\) is the number of leaves assigned to round \(\ell\). Its frontier width is

\[
w(\mathbf n)
=
n_{\ell^*},
\qquad
\ell^*=\max\{\ell:n_\ell>0\}.
\]

Define

\[
\mathrm{DP}_u(\mathbf n)
=
\text{best total objective inside subtree }u
\text{ with count vector }\mathbf n.
\]

For a leaf, the feasible states are one-hot count vectors. For an internal node with children \(v_1,\ldots,v_m\), convolve their tables:

\[
\mathrm{DP}_u(\mathbf n)
=
\max_{\sum_j\mathbf n^{(j)}=\mathbf n}
\sum_j\mathrm{DP}_{v_j}(\mathbf n^{(j)})
+
W_u\lambda_{w(\mathbf n)}.
\]

At the root, select the required capacity vector \((b_1,\ldots,b_L)\).

For fixed \(L\), this is polynomial/pseudo-polynomial in the batch capacities. If

\[
S=\prod_{\ell=1}^L(b_\ell+1),
\]

naive binary-tree convolution costs

\[
O(|T|S^2)
\]

and memory is \(O(|T|S)\), with substantial pruning possible.

For an eight-leaf random laminar instance with capacities \((3,3,2)\), exhaustive enumeration and the DP agreed within

\[
1.78\times10^{-15}.
\]

This gives an exact algorithm directly relevant to AST-subtree candidates.

## 7.3 Bounded-treewidth interaction graphs

For pairwise interactions, or hypergraphs with bounded primal-graph treewidth \(\tau\), use a nice tree decomposition. A DP state stores:

- the color of each vertex in the current bag: at most \(L^{\tau+1}\) possibilities;
- the global capacity count vector;
- accumulated objective from fully processed interactions.

When the last vertex of an interaction is forgotten, its \(\lambda_{w}\)-weighted contribution becomes known and is added.

Without capacity constraints, complexity is

\[
O(nL^{\tau+1})
\]

up to polynomial factors. With exact capacities and naive join convolutions it is bounded by

\[
O\left(
 nL^{\tau+1}
 \left[\prod_\ell(b_\ell+1)\right]^2
\right).
\]

Thus scheduling is fixed-parameter tractable in \(L+\tau\). A treewidth-one path implementation matched exhaustive enumeration within \(8.9\times10^{-16}\).

## 7.4 General hypergraphs

For arbitrary hypergraphs, retain conditional expectation plus capacity-preserving swaps as a scalable baseline. It has the deterministic random-baseline guarantee from the previous analysis, but no strong universal approximation ratio should be claimed.

For strongly correlated score dictionaries, replace raw edge weights by residualized marginal projection gains and re-estimate after selected blocks are added. Restricted-eigenvalue/weak-submodularity theory can then justify greedy block selection, although the composition with coloring constraints remains a harder problem.

---

# 8. Feedback-completeness and finite-budget latency frontiers

## 8.1 First-order projection frontier

Let \(\mathcal C_L\) be the feasible schedules with at most \(L\) feedback rounds and the required capacity/latency constraints. Define

\[
\boxed{
\eta_{\mathrm{proj}}^*(L)
=
\frac1{\operatorname{Var}(R)}
\max_{c\in\mathcal C_L}
 b_c^\top F_c^\dagger b_c.
}
\]

If schedule classes are nested as \(L\) grows, then

\[
0\le
\eta_{\mathrm{proj}}^*(1)
\le
\eta_{\mathrm{proj}}^*(2)
\le\cdots\le1.
\]

In the exact product/orthogonal interaction theorem,

\[
\eta_{\mathrm{proj}}^*(L)
=
\max_{c\in\mathcal C_L}
\frac{\sum_{e:w_c(e)=1}W_e}{\sum_eW_e}.
\]

The unique-maximum chromatic number is only the endpoint of the frontier:

\[
\boxed{
\chi_{\mathrm{um}}
=
\min\{L:\eta_{\mathrm{proj}}^*(L)=1\}.
}
\]

The values before that endpoint are more relevant to generative decoding.

## 8.2 Finite-budget frontier

Define

\[
\boxed{
\mathcal S_\varepsilon^*(L)
=
\max_{c\in\mathcal C_L}
\mathcal S_\varepsilon(c).
}
\]

In the orthogonal spectrum model,

\[
\mathcal S_\varepsilon^*(L)
=
\max_{c\in\mathcal C_L}
\sum_eW_e\lambda_{w_c(e)}(\varepsilon).
\]

This is the correct finite-budget controllability-latency curve. It distinguishes widths 2 and 10 and remains meaningful when small first-order leakage makes the formal response order equal to one.

A practical latency model may use

\[
\operatorname{Lat}(c)
=
\sum_{\ell=1}^L t_\ell,
\]

where \(t_\ell\) is the actual model time/NFE for round \(\ell\). The Pareto set is

\[
\boxed{
\mathcal P_\varepsilon
=
\operatorname{Pareto}
\left\{
\bigl(
\operatorname{Lat}(c),
\mathcal S_\varepsilon(c)
\bigr):c\in\mathcal C
\right\}.
}
\]

Equivalent scalarizations are

\[
\max_c
\left[
\mathcal S_\varepsilon(c)
-	au\operatorname{Lat}(c)
\right]
\]

or

\[
\min_c\operatorname{Lat}(c)
\quad\text{subject to}\quad
\mathcal S_\varepsilon(c)\ge\rho.
\]

## 8.3 Exact random balanced baseline

Let batch capacities be \(b_1,\ldots,b_L\), and

\[
B_{<\ell}=\sum_{j<\ell}b_j.
\]

For a fixed interaction of size \(k\), under a uniformly random balanced schedule, the probability that its final frontier has width \(w\) is

\[
\boxed{
P(W=w)
=
\frac{
\sum_{\ell=1}^L
\binom{b_\ell}{w}
\binom{B_{<\ell}}{k-w}
}{
\binom nk
}.
}
\]

The terms partition all \(k\)-subsets according to their maximum batch and multiplicity, so they sum to one by Vandermonde counting.

Therefore the exact random finite-budget baseline is

\[
\boxed{
\mathbb E F_\varepsilon(c)
=
\sum_eW_e
\sum_{w=1}^{|e|}
\lambda_w(\varepsilon)
\frac{
\sum_{\ell=1}^L
\binom{b_\ell}{w}
\binom{B_{<\ell}}{|e|-w}
}{
\binom n{|e|}
}.
}
\]

For capacities \((3,3,3)\) and a size-five interaction, exhaustive enumeration of all 1,680 balanced schedules produced

\[
P(W=1)=0.3571429,
\quad
P(W=2)=0.5,
\quad
P(W=3)=0.1428571,
\]

exactly matching the formula.

## 8.4 Pairwise frontier bounds

For pairwise orthogonal interactions,

\[
\eta^*(L)
=
\frac{\operatorname{Max}\text{-}L\text{-Cut}(G,W)}{\sum_eW_e}.
\]

Uniform independent \(L\)-coloring cuts each edge with probability \(1-1/L\), hence

\[
\boxed{
\eta^*(L)
\ge
1-\frac1L.
}
\]

Under exact capacities, replace this by

\[
P(c(i)\ne c(j))
=
1-
\frac{
\sum_\ell b_\ell(b_\ell-1)
}{n(n-1)}.
\]

In a verified seven-node weighted example, the exact frontier was:

| feedback rounds \(L\) | \(\eta^*(L)\) | speedup proxy \(n/L\) |
|---:|---:|---:|
| 1 | 0 | 7.00 |
| 2 | 0.8114 | 3.50 |
| 3 | 1.0000 | 2.33 |
| 4 | 1.0000 | 1.75 |

This is the kind of curve the paper should report, rather than only a fixed-three-round result.

---

# 9. Relationship to Planned Diffusion and parallel-decoding work

Planned Diffusion explicitly targets a quality-latency Pareto frontier by generating a semantic plan and denoising planned chunks in parallel. Dependency-aware and mean-field parallel decoders target joint consistency under the base model. DPRM targets reward-aware ordering through action-conditioned process rewards.

The revised feedback-frontier contribution is narrower and complementary:

- **base quality frontier:** does parallel commitment preserve the endpoint distribution or task quality?
- **dependency frontier:** are jointly committed token values mutually compatible?
- **reward-ordering frontier:** which candidate reveal action has high process reward?
- **feedback-completeness frontier:** how much future reward-control response remains addressable after committing a batch?

The main controlled experiment must therefore match:

- endpoint model and prompt;
- number of rounds and batch sizes;
- base-model confidence;
- within-batch dependency/TC diagnostics;
- terminal-label budget;
- total NFE and wall-clock latency.

It should then test whether projection or budget-aware frontier metrics predict steering gains beyond these variables.

---

# 10. Revised algorithmic package

A credible submission should not use one algorithm for every structural regime.

## 10.1 P-SAPS: projection-aware scheduling

For a supplied candidate library:

1. estimate path-score features under the frozen DLM;
2. estimate \(F\) and \(b\) by cross-fitting;
3. residualize overlapping candidate blocks using the Schur formula;
4. use the relevant structured solver:
   - SDP/Max-\(L\)-Cut for pairwise candidates;
   - exact laminar DP for AST hierarchies;
   - treewidth DP for sparse structured graphs;
   - CE + swaps for general hypergraphs;
5. evaluate the chosen schedule using held-out projection energy and direct finite-budget response power.

## 10.2 B-SAPS: budget-aware scheduling

At target KL/NFE regime \(\varepsilon\):

1. estimate or analytically calibrate \(\lambda_w(\varepsilon)\);
2. optimize
   \[
   \sum_e\widehat W_e\lambda_{w_c(e)}(\varepsilon)
   \]
   in the approximately orthogonal regime;
3. otherwise use direct schedule-level \(\widehat{\mathcal S}_\varepsilon(c)\) for reranking a shortlist produced by the structural solver.

This separates scalable proposal generation from correlation-aware final selection.

## 10.3 Candidate-library evaluation protocol

Every real experiment reports three curves:

- Oracle-E;
- Structural-E;
- random/matched Structural-E.

Learned-E is reported separately, with discovery labels and computation included in total cost.

---

# 11. New theorem and empirical claim hierarchy

## Headline theoretical claims

1. **Correlated-score control geometry**
   \[
   \Gamma_1^2(c)=b_c^\top F_c^\dagger b_c.
   \]
   This is foundational rather than novel by itself.

2. **Robust UM bridge**
   \[
   \|F_c-I\|_{\mathrm{op}}\le\delta
   \Longrightarrow
   \Gamma_1^2(c_W)
   \ge
   \frac{1-\delta}{1+\delta}
   \max_c\Gamma_1^2(c).
   \]

3. **Finite-budget frontier spectrum**
   \[
   g_w(\varepsilon)
   =
   [d^{-1}(\varepsilon/w)]^w
   \]
   for binary pure interactions, with a directly measurable neural analogue \(\mathcal S_\varepsilon(c)\).

4. **Itemwise-to-subset planning gap**
   Candidate-wise process rewards and top-\(m\) selection cannot generally represent batch synergies; exact subset planning can, but has combinatorial action complexity.

5. **Structured tractability**
   Pairwise scheduling reduces to capacitated Max-\(L\)-Cut; laminar scheduling has an exact capacity DP; bounded-treewidth scheduling is FPT.

6. **Feedback-completeness frontier**
   \[
   \eta^*(L),\quad \mathcal S_\varepsilon^*(L)
   \]
   formalize the latency-controllability Pareto curve, with UM coloring as only its full-completeness endpoint.

## Claims that should not be made

- weighted UM is exact for arbitrary neural DLM scores;
- exact multi-step DPRM is blind to feedback frontier;
- all candidate interactions can be discovered cheaply;
- first-order visibility alone predicts practical guidance;
- one generic greedy scheduler has a strong ratio for all hypergraphs;
- more feedback rounds necessarily improve wall-clock-adjusted performance.

---

# 12. Required frozen-DLM experiment

The smallest decisive experiment is a controlled masked-DLM setup with a fixed model and fixed terminal reward.

## 12.1 Schedules

- random balanced;
- confidence;
- dependency/TC-aware;
- itemwise DPRM;
- rollout itemwise DPRM;
- subset-rollout DPRM on small instances;
- Oracle-E P/B-SAPS;
- Structural-E P/B-SAPS.

## 12.2 Measurements

For every schedule and feedback-round count \(L\):

\[
\widehat\Gamma_1^2(c)
=
\widehat b_c^\top\widehat F_c^\dagger\widehat b_c,
\]

\[
\widehat{\mathcal S}_\varepsilon(c),
\]

terminal success/reward, NFE, generated tokens, GPU seconds, and wall-clock latency.

## 12.3 Critical tests

1. Does weighted UM predict projection energy when the measured Gram condition number is small?
2. Does projection energy outperform raw weighted UM when scores are correlated?
3. Does \(\mathcal S_\varepsilon(c)\) predict practical guidance better than the binary \(w=1\) metric?
4. Does Structural-E approach Oracle-E, and how large is \(\Delta_{\mathrm{disc}}\)?
5. Does P/B-SAPS approach subset-rollout DPRM with fewer action-value evaluations?
6. What is the full
   \[
   \text{reward/response}\;\text{vs.}\;L\;\text{vs. latency}
   \]
   frontier?

## 12.4 Decision rule

A strong result requires all of the following:

- projection or finite-budget frontier predicts held-out steering after controlling for confidence and dependency metrics;
- Structural-E retains a substantial fraction of the Oracle-E gain;
- batch-level scheduling improves over itemwise DPRM at matched labels/NFE;
- the gain remains on the Pareto frontier after wall-clock accounting;
- results are consistent across at least two objective geometries.

If weighted UM works only because neural score Gram matrices are nearly diagonal, the paper can still be coherent using the robustness theorem. If projection energy is predictive but no structured solver improves real steering, the work becomes a diagnostic/theory paper. If neither projection nor finite-budget response predicts steering, the feedback-frontier framing should be stopped.

---

# 13. Verified numerical evidence

The accompanying scripts freshly verified:

- near-orthogonal Gram bounds over thousands of random trials;
- leakage perturbation bounds;
- duplicate-feature non-double-counting;
- Schur-complement marginal projection identity;
- surrogate-schedule approximation guarantee;
- exact finite-budget binary interaction response and slopes \(w/2\);
- equality of analytic and numerical KL allocation optima;
- itemwise batch-coordination gaps on syndrome and regular-cycle constructions;
- pairwise Max-\(L\)-Cut reduction;
- exact laminar dynamic programming against exhaustive enumeration;
- capacity DP on a treewidth-one graph against exhaustive enumeration;
- exact balanced frontier-width distribution;
- candidate-library regret decomposition;
- finite-library support recovery behavior;
- a complete small feedback-completeness frontier.

Key numerical checks:

| check | result |
|---|---:|
| duplicate-feature naive overcount | exactly \(2\times\) |
| Schur marginal identity error | \(2.84\times10^{-14}\) |
| finite-KL allocation analytic/numeric gap | \(<2.9\times10^{-18}\) |
| laminar DP/brute-force gap | \(1.78\times10^{-15}\) |
| treewidth-one DP/brute-force gap | \(8.88\times10^{-16}\) |
| balanced width-distribution error | exactly 0 |
| pairwise reduction error | \(1.78\times10^{-15}\) |

---

# 14. Final research judgment

The six problems are mathematically resolvable, but they change the paper's center of gravity.

The paper should no longer claim:

> feedback width reduces exactly to weighted unique-maximum scheduling.

It should claim:

> **The exact neural quantity is schedule-dependent reward projection and finite-budget response. Weighted unique-maximum scheduling is an interpretable, provably robust structural approximation in weakly correlated regimes. Its practical value is efficient batch-level coordination relative to itemwise reward-aware ordering, and its relevant output is a controllability-latency frontier rather than one fixed schedule.**

The revised project is stronger because it now contains:

- a bridge from clean categorical theory to correlated neural scores;
- a finite-budget objective rather than only a local exponent;
- an honest DPRM relationship;
- an explicit discovery-error decomposition;
- strong exact/FPT algorithms for realistic structural classes;
- a formal latency-controllability Pareto object.

It remains a **Strong Conditional Go**, not yet a completed ICLR contribution. The remaining uncertainty is empirical rather than conceptual: whether the projected and finite-budget frontier metrics measurably predict steering in a frozen DLM and whether Structural-E scheduling improves the actual Pareto frontier over itemwise DPRM and dependency-aware decoding.
