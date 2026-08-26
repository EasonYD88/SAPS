# Guidance-Safe Parallel Decoding under Correlated Scores

## Mathematical Theory and Experimental Contract for Codex

**Working title:** *Guidance-Safe Parallel Decoding under Correlated Scores: Projection Geometry, Budget-Aware Scheduling, and Feedback–Latency Frontiers*  
**Target venue:** ICLR 2027  
**Intended reader:** Codex or another implementation agent running the computational experiments  
**Document role:** authoritative mathematical specification; implementation details may change, but definitions, comparison rules, fairness constraints, and Go/No-Go criteria should not be silently altered.

---

## 0. Executive summary

The research question is:

> Given a frozen masked generative model and a fixed number of parallel decoding rounds, how should variables be grouped into batches so that parallelism preserves as much future reward-guidance capability as possible?

The central distinction is:

\[
\boxed{
\text{distributionally safe parallelism}
\not\Rightarrow
\text{guidance-safe parallelism}.
}
\]

A batch may have low entropy, low conditional dependence, or even zero total correlation under the base generator, yet still jointly finalize variables that participate in the same downstream reward interaction. This removes intermediate feedback opportunities and can reduce the reward response available to subsequent guidance.

The ideal product-law theory gives a combinatorial description through a **feedback frontier** and unique-maximum coloring. Real neural masked diffusion models, however, have correlated path scores and overlapping control features. Therefore the exact neural objective is not a sum of independent interaction weights. It is the reward projection onto the schedule-dependent score space:

\[
\boxed{
\Gamma_1^2(c)
=
 b_c^\top F_c^\dagger b_c,
}
\]

where

\[
Y=R-\mathbb ER,
\qquad
b_c=\mathbb E[Y\Psi_c],
\qquad
F_c=\mathbb E[\Psi_c\Psi_c^\top].
\]

Here \(\Psi_c\) is the vector of first-order path-score features available under schedule \(c\). The pseudoinverse prevents duplicate or correlated feature directions from being double counted.

The product/categorical feedback-frontier objective is retained as:

1. an exact special case;
2. an interpretable structural surrogate;
3. a source of efficient scheduling algorithms;
4. a hypothesis that must be tested against the exact correlated projection objective.

At finite KL budgets, first-order visibility alone is insufficient. Define the exact finite-budget optimized response

\[
\boxed{
\mathcal V_\varepsilon(c)
=
\sup_{u:\,D_{\mathrm{KL}}(Q_{c,u}\|P)\le \varepsilon}
\left(
\mathbb E_{Q_{c,u}}Y
\right)^2.
}
\]

The practical paper must compare the entire feedback–latency frontier:

\[
\boxed{
\mathcal V_\varepsilon^*(L)
=
\max_{c\in\mathcal C_L}
\mathcal V_\varepsilon(c),
}
\]

rather than reporting one fixed number of rounds.

The intended contribution is **not** that exact reward-aware planning cannot solve the problem. Exact subset-valued process-reward planning subsumes optimal scheduling. The proposed method is a structured, interaction-aware approximation to an otherwise combinatorial batch-action planning problem.

---

## 1. Status labels used in this document

Every result should be interpreted according to its label.

- **THEOREM:** proved under the explicitly stated assumptions.
- **COROLLARY:** direct consequence of a theorem.
- **IDEAL SPECIAL CASE:** exact under product or orthogonal assumptions, not asserted for a neural DLM.
- **ROBUSTNESS THEOREM:** bridges an ideal object to a perturbed neural object.
- **ALGORITHM:** implementable procedure; may have only partial guarantees.
- **EMPIRICAL HYPOTHESIS:** must be tested and may be false.
- **UPPER BOUND / ORACLE:** intentionally stronger or more expensive than the proposed practical method.
- **NO-GO CONDITION:** failure should cause the relevant claim to be removed or the project to pivot.

Codex must preserve these distinctions in code comments, result tables, and generated summaries.

---

# Part I. Path-space control geometry

## 2. Baseline path law and terminal reward

Let a frozen masked generative model define a path law

\[
P(d x_{0:T}).
\]

The indexing convention is not essential. In reverse diffusion notation, \(X_T\) is highly masked and \(X_0\) is the completed sample. Let

\[
\mathcal F_t
\]

be the information available to the controller at the relevant point in generation.

The terminal reward is

\[
R=R(X_0),
\]

and the centered reward is

\[
Y=R-\mathbb E_P R.
\]

Assume at least

\[
Y\in L^2(P).
\]

The reward may be:

- exact and executable, such as unit-test success;
- a cheap scientific oracle;
- a planted categorical interaction;
- a learned surrogate;
- a binary rare-event indicator.

The model is frozen. No experiment in the primary causal analysis may change the pretrained parameters.

---

## 3. Causal local controls and path scores

A schedule \(c\) determines which variables are generated or finalized at each feedback round. Under a local control parameter \(u\in\mathbb R^{p_c}\), the controlled path law is denoted

\[
Q_{c,u}.
\]

Assume local absolute continuity near \(u=0\):

\[
Q_{c,u}\ll P.
\]

Define the path score vector

\[
\boxed{
\Psi_c
=
\left.
\nabla_u
\log\frac{dQ_{c,u}}{dP}
\right|_{u=0}.
}
\]

For locally normalized Markov transitions, the score decomposes into conditionally centered martingale increments:

\[
\Psi_c
=
\sum_t \psi_{c,t},
\qquad
\mathbb E[\psi_{c,t}\mid\mathcal F_{t-1}]=0.
\]

The first-order path Fisher matrix is

\[
\boxed{
F_c
=
\mathbb E_P[\Psi_c\Psi_c^\top].
}
\]

The reward–score covariance is

\[
\boxed{
b_c
=
\mathbb E_P[Y\Psi_c].
}
\]

In practice, \(\Psi_c\) should be computed from conditional categorical logit perturbations or other explicit control directions. A feature that is merely correlated with the model hidden state is not automatically a valid path score.

---

## 4. THEOREM: exact first-order projection objective

For a local direction \(a\in\mathbb R^{p_c}\), the first-order reward response is

\[
DJ_c(0)[a]
=
 a^\top b_c.
\]

The local path-Fisher cost is

\[
\mathbb E[(a^\top\Psi_c)^2]
=
 a^\top F_c a.
\]

Therefore the maximum squared first-order coefficient under unit Fisher cost is

\[
\boxed{
\Gamma_1^2(c)
=
\sup_{a:\,a^\top F_ca\le1}
(a^\top b_c)^2
=
 b_c^\top F_c^\dagger b_c.
}
\]

### Proof

If \(v\in\operatorname{null}(F_c)\), then

\[
\mathbb E[(v^\top\Psi_c)^2]=0,
\]

so \(v^\top\Psi_c=0\) almost surely and hence \(v^\top b_c=0\). Thus \(b_c\in\operatorname{range}(F_c)\). Whitening on the range of \(F_c\) gives

\[
\sup_{a:\,a^\top F_ca\le1}
(a^\top b_c)^2
=
\|F_c^{\dagger/2}b_c\|_2^2.
\]

### Geometric form

Let

\[
\mathcal T_c
=
\overline{\operatorname{span}}\{a^\top\Psi_c:a\in\mathbb R^{p_c}\}
\subset L^2(P).
\]

Then

\[
\boxed{
\Gamma_1^2(c)
=
\|\Pi_{\mathcal T_c}Y\|_2^2.
}
\]

### Interpretation

This objective:

- is invariant to invertible feature reparameterization;
- is invariant to adding exact duplicate features;
- automatically incorporates sign cancellation;
- automatically incorporates nonorthogonal score directions;
- is the correct neural-DLM replacement for a sum of independent ANOVA energies.

This projection identity is supporting mathematics, not the headline novelty.

---

## 5. Ridge-regularized practical objective

The pseudoinverse is unstable when \(F_c\) is estimated from finite samples. The practical objective is

\[
\boxed{
\Gamma_{1,\lambda}^2(c)
=
 b_c^\top(F_c+\lambda I)^{-1}b_c,
\qquad
\lambda>0.
}
\]

Codex should report:

- the chosen \(\lambda\);
- the effective rank of \(F_c\);
- the condition number before and after regularization;
- split-half stability of \(\Gamma_{1,\lambda}^2(c)\);
- sensitivity over a predeclared \(\lambda\) grid.

Do not select \(\lambda\) on the final test reward.

---

## 6. THEOREM: reparameterization and redundancy invariance

Let the score dictionary be transformed by

\[
\widetilde\Psi=A\Psi
\]

for any matrix \(A\). If \(A\) preserves the score span, then

\[
\boxed{
\widetilde b^\top\widetilde F^\dagger\widetilde b
=
 b^\top F^\dagger b.
}
\]

In particular, adding duplicate or linearly dependent score features does not increase the objective.

### Required unit test

Use

\[
F=
\begin{pmatrix}
1&1&0\\
1&1&0\\
0&0&1
\end{pmatrix},
\qquad
b=(1,1,0)^\top.
\]

Then

\[
\|b\|^2=2,
\qquad
b^\top F^\dagger b=1.
\]

Any implementation returning 2 has double counted a duplicated direction.

---

## 7. ROBUSTNESS THEOREM: near-orthogonal Gram matrix

Assume standardized score features and

\[
F_c=I+E_c,
\qquad
\|E_c\|_{\mathrm{op}}\le\delta<1.
\]

Define the diagonal surrogate

\[
W(c)=\|b_c\|_2^2.
\]

Then

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
|\Gamma_1^2(c)-W(c)|
\le
\frac{\delta}{1-\delta}W(c).
}
\]

If \(c_W\) maximizes \(W(c)\), while \(c^*\) maximizes \(\Gamma_1^2(c)\), then

\[
\boxed{
\Gamma_1^2(c_W)
\ge
\frac{1-\delta}{1+\delta}
\Gamma_1^2(c^*).
}
\]

### Interpretation

The orthogonal weighted objective remains justified only when the visible score dictionary is close to orthonormal uniformly across schedules. The experiment must estimate

\[
\delta_c=\|F_c-I\|_{\mathrm{op}}
\]

or a restricted-eigenvalue analogue. If \(\delta_c\) is large, the diagonal objective should be treated as a weak heuristic.

---

## 8. ROBUSTNESS THEOREM: perturbation of both Gram matrix and reward covariance

Suppose an ideal model has \((F_0,b_0)=(I,b_0)\), while the actual model has

\[
F=I+E,
\qquad
b=b_0+e,
\]

with

\[
\|E\|_{\mathrm{op}}\le\delta<1.
\]

Then

\[
\boxed{
\left|
 b^\top F^{-1}b
-
\|b_0\|^2
\right|
\le
\frac{\delta}{1-\delta}\|b\|^2
+
2\|b_0\|\|e\|
+
\|e\|^2.
}
\]

This separates:

- score-correlation error through \(E\);
- reward-feature covariance error through \(e\).

The neural bridge should estimate both, not only the Gram deviation.

---

## 9. ROBUSTNESS THEOREM: score-subspace perturbation

Let \(P_c^0\) be the orthogonal projector onto an ideal schedule score subspace and \(P_c\) the projector onto the actual neural score subspace. Then

\[
\boxed{
|\Gamma_1^2(c)-\Gamma_{1,0}^2(c)|
=
|\langle Y,(P_c-P_c^0)Y\rangle|
\le
\|P_c-P_c^0\|_{\mathrm{op}}\|Y\|_2^2.
}
\]

If the bound holds uniformly over schedules:

\[
\sup_c\|P_c-P_c^0\|_{\mathrm{op}}\le\zeta,
\]

then the optimal frontier values differ by at most

\[
\zeta\operatorname{Var}(R).
\]

This is the cleanest theorem connecting the product/categorical idealization to a correlated neural score space.

---

## 10. THEOREM: residualized marginal gain through Schur complement

Let \(A\) be an already selected feature block and \(E\) a candidate block. Partition

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

Assume \(F_{AA}\) and the conditional Schur complement are nonsingular. Define

\[
F_{E\mid A}
=
F_{EE}-F_{EA}F_{AA}^{-1}F_{AE},
\]

\[
b_{E\mid A}
=
b_E-F_{EA}F_{AA}^{-1}b_A.
\]

Then

\[
\boxed{
\Gamma_1^2(A\cup E)-\Gamma_1^2(A)
=
 b_{E\mid A}^\top
F_{E\mid A}^{-1}
 b_{E\mid A}.
}
\]

For ridge regularization, apply the same identity to the block matrix \(F+\lambda I\).

### Algorithmic implication

Candidate groups should not be assigned independent weights once and then summed. Their value should be recomputed or approximated after residualizing against already available score directions.

---

## 11. Finite-sample plug-in error for projection energy

Assume

\[
\lambda_{\min}(F)\ge\mu>0,
\]

and estimates satisfy

\[
\|\widehat F-F\|_{\mathrm{op}}\le\rho<\mu,
\qquad
\|\widehat b-b\|_2\le\tau.
\]

Then

\[
\boxed{
\left|
\widehat b^\top\widehat F^{-1}\widehat b
-
b^\top F^{-1}b
\right|
\le
\frac{\|\widehat b\|^2\rho}{\mu(\mu-\rho)}
+
\frac{2\|b\|\tau+\tau^2}{\mu}.
}
\]

Codex should use this theorem to produce confidence/error bars when comparing schedule scores. If \(\rho\ge\mu\), the unregularized inverse is not statistically defensible; increase ridge or reduce the feature dictionary.

---

# Part II. Ideal categorical feedback-frontier theory

## 12. Product categorical reference law

The clean structural theorem assumes

\[
P=\bigotimes_{i=1}^d p_i,
\qquad
X_i\in\mathcal A_i.
\]

For each coordinate choose an orthonormal basis

\[
\psi_{i,0}\equiv1,
\qquad
\mathbb E_{p_i}\psi_{i,a}=0,
\qquad
\mathbb E_{p_i}[\psi_{i,a}\psi_{i,b}]=\delta_{ab}.
\]

For support \(S\subseteq[d]\) and contrast index \(\alpha\), define

\[
\Psi_{S,\alpha}(X)
=
\prod_{i\in S}\psi_{i,\alpha_i}(X_i).
\]

Then

\[
Y
=
\sum_{S,\alpha}
\widehat R_{S,\alpha}\Psi_{S,\alpha},
\]

and

\[
\operatorname{Var}(R)
=
\sum_{S,\alpha}\widehat R_{S,\alpha}^2.
\]

This exact orthogonal decomposition is an **IDEAL SPECIAL CASE**, not the neural assumption.

---

## 13. Schedule and feedback frontier

A schedule is a coloring

\[
c:[d]\to[L].
\]

Variables with the same color are generated in one parallel batch. Arbitrary state-dependent feedback is allowed between batches, while the within-batch conditional sampler is factorized across positions.

For support \(S\), define its last batch

\[
\ell^*(S)=\max_{i\in S}c(i),
\]

and its feedback-frontier width

\[
\boxed{
w_c(S)
=
\left|
\{i\in S:c(i)=\ell^*(S)\}
\right|.
}
\]

This is the number of variables from interaction \(S\) that remain jointly unresolved at the last causal feedback frontier touching \(S\).

---

## 14. THEOREM: categorical response order equals frontier width

Under the product categorical law, factorized within-batch controls, and arbitrary feedback between batches,

\[
\boxed{
m_c(\Psi_{S,\alpha})=w_c(S).
}
\]

That is:

1. every admissible smooth policy has
   \[
   \mathbb E_{Q_h}\Psi_{S,\alpha}=O(h^{w_c(S)});
   \]
2. an admissible feedback policy attains a nonzero coefficient of exactly this order.

### Proof sketch

Immediately before the final relevant batch, all earlier variables in \(S\) are observed. Their interaction product becomes a known history coefficient. Each of the \(w_c(S)\) current centered categorical contrasts has conditional mean \(O(h)\). Since the batch sampler factorizes, their product is \(O(h^{w_c(S)})\). A matching policy multiplies one current logit direction by the observed earlier interaction and applies ordinary local tilts to the remaining current variables.

---

## 15. COROLLARY: exact first-order visible energy

A categorical interaction is first-order visible exactly when

\[
w_c(S)=1.
\]

Define ideal support energy

\[
W_S
=
\sum_\alpha\widehat R_{S,\alpha}^2.
\]

Then

\[
\boxed{
\Gamma_{1,\mathrm{ideal}}^2(c)
=
\sum_{S:w_c(S)=1}W_S.
}
\]

The normalized ideal completeness is

\[
\boxed{
\eta_{1,\mathrm{ideal}}(c)
=
\frac{
\sum_{S:w_c(S)=1}W_S
}{
\operatorname{Var}(R)
}.
}
\]

This additive formula is exact only under the ideal orthogonal setting.

---

## 16. Unique-maximum coloring connection

The condition

\[
w_c(S)=1
\]

means that the maximum color in hyperedge \(S\) occurs uniquely. Hence complete first-order visibility in the ideal model is equivalent to a unique-maximum coloring of the reward interaction hypergraph.

Let \(\chi_{\mathrm{um}}\) be the unique-maximum chromatic number. Then

\[
\boxed{
\chi_{\mathrm{um}}
=
\min\{L:\eta_{1,\mathrm{ideal}}^*(L)=1\},
}
\]

where

\[
\eta_{1,\mathrm{ideal}}^*(L)
=
\max_{c:[d]\to[L]}
\eta_{1,\mathrm{ideal}}(c).
\]

Unique-maximum coloring is classical. The research contribution is its role as the complete-first-order endpoint of a generative controllability–latency frontier.

---

# Part III. Controlled separation from ordinary ordering diagnostics

## 17. q-ary modular-syndrome construction

Let

\[
X_i\overset{\mathrm{iid}}\sim\operatorname{Unif}(\mathbb Z_q)
\]

and let the terminal reward be

\[
R_b(X)
=
\mathbf1
\left\{
\sum_{i\in S}X_i=b\pmod q
\right\},
\qquad
|S|=k.
\]

Construct two schedules with the same number of rounds and the same batch-size vector.

### Feedback-preserving schedule

The final batch contains exactly one relevant variable and \(k-1\) irrelevant distractors:

\[
w_{c_A}(S)=1.
\]

### Parallel-finalization schedule

The final batch contains all \(k\) relevant variables:

\[
w_{c_B}(S)=k.
\]

Under the base law, both schedules have:

\[
H(X_i)=\log q,
\qquad
\max_xP(X_i=x)=1/q,
\qquad
\mathrm{TC}(X_B)=0.
\]

As long as at least one relevant variable is unresolved,

\[
P(R_b=1\mid\text{partial state})=1/q.
\]

Thus token confidence, entropy, base total correlation, and current scalar committor are identical.

---

## 18. THEOREM: matched-diagnostic response-exponent separation

For the feedback-preserving schedule, bias the final relevant categorical variable toward the unique satisfying category. The gain satisfies

\[
\boxed{
\Delta_A(\varepsilon)
=
\frac{\sqrt{2(q-1)}}{q}
\varepsilon^{1/2}
+O(\varepsilon).
}
\]

For the parallel-finalization schedule, factorized control must simultaneously move all \(k\) relevant variables. The gain satisfies

\[
\boxed{
\Delta_B(\varepsilon)
\sim
\frac{q-1}{q}
\left(
\frac{2\varepsilon}{k(q-1)}
\right)^{k/2}.
}
\]

Therefore distributional diagnostics can be identical while future local control exponents differ.

### Important limitation

A sufficiently expressive exact subset-action reward-aware planner can detect the superior schedule. This theorem distinguishes feedback geometry from base dependency metrics and unary action ranking; it does not prove that exact Bellman planning is blind.

---

# Part IV. DPRM and subset-action planning

## 19. Correct relationship to DPRM

Do not claim that exact process-reward planning cannot represent feedback-frontier effects.

Let a batch action be

\[
B\subseteq V,
\qquad
|B|=m.
\]

An exact subset-action value is

\[
Q^*(s,B)
=
\mathbb E[
R\mid
\text{choose batch }B\text{ now and act optimally later}
].
\]

An exact planner over all subsets subsumes optimal feedback-frontier scheduling.

The practical distinction is:

\[
\boxed{
\text{itemwise scoring}
\neq
\text{subset coordination}.
}
\]

If an implementation ranks items independently and chooses top \(m\), it assumes that the batch value is adequately represented by unary scores. Interaction-aware scheduling approximates the otherwise combinatorial subset value.

---

## 20. Controlled itemwise tie example

Use a symmetric pairwise interaction graph, such as an even cycle \(C_{2m}\). The first batch must contain exactly \(m\) vertices.

By symmetry, every vertex has the same:

- confidence;
- entropy;
- unary process score;
- graph degree;
- one-step itemwise rollout value under a symmetric default continuation.

The two alternating subsets cut all cycle edges and preserve first-order accessibility of all pair interactions. Uniform tie-breaking among balanced subsets selects one of these two subsets with probability

\[
\boxed{
\frac{2}{\binom{2m}{m}}.
}
\]

For \(2m=20\), this is approximately

\[
1.08\times10^{-5}.
\]

Exact subset rollout can recover the alternating subset. Therefore the expected result hierarchy is:

\[
\boxed{
\text{exact subset planner}
\approx
\text{interaction-aware SAPS}
>
\text{itemwise top-}m.
}
\]

The value of SAPS is computational, not representational superiority over exact subset planning.

---

## 21. Required DPRM baseline ladder

Codex must implement or approximate four levels.

### 21.1 Myopic itemwise DPRM

Estimate a value for revealing each individual position, then choose top \(m\).

### 21.2 Rollout itemwise DPRM

For each individual position:

1. reveal or prioritize that position;
2. complete the remaining trajectory using a fixed continuation policy;
3. estimate terminal reward with \(K\) rollouts;
4. choose the top \(m\) individual positions.

### 21.3 Beam subset planner

Construct batches incrementally with beam sizes such as

\[
8,
32,
128.
\]

### 21.4 Exact subset planner

Enumerate all feasible batches for small dimensions. This is an **UPPER BOUND / ORACLE**, not a scalable baseline.

### Primary planner comparison

Report planner value and planning cost jointly:

\[
\boxed{
\text{achieved subset value}
\quad\text{versus}\quad
\text{model/reward evaluations used to plan}.
}
\]

---

# Part V. Finite-budget response

## 22. Exact binary width response

For a width-\(w\) Rademacher interaction, let controlled means be \(m_1,\ldots,m_w\). The KL cost for one mean is

\[
d(m)
=
\frac12
\left[
(1+m)\log(1+m)
+
(1-m)\log(1-m)
\right].
\]

Under total KL budget \(\varepsilon\), maximize

\[
\prod_{j=1}^w m_j
\]

subject to

\[
\sum_{j=1}^w d(m_j)\le\varepsilon.
\]

The exact optimum is attained by equal KL allocation:

\[
\boxed{
g_w(\varepsilon)
=
\left[
d^{-1}(\varepsilon/w)
\right]^w.
}
\]

For small \(\varepsilon\),

\[
\boxed{
g_w(\varepsilon)
\sim
\left(
\frac{2\varepsilon}{w}
\right)^{w/2}.
}
\]

This formula shows why widths 2 and 10 must not be assigned the same zero value.

---

## 23. Budget-aware structural objective

In the ideal orthogonal case, define

\[
\lambda_w(\varepsilon)
=
\left(
\frac{g_w(\varepsilon)}{g_1(\varepsilon)}
\right)^2,
\qquad
\lambda_1=1.
\]

Then the squared-response surrogate is

\[
\boxed{
F_\varepsilon^{\mathrm{orth}}(c)
=
\sum_eW_e\lambda_{w_c(e)}(\varepsilon).
}
\]

This objective is exact only in a restricted noninterfering orthogonal model. In general neural models it is a structural surrogate.

### Calibration

Use theoretical \(\lambda_w\) as initialization, then estimate an empirical monotone sequence on a development set:

\[
1=\widehat\lambda_1
\ge
\widehat\lambda_2
\ge\cdots.
\]

Do not fit \(\lambda_w\) on final test rewards.

---

## 24. Exact finite-budget neural objective

Define

\[
\boxed{
\mathcal V_\varepsilon(c)
=
\sup_{u:\,D_{\mathrm{KL}}(Q_{c,u}\|P)\le\varepsilon}
\left(
\mathbb E_{Q_{c,u}}Y
\right)^2.
}
\]

This is the true schedule value at operating budget \(\varepsilon\). It is generally expensive to optimize.

A random-direction diagnostic is

\[
\boxed{
S_\varepsilon(c)
=
\mathbb E_V
\left[
\left(
\mathbb E_{Q_{c,V,\varepsilon}}Y
\right)^2
\right],
}
\]

where \(V\) is Fisher-whitened and every perturbation is calibrated to path KL \(\varepsilon\).

For a \(p_c\)-dimensional score space,

\[
S_\varepsilon(c)
=
\frac{2\varepsilon}{p_c}
 b_c^\top F_c^\dagger b_c
+o(\varepsilon).
\]

The experiment must measure whether the structural budget-aware surrogate predicts \(S_\varepsilon(c)\) and actual optimized steering over the practical KL range.

---

## 25. Finite-budget tie counterexample

It is possible for two schedules to maximize the first-order visible energy equally while leaving different widths among the remaining interactions. Then

\[
F_{\mathrm{first}}(c_1)
=
F_{\mathrm{first}}(c_2),
\]

but

\[
F_\varepsilon(c_1)
\ne
F_\varepsilon(c_2).
\]

Codex must include a test where:

- the schedules tie under \(\mathbf1[w=1]\);
- one leaves major interactions at width 2;
- the other leaves them at widths 5–7;
- the finite-budget objective and actual response rank them differently.

This is the minimum evidence that the finite-budget extension is substantive.

---

# Part VI. Candidate interaction libraries and learnability

## 26. Three candidate-library regimes

Every main experiment must distinguish:

### 26.1 Oracle-E

The true interaction supports are known. Purpose:

- verify the structural theorem;
- estimate the scheduling ceiling;
- isolate scheduler error.

### 26.2 Structural-E

Candidates use objective-independent structure, such as:

- AST subtrees;
- dataflow or def-use groups;
- protein contact neighborhoods;
- sequence intervals;
- fixed motifs or domains;
- graph factors from the base model architecture.

This is the primary practical regime.

### 26.3 Learned-E

Candidates are discovered from terminal labels and frozen-model representations. This is an extension, not a requirement for the first complete paper.

### Required control

Include Random-Matched-E with the same:

- number of groups;
- group-size distribution;
- variable coverage distribution;
- sequence-distance or graph-distance distribution where applicable.

---

## 27. Error decomposition for candidate discovery

Let \(f_O(c)\) be an oracle-library nonnegative scheduling objective and \(f_S(c)\) the structural-library contribution, with

\[
f_S(c)\le f_O(c).
\]

Let

\[
f_O^*=\max_cf_O(c),
\qquad
f_S^*=\max_cf_S(c),
\]

and define the discovery gap

\[
\boxed{
\Delta_{\mathrm{disc}}
=f_O^*-f_S^*.
}
\]

Suppose the estimated structural objective satisfies

\[
\sup_c|\widehat f_S(c)-f_S(c)|\le\xi
\]

and the scheduler is an \(\alpha\)-approximation for \(\widehat f_S\). Then

\[
\boxed{
f_O^*-f_O(\widehat c)
\le
\Delta_{\mathrm{disc}}
+
(1-\alpha)f_S^*
+
(1+\alpha)\xi.
}
\]

This separates:

- candidate discovery error;
- combinatorial optimization error;
- statistical estimation error.

The Oracle-E versus Structural-E gap must be reported in the main results.

---

## 28. Candidate support discovery lower bound

If one true interaction must be identified among \(M\) approximately orthogonal candidates under Gaussian terminal noise, standard multiple-hypothesis testing gives a necessary sample scale

\[
\boxed{
n
=
\Omega
\left(
\frac{\sigma^2}{\theta^2}
\log M
\right),
}
\]

where \(\theta\) is the interaction signal strength and \(\sigma^2\) the reward-noise variance.

This explains why Learned-E cannot be treated as free. When

\[
M=\sum_{j=1}^k\binom dj,
\]

both statistical and computational burdens can dominate scheduling.

---

## 29. Estimating projection energy from terminal labels

Given \(N\) base or weakly perturbed trajectories, estimate

\[
\widehat b
=
\frac1N\sum_{n=1}^N Y_n\Psi_n,
\]

\[
\widehat F
=
\frac1N\sum_{n=1}^N\Psi_n\Psi_n^\top.
\]

Use cross-fitting for squared covariance or block energy estimates:

\[
\widehat W_e
=
\widehat b_e^{(1)\top}
\widehat b_e^{(2)}.
\]

This removes the positive noise bias of \(\|\widehat b_e\|^2\).

For the full projection objective, use:

- independent splits for feature selection and evaluation;
- ridge regularization;
- bootstrap or repeated split-half stability;
- held-out schedule-response validation.

---

## 30. Rank-r completeness curve

Let \(\Psi_r\) be a nested rank-\(r\) score dictionary. Define

\[
F_r=\mathbb E[\Psi_r\Psi_r^\top],
\qquad
b_r=\mathbb E[Y\Psi_r].
\]

The learnable first-order completeness is

\[
\boxed{
\eta(r)
=
\frac{b_r^\top F_r^\dagger b_r}{\operatorname{Var}(R)}.
}
\]

This curve separates:

- control expressivity: how large \(\eta(r)\) can be;
- control learnability: how accurately it can be estimated from terminal labels;
- control complexity: how large \(r\) must be to recover most reward energy.

Codex should estimate \(\eta(r)\) for a predeclared rank grid and report simultaneous confidence intervals.

---

# Part VII. Scheduling algorithms

## 31. General structural objective

Given candidate interactions \(E\), weights \(W_e\), a schedule \(c\), and width-dependent value \(\lambda_w\), define

\[
\boxed{
F_{W,\lambda}(c)
=
\sum_{e\in E}
W_e\lambda_{w_c(e)}.
}
\]

Special cases:

- first-order UM objective:
  \[
  \lambda_1=1,
  \quad
  \lambda_{w>1}=0;
  \]
- finite-budget structural objective:
  \[
  \lambda_w=\lambda_w(\varepsilon);
  \]
- latency-penalized objective:
  \[
  F_{W,\lambda}(c)-\tau\operatorname{Latency}(c).
  \]

For correlated scores, this objective is a surrogate. The exact local target remains \(b_c^\top F_c^\dagger b_c\).

---

## 32. Pairwise interactions: reduction to capacitated Max-L-Cut

For an edge \(e=\{i,j\}\),

\[
w_c(e)=1
\iff
c(i)\ne c(j),
\]

and

\[
w_c(e)=2
\iff
c(i)=c(j).
\]

Therefore

\[
\boxed{
F_{W,\lambda}(c)
=
\lambda_2\sum_eW_e
+
(\lambda_1-\lambda_2)
\operatorname{Cut}_L(c).
}
\]

Thus pairwise finite-budget scheduling is exactly a capacitated weighted Max-\(L\)-Cut problem.

### Implementations

- exact MILP for small graphs;
- semidefinite relaxation for medium graphs;
- local search / Kernighan–Lin style swaps for larger graphs;
- balanced capacity constraints must be enforced.

---

## 33. Laminar or AST interactions: exact dynamic programming

Suppose interactions form a laminar family represented by a rooted tree. Leaves are variables; each internal node represents one interaction support.

For each subtree, a DP state records the color-count vector

\[
\mathbf n=(n_1,\ldots,n_L).
\]

The width of the subtree interaction is the number of leaves assigned the largest color appearing in that subtree. The local contribution is

\[
W_u\lambda_{w(\mathbf n)}.
\]

Children are combined by count-vector convolution.

Let

\[
S=\prod_{\ell=1}^L(b_\ell+1).
\]

A direct implementation has pseudo-polynomial complexity approximately

\[
O(|T|S^2).
\]

This is suitable for small \(L\) and AST-like laminar structures.

---

## 34. Bounded-treewidth pairwise graph

For a pairwise interaction graph with treewidth \(\mathrm{tw}\), use a nice tree decomposition. A DP state records:

- colors of vertices in the current bag;
- global or partial batch-capacity counts.

Ignoring capacities, the standard dependence is

\[
O(nL^{\mathrm{tw}+1}).
\]

With capacity vectors, the algorithm becomes pseudo-polynomial in

\[
\prod_\ell(b_\ell+1).
\]

This provides an exact or FPT route for sparse structured interaction graphs.

---

## 35. General hypergraph: conditional expectation and local search

For arbitrary hypergraphs, use a heuristic stack:

1. random balanced schedule baseline;
2. conditional-expectation assignment;
3. capacity-preserving pair swaps;
4. optional multi-vertex moves;
5. residualized feature gains;
6. periodic recomputation of the projection objective.

The guarantee for the first-order UM objective is only

\[
F(c_{\mathrm{CE}})
\ge
\mathbb E_{c\sim\mathrm{random\ balanced}}F(c).
\]

Do not claim a universal constant-factor approximation.

---

## 36. Random balanced schedule formula

Let capacities be

\[
b_1,\ldots,b_L,
\qquad
B_{<\ell}=\sum_{j<\ell}b_j.
\]

For an interaction of size \(k\), a uniformly random balanced schedule gives a unique maximum with probability

\[
\boxed{
p_k(\mathbf b)
=
\frac{
\sum_{\ell=1}^L
b_\ell
\binom{B_{<\ell}}{k-1}
}{
\binom nk
}.
}
\]

This is the principled baseline for the ideal first-order objective.

---

## 37. Residualized-SAPS

A practical correlated-score scheduler should use marginal gains conditioned on previously selected or visible feature directions.

### Greedy step

For candidate block \(e\), current selected dictionary \(A\), compute

\[
\Delta_\lambda(e\mid A)
=
 b_{e\mid A}^\top
(F_{e\mid A}+\lambda I)^{-1}
 b_{e\mid A}.
\]

Use this quantity to prioritize interaction groups or schedule moves.

### Required ablation

Compare:

1. Diagonal-SAPS;
2. Residualized-SAPS;
3. Full-Projection oracle;
4. random and dependency baselines.

The paper needs to show that residualization closes a substantial fraction of the gap between diagonal and full projection in correlated neural models.

---

# Part VIII. Feedback–latency frontier

## 38. First-order correlated-score frontier

Let \(\mathcal C_L\) be schedules with at most \(L\) sequential feedback rounds and the specified capacity/latency constraints. Define

\[
\boxed{
\eta_{\mathrm{proj}}^*(L)
=
\frac{1}{\operatorname{Var}(R)}
\max_{c\in\mathcal C_L}
 b_c^\top F_c^\dagger b_c.
}
\]

If schedule classes are nested,

\[
\eta_{\mathrm{proj}}^*(L+1)
\ge
\eta_{\mathrm{proj}}^*(L).
\]

This is the exact first-order neural feedback-completeness frontier.

---

## 39. Finite-budget frontier

Define

\[
\boxed{
\mathcal V_\varepsilon^*(L)
=
\max_{c\in\mathcal C_L}
\mathcal V_\varepsilon(c).
}
\]

The ideal structural approximation is

\[
\widetilde{\mathcal V}_\varepsilon^*(L)
=
\max_{c\in\mathcal C_L}
\sum_eW_e\lambda_{w_c(e)}(\varepsilon).
\]

The experiment must report both:

- the structural surrogate frontier;
- the measured neural reward/response frontier.

---

## 40. Pareto frontier

The primary object for a real decoder is

\[
\boxed{
\mathcal P
=
\operatorname{Pareto}
\left\{
\left(
\operatorname{Latency}(c),
\operatorname{Reward}(c)
\right)
:c\in\mathcal C
\right\}.
}
\]

Report:

- sequential rounds;
- model forward passes;
- generated token count;
- wall-clock latency;
- planning cost;
- terminal reward calls;
- success and validity;
- diversity or mode collapse.

A schedule is practically better only if it improves reward at matched latency or reduces latency at matched reward.

---

# Part IX. Experimental contracts

## 41. Phase 1: exact correlated categorical benchmark

### Generator families

1. Product categorical sanity check.
2. Tree-structured Potts model with exact conditionals.
3. Small shared-hidden masked Transformer trained on the Potts distribution.

### Reward families

- unary;
- pairwise;
- modular \(k\)-way constraints;
- mixed low-order leakage plus high-order interactions;
- overlapping and redundant groups;
- positive and negative coefficient cancellation.

### Candidate libraries

- Oracle-E;
- Structural-E;
- Random-Matched-E.

### Scheduler baselines

- random balanced;
- confidence;
- entropy;
- dependency / total correlation;
- myopic itemwise DPRM;
- rollout itemwise DPRM;
- beam subset planner;
- exact subset planner;
- Diagonal-SAPS;
- Residualized-SAPS;
- Budgeted-SAPS;
- Full-Projection oracle.

### KL grid

Use at least

\[
\varepsilon\in
\{0.002,0.01,0.05,0.15,0.30\}.
\]

### Round grid

Use at least

\[
L\in\{1,2,3,4,6,d\}.
\]

---

## 42. Phase 1 primary metrics

### Score-to-gain prediction

For each candidate schedule, compute:

- diagonal structural score;
- residualized score;
- full projection score;
- budgeted score;
- confidence score;
- dependency score;
- DPRM itemwise score;
- actual optimized or probed reward gain.

Report:

\[
\operatorname{Spearman},
\qquad
\operatorname{Pearson},
\qquad
\text{top-1 regret},
\qquad
\text{top-k recall}.
\]

### Planner approximation

Normalize planner value by random and exact subset values:

\[
\boxed{
\operatorname{ApproxRatio}
=
\frac{V_{\mathrm{method}}-V_{\mathrm{random}}}
{V_{\mathrm{subset}}-V_{\mathrm{random}}}.
}
\]

### Structural-library recovery

\[
\boxed{
\operatorname{Recovery}
=
\frac{V_{\mathrm{Structural}}-V_{\mathrm{random}}}
{V_{\mathrm{Oracle}}-V_{\mathrm{random}}}.
}
\]

---

## 43. Phase 1 Go/No-Go gates

Proceed to a large frozen DLM only if all of the following hold.

### Gate A: correlated-score bridge

Full projection predicts held-out gain materially better than the diagonal objective in the correlated regime.

Suggested threshold:

\[
\rho_{\mathrm{Spearman}}^{\mathrm{proj}}
\ge0.75
\]

and at least 0.10 above the diagonal score.

### Gate B: residualization

Residualized-SAPS closes at least 60% of the prediction or planning gap between Diagonal-SAPS and Full Projection.

### Gate C: subset-planning approximation

SAPS achieves at least 90% of exact subset planner improvement over random while using at least 10 times fewer planning evaluations.

### Gate D: candidate structure

Structural-E recovers at least 60% of Oracle-E improvement over Random-Matched-E.

### Gate E: finite budget

Budget-aware scoring outperforms first-order-only scoring in mixed-leakage and first-order-tied settings.

### Gate F: frontier

SAPS improves at least one nontrivial portion of the reward–latency Pareto frontier over random, confidence, dependency, and itemwise DPRM baselines.

Failure of a gate should remove the associated claim rather than be hidden by adding more tasks.

---

## 44. Phase 2: frozen neural DLM

Use a frozen masked diffusion language/protein model. No base-model fine-tuning in the causal scheduling study.

### Required controls

All schedulers must use:

- the same initial corruption;
- the same batch-size vector;
- the same number of model rounds;
- the same temperature;
- the same local guidance controller;
- matched model calls;
- matched terminal reward calls;
- common random numbers where possible.

### First neural rewards

Start with planted categorical interactions on natural model inputs:

- unary;
- pair/contact;
- modular \(k\)-way;
- mixed leakage.

Only after the planted mechanism passes should the study use expensive real objectives.

### Candidate libraries

For code:

- AST subtrees;
- dataflow groups;
- identifier binding groups;
- function signature/call-site groups.

For proteins:

- sequence intervals;
- contact neighborhoods;
- motif/domain groups;
- objective-independent structure groups.

---

## 45. Estimating neural path scores

For control feature \(g_j(a,h)\), define a local categorical tilt

\[
q_{\alpha_j}(a\mid h)
\propto
p_\theta(a\mid h)
\exp[\alpha_j g_j(a,h)].
\]

At \(\alpha_j=0\), the local score is

\[
\psi_j
=
 g_j(A,h)
-
\mathbb E_{p_\theta(\cdot\mid h)}g_j(A,h).
\]

The path score is the sum over affected transitions:

\[
\Psi_j
=
\sum_t\psi_{t,j}.
\]

Estimate \(F\) and \(b\) only on development data. Use held-out trajectories to test whether the resulting schedule score predicts realized steering gain.

---

## 46. Path KL accounting

For Markov categorical transitions,

\[
D_{\mathrm{KL}}(Q\|P)
=
\mathbb E_Q
\sum_t
D_{\mathrm{KL}}
\left(
q_t(\cdot\mid\mathcal F_{t-1})
\|p_t(\cdot\mid\mathcal F_{t-1})
\right).
\]

Report separately:

1. token-control KL;
2. schedule-policy KL, if the schedule itself is randomized relative to a base policy;
3. planning computation;
4. latency.

Do not combine these into one unnamed “budget.”

---

# Part X. Implementation specifications for Codex

## 47. Core software interfaces

### Schedule

```python
class Schedule(Protocol):
    num_rounds: int
    batches: tuple[tuple[int, ...], ...]

    def validate(self, num_variables: int) -> None: ...
    def frontier_width(self, group: tuple[int, ...]) -> int: ...
    def latency_cost(self) -> float: ...
```

### Score feature dictionary

```python
class ScoreFeatureDictionary(Protocol):
    names: tuple[str, ...]

    def path_scores(self, trajectory: "Trajectory") -> np.ndarray: ...
    def visible_feature_ids(self, schedule: Schedule) -> np.ndarray: ...
```

### Projection estimator

```python
@dataclass
class ProjectionEstimate:
    score: float
    ridge: float
    effective_rank: float
    condition_number: float
    b: np.ndarray
    gram: np.ndarray
    bootstrap_ci: tuple[float, float] | None
```

### Candidate library

```python
@dataclass(frozen=True)
class InteractionCandidate:
    candidate_id: str
    variables: tuple[int, ...]
    source: str  # oracle, structural, random_matched, learned
    metadata: dict[str, object]
```

### Planner

```python
class BatchPlanner(Protocol):
    def propose_batch(
        self,
        state: "PartialState",
        remaining: tuple[int, ...],
        batch_size: int,
        context: "PlanningContext",
    ) -> tuple[int, ...]: ...
```

---

## 48. Required objective implementations

Implement all four.

### Diagonal first-order

\[
F_{\mathrm{diag}}(c)
=
\sum_e\widehat W_e\mathbf1[w_c(e)=1].
\]

### Budgeted structural

\[
F_{\mathrm{budget}}(c)
=
\sum_e\widehat W_e\widehat\lambda_{w_c(e)}.
\]

### Residualized

Use sequential Schur-complement marginal gains with ridge.

### Full projection

\[
F_{\mathrm{proj},\lambda}(c)
=
\widehat b_c^\top
(\widehat F_c+\lambda I)^{-1}
\widehat b_c.
\]

For small schedule spaces, compute the full objective for every schedule. For large spaces, use it as an oracle or validation score.

---

## 49. Required unit tests

1. Duplicate-feature invariance.
2. Invertible reparameterization invariance.
3. Near-orthogonal spectral bound.
4. Schur-complement marginal identity.
5. Ridge plug-in stability.
6. Categorical frontier-width exponent.
7. q-ary syndrome probability formula.
8. Same entropy/confidence/TC/committor controlled separation.
9. First-order-tied, finite-budget-separated schedules.
10. Exact subset planner matches brute-force enumeration.
11. Laminar DP matches enumeration.
12. Pairwise objective matches capacitated Max-\(L\)-Cut reduction.
13. Frontier monotonicity when schedule classes are nested.
14. Oracle-E / Structural-E / Random-Matched-E use matched library statistics.
15. Common-random-number fairness across schedulers.

No GPU experiment should start until these tests pass.

---

## 50. Result schemas

### Schedule-level table

```text
example_id
reward_id
schedule_id
num_rounds
batch_sizes
candidate_library
planner
score_confidence
score_entropy
score_dependency
score_dprm_itemwise
score_diag
score_budgeted
score_residualized
score_projection
actual_response_power
actual_optimized_gain
terminal_success
planning_calls
model_calls
reward_calls
latency_sec
```

### Projection diagnostics

```text
schedule_id
feature_count
effective_rank
ridge
condition_number
min_eigenvalue
max_eigenvalue
gram_deviation_from_identity
split_half_score_1
split_half_score_2
bootstrap_ci_low
bootstrap_ci_high
```

### Candidate-library diagnostics

```text
reward_id
library_type
num_candidates
size_distribution
coverage_distribution
oracle_weight_coverage
objective_value
oracle_gap
random_matched_gap
```

---

## 51. Statistical protocol

Use paired experiments. Each scheduler receives the same:

- example;
- corruption seed;
- model randomness or proposal cache;
- rollout random numbers where valid;
- reward evaluation budget.

Use example-level paired bootstrap with at least 10,000 resamples for final confidence intervals.

Report:

- mean paired difference;
- median paired difference;
- win rate;
- 95% confidence interval;
- standardized effect;
- planning and execution cost.

Do not report only p-values.

---

# Part XI. Claim hierarchy and paper scope

## 52. Claims that are allowed if supported

### Claim A

Parallel schedules induce different reward-relevant score subspaces under the same frozen generator.

### Claim B

The correlated projection objective predicts held-out local steering better than an orthogonal/additive surrogate when scores are correlated.

### Claim C

Interaction-aware batch planning approximates exact subset reward-aware planning at lower action-search cost.

### Claim D

Budget-aware scheduling improves over first-order-only scheduling at practical KL budgets.

### Claim E

Structural candidate libraries recover a substantial fraction of the Oracle-E feedback frontier.

### Claim F

The method improves the controllability–latency Pareto frontier on a frozen neural DLM.

---

## 53. Claims that must not be made

Do not claim:

- that path-score projection is a new mathematical identity;
- that unique-maximum coloring is newly invented;
- that exact DPRM or exact Bellman planning cannot solve batch coordination;
- that a product categorical theorem is exact for a neural DLM;
- that interaction supports can be discovered for free;
- that all high-width interactions are equally useless;
- that lower total correlation implies higher controllability;
- that a local KL exponent alone predicts the entire practical regime;
- that a schedule is superior without matching latency and compute.

---

## 54. Primary paper narrative

The recommended narrative is:

1. **Phenomenon:** a batch can preserve the base distribution yet damage future guidance.
2. **Exact neural geometry:** controllability is projection onto a schedule-dependent correlated score space.
3. **Ideal structural theorem:** feedback frontier and unique maxima characterize the orthogonal categorical limit.
4. **Efficient approximation:** interaction-aware scheduling approximates subset-valued reward planning.
5. **Finite budget and latency:** optimize and measure the full reward–latency frontier.
6. **Neural validation:** full projection and residualized scheduling predict and improve frozen-model steering.

Path-space likelihood-ratio and martingale identities belong in preliminaries or appendices.

---

# Part XII. Recommended execution order

## 55. Stage 1: exact mathematics and synthetic implementation

Complete:

- product q-ary generator;
- correlated tree-Potts generator;
- exact conditionals;
- exact subset planner;
- projection estimators;
- structural objectives;
- pairwise and laminar solvers;
- all unit tests.

Output:

- score-versus-gain plots;
- planner value-versus-cost plots;
- finite-budget tie separation;
- Oracle/Structural/Random library comparison;
- latency frontier.

---

## 56. Stage 2: small shared-hidden neural generator

Train a small masked Transformer on the correlated exact distribution. Use it only to test:

- score correlation;
- hidden-state sharing;
- model approximation leakage;
- diagonal versus projection ranking;
- robustness of structural schedules.

Do not proceed to a large DLM if the neural bridge fails here.

---

## 57. Stage 3: frozen large DLM controlled reward

Use planted interactions first. Compare:

- Oracle-E;
- Structural-E;
- Random-Matched-E;
- myopic and rollout itemwise DPRM;
- beam and exact subset on small masked sets;
- SAPS variants.

Measure full KL and latency frontiers.

---

## 58. Stage 4: one real application

The primary recommended application is code repair because it provides:

- exact terminal verification;
- natural structural candidate groups;
- low-cost repeated evaluations;
- local and dispersed interaction tasks.

Protein may be added as a scientific validation if the controlled neural study is already complete.

---

# Part XIII. Go/No-Go decision table

## 59. Strong Go

Proceed toward ICLR submission if:

1. full correlated projection significantly improves schedule-gain prediction;
2. residualization removes redundant-group false gains;
3. SAPS reaches at least 90% of exact subset improvement with materially lower planning cost;
4. Structural-E recovers a substantial fraction of Oracle-E;
5. budget-aware scheduling beats first-order-only scheduling in the practical KL range;
6. the method improves a nontrivial portion of the latency frontier in a frozen DLM.

## 60. Narrow theory paper

Consider a narrow theory submission if:

- the categorical and correlated geometry are strong;
- the neural bridge works only in controlled models;
- real-task Structural-E is too weak;
- exact subset planning remains too expensive for large tasks.

## 61. Pivot

Pivot if:

- projection scores do not predict held-out gain;
- exact subset planner substantially outperforms SAPS without a clear cost benefit;
- Structural-E performs no better than Random-Matched-E;
- dependency/TC scheduling explains all observed gains;
- finite-budget objectives do not improve practical prediction;
- latency matching removes the advantage.

---

# Part XIV. Reference anchors

Use these only to position the work; do not copy claims without checking the papers.

1. **DPRM: A Plug-in Doob h transform-induced Token-Ordering Module for Diffusion Language Models.** arXiv:2604.24357.
2. **Generation Order and Parallel Decoding in Masked Diffusion Models: An Information-Theoretic Perspective.** arXiv:2602.00286.
3. **Planned Diffusion.** arXiv:2510.18087.
4. **Commitment Before Realization: When Classifier-Free Guidance Becomes Unnecessary in Masked Diffusion Language Models.** arXiv:2608.08082.
5. **Unique-maximum and conflict-free colorings for hypergraphs and tree graphs.** arXiv:1002.4210.
6. **Submodular Meets Spectral: Greedy Algorithms for Subset Selection, Sparse Approximation and Dictionary Selection.** arXiv:1102.3975.

The novelty target is not any single ingredient. It is the synthesis:

\[
\boxed{
\begin{aligned}
&\text{correlated neural score geometry}\\
+{}&\text{feedback-frontier structural approximation}\\
+{}&\text{batch-level approximation to subset reward planning}\\
+{}&\text{finite-budget and latency frontiers}\\
+{}&\text{frozen-model causal validation}.
\end{aligned}
}
\]

---

# Appendix A. Compact theorem list

## A.1 Exact correlated first-order objective

\[
\Gamma_1^2(c)=b_c^\top F_c^\dagger b_c.
\]

## A.2 Near-orthogonal approximation

If \(\|F_c-I\|\le\delta<1\),

\[
\frac{\|b_c\|^2}{1+\delta}
\le
\Gamma_1^2(c)
\le
\frac{\|b_c\|^2}{1-\delta}.
\]

## A.3 Subspace perturbation

\[
|\Gamma_1^2(c)-\Gamma_{1,0}^2(c)|
\le
\|P_c-P_c^0\|_{\mathrm{op}}\operatorname{Var}(R).
\]

## A.4 Residualized marginal gain

\[
\Gamma_1^2(A\cup E)-\Gamma_1^2(A)
=
 b_{E\mid A}^\top F_{E\mid A}^{-1}b_{E\mid A}.
\]

## A.5 Categorical feedback-frontier response order

\[
m_c(\Psi_{S,\alpha})=w_c(S).
\]

## A.6 Ideal first-order energy

\[
\Gamma_{1,\mathrm{ideal}}^2(c)
=
\sum_{S:w_c(S)=1}W_S.
\]

## A.7 Exact binary finite-budget width gain

\[
g_w(\varepsilon)
=
[d^{-1}(\varepsilon/w)]^w.
\]

## A.8 Candidate-library gap

\[
f_O^*-f_O(\widehat c)
\le
\Delta_{\mathrm{disc}}
+(1-\alpha)f_S^*
+(1+\alpha)\xi.
\]

## A.9 Correlated first-order frontier

\[
\eta_{\mathrm{proj}}^*(L)
=
\frac{1}{\operatorname{Var}(R)}
\max_{c\in\mathcal C_L}
 b_c^\top F_c^\dagger b_c.
\]

## A.10 Finite-budget frontier

\[
\mathcal V_\varepsilon^*(L)
=
\max_{c\in\mathcal C_L}
\mathcal V_\varepsilon(c).
\]

---

# Appendix B. Minimum Codex deliverables

Before any large-model run, Codex must produce:

```text
feedback_frontier/
├── src/
│   ├── projection.py
│   ├── residualization.py
│   ├── finite_budget.py
│   ├── candidate_library.py
│   ├── dprm_itemwise.py
│   ├── subset_planner.py
│   ├── saps.py
│   ├── pairwise_solver.py
│   ├── laminar_dp.py
│   └── frontier.py
├── tests/
│   ├── test_projection_invariance.py
│   ├── test_near_orthogonal_bound.py
│   ├── test_schur_gain.py
│   ├── test_qary_frontier.py
│   ├── test_finite_budget.py
│   ├── test_subset_planner.py
│   ├── test_laminar_dp.py
│   └── test_fairness.py
└── outputs/
    ├── schedule_scores.parquet
    ├── planner_costs.parquet
    ├── frontier_results.parquet
    └── experiment_manifest.json
```

The first milestone is complete only when:

- every listed test passes;
- exact subset values match enumeration;
- projection invariance is numerically verified;
- diagonal, residualized, and full projection scores are all logged;
- a finite-budget tied-schedule example is reproduced;
- a complete latency frontier is generated for the synthetic benchmark.

