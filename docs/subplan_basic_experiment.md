# SAPS Phase 1 Synthetic Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在不接入 DPLM、不训练神经网络的前提下，建立一条可重复、可配对、可严格验收的 synthetic 实验链，验证 correlated-score projection、batch coordination、finite-budget objective 和 latency–controllability frontier。

**Architecture:** 新建独立 Python 包 feedback_frontier/。生成器只负责精确分布和条件边缘；reward、candidate library、controller、scheduler 通过小接口解耦；runner 统一管理 common random numbers、计数器和结构化输出；analysis 只读取冻结的输出。所有 planner 在同一状态、同一 batch capacity、同一 proposal cache 和同一随机数预算下比较。

**Tech Stack:** Python 3.11+、NumPy、SciPy、pandas、PyArrow、PyYAML、Matplotlib、Seaborn、pytest；CPU-only。

## Global Constraints

- 本计划只覆盖原稿的 Phase 0/1；DPLM、真实 biological reward、code repair 和 shared-hidden Transformer 明确排除。
- Phase 1 随机实验固定为 d=12, q=4；Task 0/6 的 binary theorem micro-tests 使用 q=2，不进入主结果。exact subset planner 仅允许 d <= 14。
- smoke grid 固定为 coupling={0.0,0.7}、reward={unary,pairwise,modular,mixed}、L={1,2,3,4}、epsilon={0.01,0.05,0.15}、seed={0,1}。
- screening/main 的 Potts coupling 为 {0.0,0.3,0.7,1.2}，KL grid 为 epsilon={0.002,0.01,0.05,0.15,0.30}，latency grid 为 L={1,2,3,4,6,12}。
- num_instances 表示每个 seed 的总 instance 数，不是每个 Cartesian combination 的数量。按 reward 严格等分，再在 product、chain-Potts 和 balanced-tree-Potts regimes 中 deterministic round-robin；main 因而每个 reward、每个 seed 恰有 64 个 instances。
- 每个方法必须使用相同 initial state、base samples、candidate proposals 和 rollout uniforms；scheduler 内禁止创建未登记的 RNG。
- schedule 必须是位置集合的有序分割，不能遗漏、重复或越界；batch 容量只能相差 1。
- token-control KL 与 schedule-policy KL 分开记录。Phase 1 的 schedule-policy KL 记为 0.0。
- locked evaluation 前冻结 method list、ridge 规则和 calibrated width weights；不得用 locked reward 调参。
- held-out 语义固定为 **held-out reward instance + fixed-budget few-shot adaptation**，不是 zero-shot held-out instance。每个 held-out reward instance 可使用与最终 gain evaluation 严格隔离的 terminal-label adaptation trajectories 估计其 instance-specific score geometry；这些标签必须计入方法预算。
- 当前目录没有 Git metadata；任务以测试命令和产物校验作为 checkpoint。
- 所有随机实验先跑 smoke；smoke 失败时不得启动 screening/main。
- math_theory/correlated_budgeted_feedback_frontier_report.md 及其 numerical/symbolic verification artifacts 是本计划的理论规范；实现不得改写这些已验证文件。
- manifest 必须记录理论报告 SHA256 d0341f361899927fb6616b7ba059b4cc490a793619e9282758e41b88789486aa 和 numerical results SHA256 fae39d94526f283e59971700a39d128848995ca41eb064dd41e6c6863cb1b12c。

---

## 0. 冻结实验语义

### 0.1 解码与控制

给定部分状态 observed，生成器返回每个未决位置的精确条件边缘 \(p_i(a\mid observed)\)。一个 batch 内所有位置都基于 batch 开始前的同一个 observed 计算并并行采样；batch 结束后一次性写回，以保留 parallel decoding bias。

controller 计算 Monte Carlo action value：

\[
\widehat V(i,a\mid s)=K^{-1}\sum_{k=1}^K R(X^{(k)}),\quad
X^{(k)}\sim P(\cdot\mid s,X_i=a).
\]

受控边缘为：

\[
q_{i,\alpha}(a\mid s)\propto p_i(a\mid s)\exp(\alpha\widehat V(i,a\mid s)).
\]

alpha 用确定性 bisection 校准到目标 epsilon；不可达时使用最大可达 KL 并记录 kl_saturated=true。uncontrolled baseline 等价于 alpha=0。

### 0.2 Planner value

在状态 s、容量 m 下，BatchValueOracle(B) 用共享 rollout cache 估计“下一批选择 B，之后按固定 confidence schedule 完成”的 terminal reward。exact_subset 的“exact”只表示穷举全部 \(\binom{n}{m}\) 个 batch，不表示 Monte Carlo value 无噪声。value tie 用位置 tuple 字典序打破。

### 0.3 Feedback width

令 \(\ell_e^*=\max\{\ell:B_\ell\cap e\ne\varnothing\}\) 为 interaction e 最后出现的 round。理论中的 frontier width 是：

\[
w_c(e)=|B_{\ell_e^*}\cap e|.
\]

它是“最后一个 occupied batch 中仍被同时提交的变量数”，不是 interaction 横跨的 batch 数。first-order 只累计 \(w_c(e)=1\) 的 group；budgeted 累计 \(W_e\lambda_{w_c(e)}(\epsilon)\)。

### 0.4 Correlated-score estimator

\[
\hat b=N^{-1}\Psi^\top R,\qquad
\hat F=N^{-1}\Psi^\top\Psi.
\]

\(\Psi_c\) 必须是 base path law 下均值为零、由 schedule 决定的 admissible local guidance score class。对 reward instance \(r\)，相关向量明确为

\[
b_{r,c}=\mathbb E_{P_c}\left[(R_r-\mathbb E R_r)\Psi_c\right].
\]

因此 \(b_{r,c}\) 一般不能跨随机 reward instance 迁移。每个坐标先按该 instance 的 adaptation-set variance 标准化，零方差坐标丢弃并记录。标准化 Gram 为 \(F_{r,c}\)，并报告：

\[
\delta_c=\|F_c-I\|_{\rm op},\qquad
W(c)=\|b_c\|^2,\qquad
\Gamma_1^2(c)=b_c^\top F_c^\dagger b_c.
\]

Moore–Penrose 伪逆值 \(\Gamma_1^2\) 是理论主量。ridge 版本只用于有限样本稳定性分析，必须单独记为 gamma_ridge，不能替换 gamma_pinv。outer development reward instances 只用于冻结全局选择（method list、ridge rule、width weights 和阈值规则）。对每个 outer held-out reward instance，允许使用预先固定数量、仅属于该 instance 的 terminal-label adaptation trajectories 估计 \(b_{r,c}\) 和 \(F_{r,c}\)，并在 adaptation set 内做 5-fold cross-fitting。最终 gain evaluation 使用另一组 RNG domain 和 trajectories；其 reward、success 或 gain 不得回流到 geometry fit、schedule selection、ridge selection或 width calibration。

这一定义是 **few-shot adaptation to a held-out reward instance**。任何输出、图、报告或论文不得将其表述为 zero-shot。配置中的 `adaptation_trajectories` 是每个被评价 schedule geometry 的固定 terminal-label 数；方法的总 adaptation budget 为该数乘以该方法实际查询的 unique schedule geometries 数。P-SAPS shortlist 中所有被 projection/residualization 查询的 schedules 都必须计费，缓存复用不能把逻辑预算记为零。

product benchmark 还计算 ideal theorem coefficient \(b_c^{(0)}\) 与 leakage：

\[
\rho_c=\|b_c-b_c^{(0)}\|/\sqrt{Var(R)}.
\]

在 delta<1 时回归验证报告中的 combined correlation/leakage bound。Tree-Potts 没有已知 \(b^{(0)}\)，其 rho 记为 unavailable，不能用 fitted proxy 冒充。

### 0.5 Finite-budget response power

一般 correlated generator 的主诊断是：

\[
\mathcal S_\epsilon(c)=\mathbb E_V\left[
\left(\mathbb E_{Q_{c,V,\epsilon}}R-\mathbb E_PR\right)^2
\right],
\]

其中 V 在 Fisher-whitened admissible score space 的单位球面均匀采样，并把 controlled path law 校准到 path KL epsilon。binary pure interaction 的解析基准为：

\[
d(m)=\tfrac12[(1+m)\log(1+m)+(1-m)\log(1-m)],
\]

\[
g_w(\epsilon)=\left[d^{-1}(\epsilon/w)\right]^w,\qquad
\lambda_w(\epsilon)=g_w(\epsilon)^2.
\]

不得用手写 exponential decay 代替这个解析基准。对于 q-ary/相关模型，lambda 只能由 response probes 校准；直接测得的 \(\mathcal S_\epsilon\) 是最终诊断，weighted spectrum 只是结构 surrogate。

### 0.6 Go gates

| 假设 | 主指标 | Go gate |
|---|---|---|
| H1 correlated-score bridge | held-out gain 的 Spearman | cross-fitted gamma_pinv rho >= 0.75，且比 standardized weighted UM 高至少 0.10 |
| H2 batch coordination | random-normalized approximation ratio | p_saps_residualized >= 0.90，action-value evaluations 比 subset_exact 至少低 10x |
| H3 finite budget | epsilon={0.05,0.15} 的 paired actual gain | b_saps_budgeted-minus-saps_diagonal 的 95% bootstrap CI 下界 > 0 |
| H4 latency frontier | finite-budget response-vs-latency Pareto AUC | b_saps_budgeted-minus-random 的 AUC 95% CI 下界 > 0 |

辅助 gate：p_saps_residualized 关闭 saps_diagonal 到 p_saps_projection 差距的 >=60%；Structural-E 恢复 Oracle-E 相对 random 改善的 >=60%。

### 0.7 Claim guardrails

- weighted unique-maximum 只在标准化 Gram 满足 \(\delta<1\) 时有 \((1-\delta)/(1+\delta)\) approximation guarantee；不得宣称对任意 correlated scores 精确。
- omniscient exact subset-DPRM 通过 Bellman optimality 包含最优 schedule；实验只比较 SAPS 相对 itemwise DPRM 的 batch coordination，以及相对 exact subset 的计算效率。
- candidate discovery 与 scheduling 分开报告；Structural-E 不得伪装成 oracle support。
- feedback frontier 使用 at-most-L 的 nested envelope。exact-L 曲线可以不单调；不得宣称增加 rounds 必然改善 wall-clock-adjusted performance。

---

## 1. 目标文件结构

    feedback_frontier/
    ├── pyproject.toml
    ├── README.md
    ├── configs/{smoke,screening,phase1_main}.yaml
    ├── src/feedback_frontier/
    │   ├── __init__.py
    │   ├── cli.py
    │   ├── config.py
    │   ├── schemas.py
    │   ├── rng.py
    │   ├── theory.py
    │   ├── generators/{categorical_product,potts_tree}.py
    │   ├── rewards/synthetic.py
    │   ├── candidates/libraries.py
    │   ├── features/path_scores.py
    │   ├── estimators/{projection,response_power}.py
    │   ├── controllers/rollout_local.py
    │   ├── schedulers/{base,non_reward,dprm,subset,structured,saps}.py
    │   ├── runners/{probe_response,evaluate_planner}.py
    │   └── analysis/{aggregate,bootstrap,figures}.py
    ├── tests/
    └── outputs/

generator 不知道 scheduler/reward；scheduler 不写文件；runner 是唯一允许创建实验输出的模块；analysis 不重新运行模型。

---

### Task 0: 锁定并回归验证 math_theory 理论合同

**Files:**

- Read only: math_theory/correlated_budgeted_feedback_frontier_report.md
- Read only: math_theory/verify_correlated_budgeted_feedback_frontier.py
- Read only: math_theory/verify_correlated_budgeted_feedback_frontier_symbolic.py
- Read only: math_theory/correlated_budgeted_feedback_frontier_{results,symbolic_results}.json
- Create: feedback_frontier/src/feedback_frontier/theory.py
- Test: feedback_frontier/tests/test_theory_regressions.py

**Interfaces:**

- kl_rademacher_mean(m: float) -> float
- inverse_kl_mean(a: float) -> float
- exact_binary_interaction_gain(width: int, epsilon: float) -> float
- frontier_width(round_of_position: tuple[int,...], support: tuple[int,...]) -> int
- balanced_width_probability(n, capacities, support_size, width) -> float
- gram_delta(standardized_F: ndarray) -> float
- um_approximation_lower_bound(delta: float) -> float

- [ ] **Step 1: 写理论 hash 与数值回归红灯测试**

测试先校验 report/results 的两个固定 SHA256，再从 committed JSON 读取 duplicate-feature、finite-budget、Schur、pairwise reduction、laminar DP、treewidth DP 和 balanced-width expected values。

    assert exact_binary_interaction_gain(3, 0.05) == pytest.approx(
        0.006035048130208016, rel=1e-12
    )
    assert frontier_width((0, 1, 0, 1), (0, 1, 2)) == 1
    assert frontier_width((0, 0, 1, 1), (0, 2, 3)) == 2

- [ ] **Step 2: 确认红灯**

Run: cd feedback_frontier && python -m pytest tests/test_theory_regressions.py -q

Expected: import feedback_frontier.theory 失败。

- [ ] **Step 3: 移植纯函数，不复制实验 driver**

theory.py 只移植 verification script 中无副作用的解析/组合函数；不得导入其全局 RNG、不得写 /mnt/data、不得修改 math_theory。inverse_kl_mean 使用 scipy.optimize.brentq；epsilon >= width*log(2) 时 gain 饱和为 1。

- [ ] **Step 4: 补齐 theorem regressions**

断言 duplicate feature projection 等于 single feature、near-orthogonal sandwich、Schur identity、binary small-epsilon log-log slope 为 w/2、balanced width probabilities 对 w 求和为 1、pairwise objective 等于常数加 weighted Max-L-Cut。

Run: cd feedback_frontier && python -m pytest tests/test_theory_regressions.py -q

Expected: PASS，且 numerical reference gaps 不大于 committed results 中的 tolerance。

---

### Task 1: 建立可安装包、CLI 和严格配置

**Files:**

- Create: feedback_frontier/pyproject.toml
- Create: feedback_frontier/README.md
- Create: feedback_frontier/src/feedback_frontier/{__init__,config,cli}.py
- Create: feedback_frontier/configs/{smoke,screening,phase1_main}.yaml
- Test: feedback_frontier/tests/test_smoke_contract.py

**Interfaces:**

- Produces: ExperimentConfig.from_yaml(path: Path) -> ExperimentConfig
- Produces CLI: feedback-frontier validate-config|run|analyze

- [ ] **Step 1: 写配置契约红灯测试**

    def test_smoke_grid_is_frozen():
        cfg = ExperimentConfig.from_yaml(Path("configs/smoke.yaml"))
        assert cfg.d == 12 and cfg.q == 4
        assert cfg.couplings == (0.0, 0.7)
        assert cfg.rewards == ("unary", "pairwise", "modular", "mixed")
        assert cfg.rounds == (1, 2, 3, 4)
        assert cfg.epsilons == (0.01, 0.05, 0.15)
        assert cfg.seeds == (0, 1)
        assert cfg.num_instances == 16
        assert cfg.rollouts == 2

另断言 d>14 配合 exact subset 时抛 ValueError("exact_subset requires d <= 14")。

- [ ] **Step 2: 确认测试失败**

Run: cd feedback_frontier && python -m pytest tests/test_smoke_contract.py -q

Expected: import feedback_frontier.config 失败。

- [ ] **Step 3: 建立 package 和配置**

pyproject runtime dependencies 固定为 numpy>=2.0,<3、scipy>=1.13,<2、pandas>=2.2,<3、pyarrow>=16,<20、PyYAML>=6,<7、matplotlib>=3.9,<4、seaborn>=0.13,<1；dev dependency 为 pytest>=8,<9。

| config | instances | seeds | rollouts | adaptation trajectories / schedule | epsilon | L |
|---|---:|---|---:|---:|---|---|
| smoke | 16 | 0,1 | 2 | 40 | 0.01,0.05,0.15 | 1,2,3,4 |
| screening | 64 | 0 | 4 | 80 | 0.002,0.01,0.05,0.15,0.30 | 1,2,3,4,6,12 |
| phase1_main | 256 | 0,1,2 | 16 | 320 | 0.002,0.01,0.05,0.15,0.30 | 1,2,3,4,6,12 |

配置还包含 topologies、candidate_libraries、全部 methods、dprm_beta=1.0、ridge_multipliers、bootstrap_replicates=10000、response_directions={smoke:8,screening:16,main:64} 和两个 theory SHA256。

- [ ] **Step 4: 实现 frozen dataclass 解析**

YAML list 转 tuple。拒绝未知 method/reward、非正 rollout、非法 epsilon、L>d；错误消息必须包含字段和值，不得静默回退。

- [ ] **Step 5: 安装与绿灯**

Run: cd feedback_frontier && python -m pip install -e '.[dev]' && feedback-frontier validate-config --config configs/smoke.yaml

Expected: exit 0，输出 valid: d=12 q=4 instances=16。

---

### Task 2: 冻结 schema、schedule 和 RNG

**Files:**

- Create: feedback_frontier/src/feedback_frontier/{schemas,rng}.py
- Create: feedback_frontier/src/feedback_frontier/schedulers/base.py
- Test: feedback_frontier/tests/{test_schedules,test_common_random_numbers}.py

**Interfaces:**

- Schedule(batches: tuple[tuple[int,...],...], d: int)
- Schedule.round_of_position -> tuple[int,...]，供 theory.frontier_width 使用
- balanced_capacities(d, rounds) -> tuple[int,...]
- SeedBook(master_seed).rng(namespace, *keys) -> np.random.Generator
- frozen TrajectoryRecord、ScheduleScoreRecord

- [ ] **Step 1: 写 schedule tests**

    @pytest.mark.parametrize("d,L,expected", [
        (12, 1, (12,)), (12, 3, (4, 4, 4)),
        (12, 4, (3, 3, 3, 3)), (12, 5, (3, 3, 2, 2, 2)),
    ])
    def test_capacities(d, L, expected):
        assert balanced_capacities(d, L) == expected

重复、遗漏、越界位置都必须抛含 partition 的 ValueError。

- [ ] **Step 2: 写 common-random-number tests**

相同 master_seed/namespace/keys 产生相同数组；scheduler 名不得进入 base/proposal/rollout namespace；不同 example/round 产生不同数组。

- [ ] **Step 3: 实现稳定 seed derivation**

禁止 Python hash()。将 namespace/keys canonical JSON 编码，经 blake2b(digest_size=16) 转 SeedSequence。namespace 只允许 instance、base、proposal、rollout、scheduler_tie、bootstrap。

- [ ] **Step 4: 实现记录 schema**

TrajectoryRecord 精确包含：

    example_id, seed, generator_name, reward_name, candidate_library,
    scheduler, controller, num_rounds, batch_sizes, schedule, mask_fraction,
    epsilon_target, path_token_kl, schedule_policy_kl, kl_saturated,
    n_model_calls, n_reward_calls, wall_time_sec, planning_time_sec,
    terminal_reward, uncontrolled_reward, terminal_gain, success, validity,
    diversity_group, response_power_direct, gram_delta, gram_condition,
    gamma_pinv, gamma_ridge, leakage_rho, leakage_bound,
    theory_report_sha256

ScheduleScoreRecord 精确包含：

    example_id, seed, schedule_id, schedule, score_random, score_confidence,
    score_dependency, score_dprm_itemwise, score_diag_um, score_projection,
    score_residualized, score_budgeted, actual_response_power,
    actual_terminal_gain, frontier_width_histogram, gamma_pinv, gamma_ridge,
    gram_delta, ridge_multiplier, used_pinv

- [ ] **Step 5: 验证**

Run: cd feedback_frontier && python -m pytest tests/test_schedules.py tests/test_common_random_numbers.py -q

Expected: PASS。后续随机数只能来自 SeedBook。

---

### Task 3: 实现 product 与 tree-Potts 生成器

**Files:**

- Create: feedback_frontier/src/feedback_frontier/generators/{categorical_product,potts_tree}.py
- Test: feedback_frontier/tests/test_generators.py

**Interfaces:**

- conditional_marginals(observed) -> ndarray[d,q]；observed 行填 NaN
- sample_conditioned(observed, uniforms: ndarray[d]) -> ndarray[d]
- log_prob(x) -> float

- [ ] **Step 1: 写精确性红灯测试**

在 d=4,q=3 枚举全部状态，断言 exp(log_prob) 和为 1；固定 observed 后，把枚举条件边缘与实现比较，误差 <1e-10。

- [ ] **Step 2: 实现 product generator**

每位置 logits 从 seeded Normal 抽样后 stable softmax。sample_conditioned 只接收 uniforms 并 inverse-CDF；不得接收 RNG。

- [ ] **Step 3: 实现 topology/potentials**

chain edges 为相邻点；balanced_tree 中 i>0 的 parent 为 (i-1)//2。node logits 为 seeded Normal；edge matrix 为 seeded Normal 乘 coupling；coupling=0 时全零。

- [ ] **Step 4: 实现 log-space sum-product**

以 0 为 root 做 upward/downward messages；observed token 使用 one-hot log evidence。public methods 验证 token 范围和 shape。

- [ ] **Step 5: 绿灯**

新增 20,000 samples 的经验边缘测试，容差 0.02；相同 node logits 下 coupling=0 Potts 与 product log-prob 差 <1e-10。

Run: cd feedback_frontier && python -m pytest tests/test_generators.py -q

Expected: PASS under 10 seconds。

---

### Task 4: 实现 reward suite 与 candidate libraries

**Files:**

- Create: feedback_frontier/src/feedback_frontier/rewards/synthetic.py
- Create: feedback_frontier/src/feedback_frontier/candidates/libraries.py
- Test: feedback_frontier/tests/test_rewards_and_candidates.py

**Interfaces:**

- SyntheticReward.__call__(x) -> float；SyntheticReward.supports
- build_oracle_library(reward) -> CandidateLibrary
- build_structural_library(d, tree_edges) -> CandidateLibrary
- build_random_matched_library(structural_library, rng) -> CandidateLibrary
- build_nested_diagnostic_library(structural_library, oracle_library) -> CandidateLibrary
- CandidateLibrary 只冻结 groups/source；estimated_weights 在 Task 5 的 development-only cross-fitting 后附加

- [ ] **Step 1: 写四类 reward tests**

用固定 tiny arrays 验证 unary sum、pairwise lookup、modular indicator、mixed=delta*unary+epistatic；delta 固定轮换 {0,1e-3,1e-2,1e-1}。

- [ ] **Step 2: 实现 reward factory**

unary 使用 centered q-vector；pairwise 使用 tree edges 加 seeded overlap edges 和 centered q×q table；modular 选择 size 3–5 group/syndrome；mixed 加归一化 unary。用 4,096 fixed base samples 标准化；std<1e-8 抛错。

- [ ] **Step 3: 实现 Oracle-E**

返回真实 supports。另保存 analytic_oracle_weight：support term 在固定 base samples 上的 variance，归一化至 sum 1；它只用于 product-theorem ceiling 和 estimator calibration，不能泄漏给 Structural-E。

- [ ] **Step 4: 实现 Structural-E**

只从 tree edges 构造全部 edge、长度 2–4 connected paths 和 contiguous windows size 2–4；禁止读取 reward coefficient/syndrome。

- [ ] **Step 5: 实现 Random-Matched-E**

在同 size groups 上做 seeded swap search，完全匹配 Structural-E 的 group-size histogram 和 vertex-degree vector；10,000 swaps 后仍不匹配则抛 RuntimeError。

- [ ] **Step 6: 实现 discovery-gap diagnostic**

真实 Structural-E 通常不嵌套于 Oracle-E，因此不能直接套用理论的 deterministic regret decomposition。额外构造只用于诊断的 Nested-E = Structural-E ∩ Oracle-E，计算：

\[
\Delta_{\rm disc}=f_{\mathcal O}^*-f_{\mathcal N}^*,
\]

并验证 \(f_{\mathcal O}^*-f_{\mathcal O}(\hat c)\le\Delta_{\rm disc}+[f_{\mathcal N}^*-f_{\mathcal N}(\hat c)]\)。Structural-E 主结果继续独立报告，不能用 Nested-E 替代。

所有 Structural-E、Random-Matched-E 和 Nested-E 的 reward-relevant weights 都由 Task 5 的 cross-fitted b/F residual block energy 估计。library construction 本身不得访问 terminal labels。

Run: cd feedback_frontier && python -m pytest tests/test_rewards_and_candidates.py -q

Expected: PASS。

---

### Task 5: 实现 projection 与 residualization

**Files:**

- Create: feedback_frontier/src/feedback_frontier/features/path_scores.py
- Create: feedback_frontier/src/feedback_frontier/estimators/projection.py
- Test: feedback_frontier/tests/test_projection.py

**Interfaces:**

- path_score(schedule,state_trace,library) -> ScoreVector
- standardize_scores(scores,fit_indices) -> StandardizedScores
- fit_moments(scores,rewards) -> Moments(b,F)
- orthogonal_surrogate(b) -> float
- projection_energy(b,F,pinv_rtol) -> ProjectionResult
- ridge_projection_energy(b,F,ridge_multiplier) -> ProjectionResult
- residualized_marginal(b,F,selected,candidate,pinv_rtol) -> float
- crossfit_geometry(scores,rewards,fold_ids) -> CrossFitGeometry
- ideal_product_coefficients(schedule,reward,library) -> ndarray[p]

- [ ] **Step 1: 写 path-score 和 theorem tests**

对 candidate group e，先取其 final frontier \(J_e=B_{\ell_e^*}\cap e\)。每个 \(i\in J_e\) 和 category a=0,...,q-2 对应一个 categorical transition-score coordinate：

\[
\psi_{e,i,a}=\mathbf 1[X_i=a]-p_i(a\mid s_{\rm prebatch}).
\]

因此 width=1 的 pure product interaction 有单变量 final score block；width>1 时保留全部同时提交变量的 local directions，允许 projection 直接测量 correlation/leakage。测试断言每列在 50,000 个 base trajectories 上均值绝对值 <0.02；共享同一 (state,position,category) 的坐标数值必须完全相同。这一定义不读取 reward coefficient/syndrome，reward 只通过 \(b=Cov(R,\Psi)\) 进入。

    def test_duplicate_feature_does_not_double_count():
        b = np.array([1.0, 1.0])
        F = np.ones((2, 2))
        assert projection_energy(b, F, 1e-12).value == pytest.approx(1.0)
        assert orthogonal_surrogate(b) == pytest.approx(2.0)

另测可逆 reparameterization invariance、near-orthogonal bound、correlation-plus-leakage bound、Schur marginal 等于 full energy difference（误差 <1e-9）。

- [ ] **Step 2: 实现 development-only standardization**

先实现 path_score，column metadata 固定为 (group,frontier_position,category)。只用当前 reward instance 的 adaptation-set fit_indices 计算均值/标准差；std<1e-10 的坐标丢弃。adaptation cross-fit 的 validation folds 只能应用对应 fit folds 的 statistics，最终 gain-evaluation trajectories 不得参与任何标准化或矩估计。标准化后计算 delta=||F-I||op 和 condition number。

对 q=4 product reward，ideal_product_coefficients 直接从已知 centered reward term table、base categorical probabilities 和 unique-final-frontier schedule 解析计算 \(b^{(0)}\)；不得用同一 trajectories 回归出 \(b^{(0)}\)。用 mixed leakage delta grid 验证 rho 随 injected unary leakage 增大。

- [ ] **Step 3: 实现 pseudoinverse primary 与 ridge sensitivity**

fit_moments 拒绝 NaN/inf、N<2、shape mismatch，并对称化 F。理论主量始终用 scipy.linalg.pinvh(F,rtol=1e-12)。ridge 值为 multiplier*trace(F)/p，并单独返回。二者不得共用同一个输出字段。

- [ ] **Step 4: 实现 Schur residual gain**

使用 Moore–Penrose blocks 计算 \(\widetilde F_{E|A}\)、\(\widetilde b_{E|A}\) 和 marginal projection energy；只允许 1e-10 的负数被 clip。

- [ ] **Step 5: 实现 5-fold cross-fitting**

fold id 由 SeedBook 和 example_id 确定。每 fold 用另外四 folds 拟合 standardization/F/b/ridge，再给 held-out fold 的 candidate schedules 评分。最终 score 是五个 out-of-fold predictions 的 pooled value。ridge multiplier 只按 condition<=1e8 与 fold-to-fold reproducibility 选择，不能使用 final steering gain。

Run: cd feedback_frontier && python -m pytest tests/test_projection.py -q

Expected: path-score centering、无 leakage standardization、四项 theorem tests 和 cross-fit isolation 全部 PASS。

---

### Task 6: 实现 finite-budget response

**Files:**

- Create: feedback_frontier/src/feedback_frontier/estimators/response_power.py
- Test: feedback_frontier/tests/test_response_power.py

**Interfaces:** kl_rademacher_mean、inverse_kl_mean、binary_gain、binary_response_power、calibrate_width_weights、first_order_score、budgeted_score、balanced_random_budgeted_baseline。

- [ ] **Step 1: 写 monotonicity 和 tied-schedule tests**

对 epsilon grid 和 w=1,...,12 断言 \(g_1>g_2>\cdots>0\)，并与 committed theory JSON 中 w={1,2,3,4,6,10} 的精确值比较。构造 first-order tie 但 final-frontier width histogram 不同的 schedules，断言 budgeted 更偏好较窄者。

- [ ] **Step 2: 复用 exact binary theory**

\[
d(m)=\tfrac12[(1+m)\log(1+m)+(1-m)\log(1-m)],
\]

\[
g_w(\epsilon)=[d^{-1}(\epsilon/w)]^w,\qquad
\lambda_w(\epsilon)=g_w(\epsilon)^2.
\]

调用 feedback_frontier.theory 的已回归函数，不再实现第二套公式。epsilon>=w*log(2) 时 gain=1；width<1 或 epsilon<0 抛错。small-epsilon log-log slope 必须在 5e-4 内等于 w/2。

- [ ] **Step 3: 实现 probe calibration**

输入列固定为 epsilon,width,actual_response_power。q=2 theorem micro-test 直接使用解析 \(\lambda_w=g_w^2\)，禁止数据拟合。主实验的 q=4 product/Tree-Potts 按 epsilon/width 汇总 direct response probes，用 weighted isotonic regression 拟合 \(\lambda_1\ge\lambda_2\ge\cdots\ge0\)；不得强制 lambda1=1。缺 width=1 或每个 width 少于 20 probes 时标记 calibration_inconclusive，不允许无提示回退后进入 main。

- [ ] **Step 4: 实现 exact random-balanced baseline**

\[
P(W=w)=\frac{\sum_{\ell=1}^L
\binom{b_\ell}{w}\binom{B_{<\ell}}{k-w}}{\binom nk}.
\]

对每个 group size k 计算解析 baseline \(\sum_w\lambda_wP(W=w)\)。用 capacities=(3,3,3), k=5 的全部 1,680 schedules 枚举验证概率为 (0.3571428571,0.5,0.1428571429)，最大误差 <1e-12。

Run: cd feedback_frontier && python -m pytest tests/test_response_power.py -q

Expected: exact binary、isotonic calibration、balanced-width identity 全部 PASS。

---

### Task 7: 实现 non-reward schedulers

**Files:**

- Create: feedback_frontier/src/feedback_frontier/schedulers/non_reward.py
- Test: feedback_frontier/tests/test_schedules.py

**Methods:** random_balanced、confidence、entropy、dependency_cmi、min_within_batch_tc。

- [ ] **Step 1: 写合法性、确定性、no-leakage tests**

五种方法都返回合法 partition；相同 context 完全一致；替换 reward object 不改变结果。

- [ ] **Step 2: 实现 random/confidence/entropy**

random 用 registered tie RNG permutation；confidence 按 max probability 降序；entropy 按 entropy 升序；tie 用 position id。

- [ ] **Step 3: 实现 dependency matrix**

对 unresolved pair 枚举 q²，调用 exact conditional probability 计算 conditional MI；负数 clip 0。缓存 key 为 generator instance + canonical observed。

- [ ] **Step 4: 实现 dependency/min-TC**

dependency_cmi 从和 observed 总 CMI 最大位置开始，随后选与当前 batch 平均 CMI 最小者。min_within_batch_tc 穷举 batch，最小化 pairwise CMI sum。manifest 必须标为 pairwise TC surrogate。

Run: cd feedback_frontier && python -m pytest tests/test_schedules.py -q

Expected: PASS。

---

### Task 8: 实现 controller、DPRM 和 subset planners

**Files:**

- Create: feedback_frontier/src/feedback_frontier/controllers/rollout_local.py
- Create: feedback_frontier/src/feedback_frontier/schedulers/{dprm,subset}.py
- Test: feedback_frontier/tests/test_planners.py

**Interfaces:** RolloutCache.action_values、calibrate_alpha、BatchValueOracle.value；methods 为 dprm_myopic、dprm_rollout、subset_exact、subset_beam_8、subset_beam_32。

- [ ] **Step 1: 写 KL/counter tests**

常数 value 得到 alpha=0, KL=0, saturated=true；可达 KL 误差 <1e-6。重复 query 命中 cache，counters 不增长。

- [ ] **Step 2: 实现 shared rollout cache**

每个 (example,state,position,token,k) 使用 rollout namespace uniforms；cache key 不含 scheduler。一次 completion 计 model call，一次 reward 计 reward call。

- [ ] **Step 3: 实现 controller**

batch 内所有位置基于相同 pre-batch state 求 base probabilities/action values，分别校准 alpha，再用 base uniforms inverse-CDF sampling。

- [ ] **Step 4: 实现 itemwise DPRM**

统一 process value 为：

\[
R_t^\star(a;s)=\beta^{-1}\log\mathbb E[\exp(\beta R(X_T))\mid s,a].
\]

用 stable logmeanexp 实现，beta->0 的测试极限为 reward mean。dprm_myopic 对每个位置仅评价“本轮 reveal 该位置，之后 uncontrolled confidence completion”的 process value；dprm_rollout 评价“本轮 reveal 后，后续 rounds 继续使用相同 local controller 和 confidence schedule”的 process value。二者都对 position 做 itemwise score 后 top-capacity，不进行 subset reranking。

- [ ] **Step 5: 实现 exact/beam subset**

exact 枚举 capacity-sized subsets，并用与 SAPS 完全相同的 downstream controller/future feedback 查询 BatchValueOracle；因此它是 Bellman-compatible planning upper bound。d>14 抛错。beam 从空 set 逐层扩展、去重，保留 8/32，最终同 oracle 精排。

- [ ] **Step 6: 验证 optimality/gap**

d=6,m=3 direct enumeration 等于 exact planner；beam32 在 tiny case 等于 exact。另按 math_theory 实现两个无 unary 信息差异的构造：

- regular cycle C_2m：所有 itemwise scores 相同，balanced optimum probability 必须等于 \(2/\binom{2m}{m}\)；
- syndrome n=2k：所有 singleton committors 相同，frontier-optimal batch probability 必须等于 \(k^2/\binom{2k}{k}\)。

测试 m=6、k=6 的 direct enumeration。报告只声称 itemwise top-m 无法表示 batch synergy；不得声称 exact subset-DPRM 看不到 frontier。

Run: cd feedback_frontier && python -m pytest tests/test_planners.py -q

Expected: PASS。

---

### Task 9: 实现结构化 solver 与 P/B-SAPS

**Files:**

- Create: feedback_frontier/src/feedback_frontier/schedulers/saps.py
- Create: feedback_frontier/src/feedback_frontier/schedulers/structured.py
- Test: feedback_frontier/tests/test_planners.py

**Methods:** saps_diagonal、p_saps_residualized、p_saps_projection、b_saps_budgeted；solvers 为 pairwise_milp、laminar_dp、treewidth_path_dp、conditional_expectation_swaps。

- [ ] **Step 1: 写 objective oracle tests**

在 d=6,L=3 穷举 balanced schedules，按 final-frontier width 手算 first-order/budgeted score。duplicate feature case 中 projection 不 double count。额外复现 committed theory results 的 pairwise reduction、8-leaf laminar DP 和 9-node path DP gaps。

- [ ] **Step 2: 实现 pairwise capacitated Max-L-Cut MILP**

使用 scipy.optimize.milp。binary x[i,l] 表示 token i 的 round；约束每个 i 恰有一个颜色、每个 l 恰好 capacity[l]。为每个 edge/color 线性化 z[i,j,l]=x[i,l]x[j,l]，最大化 \(\sum_{ij}W_{ij}(1-\sum_lz_{ij,l})\)。在 d<=10 上与全部 balanced colorings 枚举一致，objective gap <1e-9。

- [ ] **Step 3: 实现 laminar 与 treewidth-one DP**

laminar state 为 subtree 的 capacity count vector；child tables 做 capacity-bounded convolution，并在 internal node 加 \(W_u\lambda_{w(counts)}\)，其中 width 是最后非零 color 的 count。path/treewidth-one state 为 (used_counts,last_color)。分别与 math_theory verified tiny instances 的 brute optimum 比较，gap <1e-10。

- [ ] **Step 4: 实现 general-hypergraph CE + swaps**

初始 assignment 用 exact balanced random width distribution 做 conditional expectation，逐 token/round 固定并保持剩余 capacity 可行；随后枚举所有跨 batch token swaps，接受严格改善，最多 100 sweeps。若一整 sweep 无改善即停止。这个 solver 只继承 deterministic random-baseline guarantee，不声明通用 approximation ratio。

- [ ] **Step 5: 路由 solver**

- pairwise library：pairwise_milp；
- laminar library：laminar_dp；
- tree/path pairwise library：treewidth_path_dp；
- 其余 overlap hypergraph：conditional_expectation_swaps。

每个 solver 返回 objective、optimality_status、lower_bound、upper_bound、planning_time、state/action evaluations。solver failure 不得静默切换；仅 smoke 可在 manifest 标明后使用 exhaustive tiny fallback。

- [ ] **Step 6: 实现 P/B-SAPS objectives**

- diagonal：\(\sum_e W_e1[w(e)=1]\)
- residualized：按 (width,group) 顺序累计 Schur marginal
- projection：用 cross-fitted \(\Gamma_1^2(c)\) 对 structured solver shortlist rerank
- budgeted：\(\sum_e W_e\lambda_{w(e)}(\epsilon)\)

structured solver 产生最多 128 个 schedules，加 non-reward/DPRM schedules。强相关 regime（delta>=0.25）还必须用 direct \(\widehat{\mathcal S}_\epsilon(c)\) rerank top 16。tie 先选 objective 较高、再选 latency 较低、最后选 serialized schedule 字典序较小者。

- [ ] **Step 7: 记录 cost**

记录 planning_time_sec、proposal_count、linear_solve_count、MILP nodes/DP states/action evaluations；feature arithmetic 不伪装成 model/reward calls。

Run: cd feedback_frontier && python -m pytest tests/test_planners.py -q

Expected: PASS。

---

### Task 10: 实现 paired runner 与原子输出

**Files:**

- Create: feedback_frontier/src/feedback_frontier/runners/{probe_response,evaluate_planner}.py
- Modify: feedback_frontier/src/feedback_frontier/cli.py
- Test: feedback_frontier/tests/{test_common_random_numbers,test_smoke_contract}.py

**Raw outputs:** schedule_scores.csv、trajectory_results.parquet、response_probes.parquet、experiment_manifest.json。

- [ ] **Step 1: 写 1-instance integration test**

运行 d=6,q=3,instances=1,methods=[random_balanced,subset_exact]；断言四个 raw files 存在、schema 精确匹配、两方法 base/proposal/rollout fingerprints 相同。

- [ ] **Step 2: 实现两阶段 runner**

Stage A 使用 outer development reward-instance shard 冻结 ridge rule、response-probe width weights 和其他全局选择。Stage B 处理 outer held-out reward instances：对每个 test reward instance 和 candidate schedule，先从独立 adaptation RNG domain 生成固定 `adaptation_trajectories` 条 terminal-label trajectories，估计 instance-specific \(b_{r,c}/F_{r,c}\) 并完成 schedule selection；随后只在 gain-evaluation RNG domain 上测 terminal gain。adaptation 与 gain evaluation 的 sample id、uniforms、terminal labels 和缓存键必须完全不相交。先按 (generator regime,reward,seed) 分层，再把每层按 canonical instance id 排序：偶数 rank 为 development，奇数 rank 为 held-out。

- [ ] **Step 3: 实现 direct finite-budget response probe**

对每个 candidate schedule：

1. 在 standardized Fisher 的正特征值子空间采样 z~Normal 并归一化；
2. 令 \(u=F^{\dagger/2}z\)，把 u 对应到各 batch categorical logit directions；
3. 用 bisection 缩放 u，使 trajectory 上 conditional token KL 之和等于 epsilon，误差 <1e-5；
4. 用 paired common-random-number rollouts 估计 \(G_c(V,\epsilon)=E_QR-E_PR\)；
5. 对 config 中 8/16/64 个 directions 求 mean(G²)，写 response_power_direct。

方向 RNG key 只含 example、schedule、epsilon、direction_id，不含 method。rank=0 的 schedule 记录 response_space_empty=true 而不是除零。

- [ ] **Step 4: 实现 paired execution**

循环固定为 instance -> L -> epsilon -> method。method loop 前创建 shared caches/fingerprints；每个 method 从 immutable initial state 开始。

- [ ] **Step 5: 实现 counters/success**

success 为 terminal reward 高于同 reward instance 的独立 adaptation-set uncontrolled reward 75th percentile；该阈值不能读取 gain-evaluation rewards。validity=1.0；diversity_group 为 sequence stable hash prefix。

- [ ] **Step 6: 实现原子输出**

先写 .staging/；schema/row-count 验证后 Path.replace。manifest 包含 UTC、版本、完整 config、cross-fit folds、pinv tolerance、ridge sensitivity、width weights、seed version、两个 theory SHA256、row counts、artifact SHA256、status；无 Git 时 commit=null。理论 hash 不匹配时在任何 rollout 前失败。异常只留 failure.json。

manifest 还必须冻结 `evaluation_protocol.name=held-out_reward_instance_fixed-budget_few-shot_adaptation`、`zero_shot=false`、每 schedule adaptation budget、总预算计费规则，以及 adaptation/gain RNG domain。trajectory 和 schedule-score raw tables 必须逐行记录 `adaptation_terminal_labels`。

Run: cd feedback_frontier && python -m pytest tests/test_common_random_numbers.py tests/test_smoke_contract.py -q

Expected: PASS；pytest temp 目录外无 output。

---

### Task 11: 实现统计、图与 gate report

**Files:**

- Create: feedback_frontier/src/feedback_frontier/analysis/{aggregate,bootstrap,figures}.py
- Modify: feedback_frontier/src/feedback_frontier/cli.py
- Test: feedback_frontier/tests/test_analysis.py

- [ ] **Step 1: 写 paired bootstrap test**

toy dataframe 下 10,000 resamples 的 CI 可复现。resampling unit 是 example_id，同 example 的所有 method/epsilon/L 一起抽。

- [ ] **Step 2: 实现 metrics**

held-out rows 计算 Pearson、Spearman、top-1 regret、top-5 recall、paired mean/median、win rate、standardized effect size。

\[
ApproxRatio=(V_{method}-V_{random})/(V_{exact}-V_{random}).
\]

分母绝对值 <1e-8 时记 NaN 并报告数量。另计算：

- Gram diagnostics：delta、condition number，以及 observed \(W/\Gamma\) 是否落入 near-orthogonal sandwich；
- product leakage：rho 与 combined bound；Tree-Potts 明确显示 unavailable；
- surrogate guarantee：在 exhaustive tiny benchmark 上取所有 feasible schedules 的 \(\delta_{\max}\)，验证 weighted schedule 的 \(\Gamma\) ratio 不低于 \((1-\delta_{\max})/(1+\delta_{\max})\)。main 只能对 evaluated candidate pool 报 empirical diagnostic，不能把 pool 内最大 delta 当成全 feasible class 的 theorem guarantee；
- discovery：Nested-E 的 \(\Delta_{\rm disc}\) 与 regret decomposition；真实 Structural-E 单独报告 recovery，不套 nested bound；
- exact random baseline：解析 balanced-width expectation 与 empirical random mean 的差；
- frontier：exact-L points，以及对 L 取 cumulative maximum 的 at-most-L nested envelope \(\eta_{\rm proj}^*(L)=\max\Gamma_1^2/Var(R)\) 和 \(\mathcal S_\epsilon^*(L)\)。

frontier AUC 在 measured latency 归一化后 trapezoid integration；Structural recovery 同样处理零分母。H1 另做预注册 multivariable regression/partial Spearman：actual gain 对 projection score，在控制 confidence、entropy、dependency/TC、L、epsilon 后 projection coefficient 的 bootstrap CI 必须报告。

- [ ] **Step 3: 实现 figures**

- projection_vs_gain：weighted UM、gamma_pinv、gamma_ridge、direct response power panels，并按 gram-delta regime 分面
- planner_value_vs_cost：model/reward calls 与 planning seconds panels
- finite_budget_crossover：epsilon 对 b_saps_budgeted-minus-saps_diagonal paired gain
- latency_frontier：exact-L 与 at-most-L envelope；x 分别为 rounds、NFE、wall time，y 分别为 eta_proj、direct response power、actual reward

PNG dpi=200，同时输出 PDF；analysis 不覆盖 raw artifacts。

- [ ] **Step 4: 实现 machine-readable gates**

每个 gate 写 metric、threshold、estimate、ci_low、ci_high、pass、reason、theory_assumption_status。全部主 gate pass 为 GO；任一 fail 为 NO_GO；样本、分母、calibration 或 theory hash 不足为 INCONCLUSIVE。near-orthogonal guarantee 只在 exhaustive feasible class 的 delta_max<1 时评价；main candidate-pool diagnostics 不计为 theorem proof/violation。

Run: cd feedback_frontier && python -m pytest tests/test_analysis.py -q

Expected: PASS。

---

### Task 12: 全量测试与机制 smoke

- [ ] **Step 1: 全量 tests**

Run: cd feedback_frontier && python -m pytest -q

Expected: PASS；theory-regression、planner、CRN、schema、integration tests 无 skip。

- [ ] **Step 2: 执行 smoke**

Run: cd feedback_frontier && feedback-frontier validate-config --config configs/smoke.yaml

Expected: valid: d=12 q=4 instances=16。

Run: cd feedback_frontier && feedback-frontier run --config configs/smoke.yaml --run-id phase1-smoke-v1

Expected: manifest status=complete；raw files 非空；无 NaN reward、非法 schedule 或 seed mismatch。

- [ ] **Step 3: 分析 smoke**

Run: cd feedback_frontier && feedback-frontier analyze --run-dir outputs/phase1-smoke-v1

Expected: 四张 PNG/PDF 和 metrics、bootstrap、gate reports 全部生成。

- [ ] **Step 4: 六个机制 gate**

1. math_theory hashes、duplicate/reparameterization、Schur、binary finite-KL、balanced-width、pairwise/DP regressions 全部通过；
2. 至少一个 regular-cycle 或 syndrome instance 上 exact subset batch 不同于 itemwise DPRM，且 exact value 更高；
3. 至少一个 first-order tied pair 被 exact/calibrated finite-budget objective 区分；
4. 小 epsilon 下 direct response power 与 \((2\epsilon/p_c)\Gamma_1^2\) 的 ratio 95% CI 包含 1；
5. unary reward 上各 method 相对 random 的 95% CI 均包含 0；
6. exact balanced random baseline 与 empirical random mean 的误差在 Monte Carlo 95% CI 内。

任一失败：停止，不运行 screening。smoke 验收机制链，不验收 SAPS 获胜。

---

### Task 13: Screening、冻结与 Phase 1 main

- [ ] **Step 1: 跑 screening**

Run: cd feedback_frontier && feedback-frontier run --config configs/screening.yaml --run-id phase1-screening-v1

Run: cd feedback_frontier && feedback-frontier analyze --run-dir outputs/phase1-screening-v1

Expected: complete artifacts 和 gate report。

- [ ] **Step 2: 应用 method freeze**

main 强制保留 random_balanced、confidence、entropy、dependency_cmi、min_within_batch_tc、dprm_myopic、dprm_rollout、subset_exact、saps_diagonal、p_saps_residualized、p_saps_projection、b_saps_budgeted。beam 8/32 仅当 screening approximation ratio 比 itemwise 高 >=0.05 时保留；理由写入 freeze_reason。

- [ ] **Step 3: 冻结 inputs**

pinv tolerance、cross-fit fold assignment、ridge sensitivity rule、width weights、method list、success thresholds、development ids 和 theory artifacts 的 SHA256 写入 main config。main 启动时重算；任一 hash mismatch 即拒绝。

- [ ] **Step 4: 跑 main**

Run: cd feedback_frontier && feedback-frontier run --config configs/phase1_main.yaml --run-id phase1-main-v1

Expected: complete manifest；instances=256、seeds={0,1,2}；required methods 的 (example,L,epsilon) key coverage 完全相同。

- [ ] **Step 5: 生成 decision**

Run: cd feedback_frontier && feedback-frontier analyze --run-dir outputs/phase1-main-v1

Expected: decision 只由 Section 0.6 阈值产生。

- [ ] **Step 6: 收口**

- GO：另写 DPLM plan；本计划不接 DPLM。
- NO_GO：H1 fail 删除 correlated-score bridge claim；H2 fail 将结果收缩为 exact-subset diagnostic；H3 fail 退回 infinitesimal projection theory；H4 fail 停止 latency-frontier framing。
- INCONCLUSIVE：只增加 seeds/examples 或修复缺失数据；不得改 objective 后沿用 run id。

---

## 2. 产物合同

每个成功 run 最终包含：

    schedule_scores.csv
    trajectory_results.parquet
    response_probes.parquet
    experiment_manifest.json
    metrics.csv
    bootstrap_intervals.csv
    gate_report.json
    gate_report.md
    projection_vs_gain.{png,pdf}
    planner_value_vs_cost.{png,pdf}
    finite_budget_crossover.{png,pdf}
    latency_frontier.{png,pdf}

schedule_scores.csv 一行对应 candidate schedule；trajectory_results.parquet 一行对应 method trajectory。两表通过 example_id、seed、schedule_id 连接。禁止只保存聚合结果。

## 3. 执行顺序与停止规则

    Task 0      理论 hash 与 theorem regressions
         ↓
    Tasks 1–4   数据与契约
         ↓
    Tasks 5–6   数学 estimator
         ↓
    Tasks 7–9   scheduler/controller/planner
         ↓
    Tasks 10–11 runner 与 analysis
         ↓
    Task 12     tests → smoke → 六项机制 gate
         ↓ pass only
    Task 13     screening → freeze → main → Go/No-Go

立即停止条件：

- theorem、CRN、schedule capacity test 任一失败；
- manifest schema/artifact hash 失败；
- paired method key coverage 不一致；
- exact subset 与 tiny direct enumeration 不一致；
- smoke 六项机制验收任一失败。

## 4. 需求覆盖矩阵

| 原稿要求 | 对应任务 |
|---|---|
| math_theory hashes、定义和已验证数值回归 | Task 0 |
| Product + correlated tree-Potts | Task 3 |
| unary/pairwise/modular/mixed/overlap | Task 4 |
| Oracle/Structural/Random-Matched E | Task 4 |
| objective-independent path-score features | Task 5 |
| standardized Gram、pinv projection、cross-fitting | Task 5 |
| exact binary response 与 balanced random baseline | Tasks 0,6 |
| random/confidence/entropy/dependency/TC | Task 7 |
| myopic/rollout DPRM、exact/beam subset | Task 8 |
| regular-cycle/syndrome itemwise gap | Task 8 |
| Max-L-Cut、laminar DP、treewidth DP、CE+swaps | Task 9 |
| diagonal/P-SAPS/B-SAPS | Tasks 5,6,9 |
| theorem/CRN/capacity tests | Tasks 2,3,5,6,8,12 |
| matched KL、direct response power、分离成本 | Tasks 8,10 |
| paired bootstrap、CI、effect size | Task 11 |
| projection/response/cost/frontier 四张核心图 | Task 11 |
| smoke grid 与原始产物 | Task 12、产物合同 |
| Go/No-Go | Tasks 11,13 |

明确延期：shared-hidden masked Transformer、DPLM/Cas9-DPLM、MHC/PLM naturalness、full-length protein、code repair。它们必须在 GO 后分别形成新 plan。

## 5. 完成定义

- Tasks 0–13 全部打勾；
- python -m pytest -q 全绿且关键测试无 skip；
- smoke、screening、main 均有 complete manifest 和完整产物；
- main 全部方法通过 paired key/seed fingerprint 检查；
- gate_report.json 给出可由 raw rows 重算的 GO、NO_GO 或 INCONCLUSIVE；
- README 包含从空环境安装、复现三个 run、重做 analysis 的精确命令；
- package 中仍无 DPLM 代码或配置。
