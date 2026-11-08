# G3 — Carbon-aware quantization / sparsity allocation

**Date:** 8 November 2026.
**Status:** **KILLED BY PILOT at 0.5B/RTX 3060 (8 Nov).** The pre-pilot repaired score was **7.875 / 7.975**; the pilot fired K1 (NF4 uses 15% more energy than FP16 on Qwen) and K4 (held-out additive \(R^2\) 0.506/−0.99), and the pass comparison is unreachable (no mix at \(0.8\times\)FP16). See `pilots/03_g3_pilot.md`. The original knapsack/greedy framing was already **6.75 / 7.05** and is also dead.
**Scores:** Pre-pilot Core **7.875** / Weighted **7.975** / Independence **7.5/10** (repaired). **Executable verdict: KILL.**

---

## 0. ≤300-word summary

**Verdict: CONDITIONAL.** The original G3 — “max accuracy s.t. $\sum_l E_l(b_l,s_l)\le B$,” with a knapsack/integrality-gap or submodular greedy $(1-1/e)$ theorem — is a 6.75-Core idea. Layer-wise bit/sparsity allocation under a resource budget is crowded (SigmaQuant 2602.22136, LieQ 2508.03332, PolyQ 2607.14618), and the combinatorial problem is a multiple-choice knapsack with $L\le32$, $|C_l|\le10$, exactly solvable by DP in milliseconds; an integrality-gap or greedy theorem is therefore not decision-changing. **No 2024–2026 paper has the same estimand + finite-sample guarantee**, so this is not a literal KILL. The defensible repair makes **measurement noise and non-separability** the core: define the certified frontier $A^*_\delta(B)$; prove a price-of-noise rate $O(\lambda^*\sigma\sqrt{\log(1/\delta)/n})$ with a matching $\Omega(\lambda^*\sigma/\sqrt{n})$ lower bound, where $\lambda^*$ is the shadow price of a joule; give a counterexample showing layer-separable accuracy surrogates can be off by $\Omega(1)$. Pilot (≤2 h, 3060): Qwen2.5-0.5B / Pythia-410M, FP16/INT8/NF4, ARC-Easy + MMLU 200 each, `nvidia-smi` at 100 ms; kill if energy headroom <10%, accuracy spread <1 point, CV >20%, additive-energy $R^2<0.8$, or Spearman(energy, latency) >0.95. Scores (repaired): Nov 8.0, Method 8.0, Math 8.0, Exp 7.5, Data 8.0, Q1 8.0, Feas 8.0, Compute 8.5 → Core 7.875, Weighted 7.975, Independence 7.5/10. Path: pilot; if K1–K5 pass and the gain is ≥3 points at $B=0.8\times$FP16, run the full study and target 8.5; if energy ≈ latency, pivot to carbon-intensity-conditioned profiles; if a same-estimand paper appears, kill or pivot.

---

## 1. Plain-language problem

A small LLM can be compressed unevenly: some layers tolerate 2-bit weights or 50% sparsity, others break. The practical question is not “what bit-width is best?” but “given a fixed energy or carbon budget per query, which layers should get the bits and which should get the sparsity?” Today’s answer is a grid search over a few uniform configurations, or a per-layer sensitivity heuristic with no guarantee that the resulting model actually fits the budget when the energy measurement itself is noisy. We formalize the allocation as a **certified budget** problem, show what the noise costs, and test whether the optimal allocation beats uniform compression at the same measured joules.

**Decision it changes.** A deployment engineer choosing per-layer compression for a fixed gCO2e/query budget; and an experimenter deciding how many energy repetitions and validation questions are worth spending to certify that budget.

---

## 2. Exact estimand/functional

### 2.1 Action space

Let $M$ be a decoder-only LM with $L$ layers. For layer $\ell$, choose
$$
c_\ell=(b_\ell,s_\ell)\in C_\ell,\qquad
b_\ell\in\{2,3,4,8,16\},\quad s_\ell\in\{0,0.5\},
$$
where $b_\ell$ is weight bit-width (activations fixed at 16-bit for the main study; an activation-bit arm is a secondary experiment) and $s_\ell$ is structured 2:4 sparsity (dense vs. 2:4). Thus $|C_\ell|=10$. A global allocation is $x=(c_1,\dots,c_L)\in C=\prod_\ell C_\ell$.

The action space deliberately excludes unstructured sparsity: on an RTX 3060 (Ampere), unstructured sparsity does not reliably save energy because the dense kernels still run; 2:4 structured sparsity is the hardware-supported action. This is a measurement-driven choice, not an arbitrary restriction.

### 2.2 Energy model and noise

For a workload $w=(\text{prompt length},\text{output length},\text{batch size})$, the mean energy is
$$
\mu(x,w)=\mu_0(w)+\sum_{\ell=1}^{L}\mu_\ell(c_\ell,w)+\rho(x,w),
$$
where $\mu_0$ is the fixed overhead (embedding, LM head, kernel launches), $\mu_\ell$ is the layer’s marginal energy, and $\rho$ is the interaction residual (kernel fusion, memory reuse, clock effects). The **additive approximation** sets $\rho=0$; the paper measures $\rho$ on held-out allocations and reports $|\rho|$ relative to $B$. If $|\rho|$ is not negligible, the problem is a **quadratic knapsack**, not a linear one.

For each layer–configuration pair $(\ell,c)$, we run $n_{\ell,c}$ repetitions and observe
$$
\widehat E_{\ell,c}=\frac1{n_{\ell,c}}\sum_{i=1}^{n_{\ell,c}}E_{\ell,c}(\xi_i),
\qquad \mathbb E[\widehat E_{\ell,c}]=\mu_\ell(c),
$$
with centered noise assumed $\sigma_{\ell,c}$-sub-Gaussian. (On a GPU, power samples are autocorrelated; the effective $n$ is smaller. The protocol uses batch means or a block bootstrap to estimate $\sigma_{\ell,c}$ honestly.) The one-sided confidence radius is
$$
\varepsilon_{\ell,c}(\delta)=\sigma_{\ell,c}\sqrt{\frac{2\log(2L|C_\ell|/\delta)}{n_{\ell,c}}}.
$$
The union bound is over $\sum_\ell |C_\ell|\approx280$ pairs, not over the $\prod_\ell|C_\ell|$ global allocations. This is the technical payoff of additivity: the certificate scales with the number of layer–configuration pairs, not with the combinatorial explosion.

### 2.3 Accuracy, oracle frontier, shadow price

Let
$$
A(x)=\mathbb P_{(q,y)\sim D}\big[\arg\max_{y'}M_x(y'\mid q)=y\big]
$$
be the expected accuracy on the deployment distribution $D$ (MMLU/ARC for this paper). Accuracy is **not** separable across layers; define the interaction residual
$$
\iota(x)=A(x)-A_0-\sum_\ell a_\ell(c_\ell),
$$
where $a_\ell(c_\ell)$ is the single-layer marginal effect measured by ablation and $A_0$ is the full-precision accuracy. The paper estimates $\iota$ on a calibration set and tests whether it is negligible.

The oracle problem is
$$
A^*(B;w)=\max_{x\in C}\ A(x)\quad\text{s.t.}\quad \mu(x,w)\le B,
$$
and the **frontier** is $B\mapsto A^*(B;w)$. The **shadow price of a joule** is
$$
\lambda^*(B;w)=\frac{\partial A^*}{\partial B}
$$
where differentiable; it is the accuracy gained per extra joule, and the conversion rate between measurement noise and accuracy loss.

### 2.4 Certified problem

The **certified feasible set** is
$$
C_\delta(B)=\Big\{x\in C:\ \widehat\mu_0+\sum_\ell\widehat\mu_\ell(c_\ell)+\sum_\ell\varepsilon_{\ell,c_\ell}\le B\Big\},
$$
and the **certified allocation** is
$$
\widehat x=\arg\max_{x\in C_\delta(B)}\widehat A(x),
$$
where $\widehat A$ is the validation-set accuracy estimate with uniform error $\varepsilon_A$ (Hoeffding or a VC bound). The **certified frontier** is
$$
A^*_\delta(B)=\max_{x\in C_\delta(B)}A(x),
\qquad
\Pi_\delta(B)=A^*(B)-A^*_\delta(B),
$$
where $\Pi_\delta$ is the **price of certification**.

The estimand is the function
$$
B\ \longmapsto\ \big(A^*(B),\ A^*_\delta(B),\ \lambda^*(B)\big),
$$
not a single allocation. The decision variables include the measurement counts $n_{\ell,c}$ and the validation size $n_{\rm val}$, not only $(b_\ell,s_\ell)$.

### 2.5 What is genuinely new

1. The estimand is the **certified frontier**, not a point on an empirical Pareto curve.
2. The guarantee is **finite-sample feasibility plus near-optimality**, not “our heuristic beats baselines.”
3. The theory identifies $\lambda^*$ as the **price of a joule**, and therefore the price of measurement noise: $\Pi_\delta\approx 2\lambda^*\varepsilon_n$.
4. The experiment-design problem (how many measurements per layer, how many validation questions) is part of the optimization, not an afterthought.
5. The paper confronts **non-separability** directly: per-layer sensitivity proxies are not sufficient, and the interaction residual is measured rather than assumed away.

### 2.6 What is *not* new (and must not be claimed)

- Layer-wise bit allocation under a resource budget is not new (SigmaQuant, LieQ, PolyQ, HAWQ-V3, DSA, LAMP).
- Exact DP for a multiple-choice knapsack is textbook.
- The LP dual/shadow price is textbook.
- “Carbon-aware” with a fixed regional carbon intensity is just energy-constrained allocation with a constant multiplier. The carbon-specific contribution requires either time-varying carbon intensity or multiple regions; the static experiment is honest about this.

---

## 3. Closest 5 papers 2024–2026 at the functional level

**Search date:** 8 November 2026. **Sources:** arXiv API/search, arXiv abstract pages, keyless web search. Tavily quota was exhausted; some backends were degraded (1/6 at times). “Verified” below means the abstract page was read on 2026-11-08; venue claims are from the arXiv comments/journal-ref field and are marked separately.

| # | Paper | ID / venue | Functional (what it optimizes) | Guarantee | Same estimand + guarantee? |
|---|-------|------------|-------------------------------|-----------|----------------------------|
| 1 | **SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference** — Liu, Yu, Zapater, Atienza | **arXiv:2602.22136** (v2, 3 Mar 2026; preprint) | Layer-wise heterogeneous bit-width allocation under user-specified **memory and accuracy targets**; two-phase distribution/KL search; energy is evaluated post hoc on a shift-add accelerator | Empirical; no finite-sample guarantee; no noise model | **No.** Closest overall. It *mentions* an energy budget in the abstract, but the optimization targets are memory/accuracy; energy is a reported hardware metric, not a hard constraint. |
| 2 | **Exploring Layer-wise Information Effectiveness for Post-Training Quantization in Small Language Models (LieQ)** — Xiao et al. | **arXiv:2508.03332** (v2, 27 Dec 2025; preprint) | Layer-wise bit allocation for sub-8B LMs under a **target average-bit budget**; geometry-driven sensitivity proxy; closed-form allocation for 2/3/4-bit settings | Empirical/closed-form; no energy constraint; no guarantee | **No.** Budget is bits, not joules. “Energy footprint” is motivation, not a constraint. |
| 3 | **PolyQ: Codesigning End-to-End Quantization Framework for Scalable Edge CPU LLM Inference** — Oh et al. | **arXiv:2607.14618**; ICCAD 2026 | Channel-wise bit allocation from {2,3,4,8,16} under a **user-specified average-bit budget**; compiler layout regularization; reports energy/token overhead below 2% | Empirical; no finite-sample guarantee; no energy budget | **No.** Budget is average bits; energy is measured, not constrained. The closest LLM-specific allocation paper. |
| 4 | **Towards Green AI: Evaluating Energy Efficiency for Quantized LLM Inference** — Basta | **arXiv:2610.17531** (13 Oct 2026; preprint) | Measurement study of 10 PTQ strategies on 5 LLMs and 4 GPUs; energy, carbon intensity, accuracy, runtime across input/output lengths and batch sizes | Empirical benchmark; no allocation; no guarantee | **No.** It is the measurement *input* to G3, not an allocation method. Critically, it finds **energy closely tracks runtime**, which threatens the distinctness of an energy budget (see §6.3, K5). |
| 5 | **Carbon-Taxed Transformers: A Green Compression Pipeline for Overgrown Language Models (CTT)** — Alam, Roy, Roy, Roy, Schneider | **arXiv:2604.25903**; FSE 2026, Proc. ACM Softw. Eng. 3, Article FSE047; DOI 10.1145/3797075 | Multi-architectural compression pipeline ordering (pruning/quantization/distillation) guided by a **computational carbon-tax penalty**; SE tasks (clone detection, summarization, code generation) | Empirical; up to 81% CO2 reduction, 49× memory reduction; no hard per-query carbon budget; no theorem | **No.** A pipeline-level heuristic, not layer-wise allocation; the “tax” is a penalty, not a constrained budget. |

### 3.1 Near neighbors also checked

- **Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization** — arXiv:2607.22583, ICTAI 2026 (verified abstract). Layer-wise pruning ratios under **latency** constraints via parallel Bayesian optimization; energy is motivation, not constraint. Covers the $s_\ell$ half of G3 but not the energy guarantee.
- **KVTuner** — arXiv:2502.04420, ICML 2025 (verified abstract). Layer-wise mixed-precision **KV-cache** allocation with multi-objective (throughput/latency) search; not weight/activation energy.
- **HAWQ-V3** — arXiv:2011.10680, ICML 2021 (verified abstract; precursor outside 2024–2026). Hardware-aware mixed-precision bit allocation with an integer program under model-size/latency constraints. The direct ancestor of the “bit allocation under a budget” formulation.
- **DSA: Differentiable Sparsity Allocation** — arXiv:2004.02164 (2020; verified abstract; precursor). Layer-wise sparsity allocation under a resource budget; gradient-based, no guarantee.
- **LAMP: Layer-adaptive sparsity for magnitude-based pruning** — arXiv:2010.07611, ICLR 2021 (verified abstract; precursor). Layer-wise sparsity scores; no energy budget.
- **Can We Optimize the Performance-Carbon Emission Break-Even Point?** — arXiv:2608.08744 (2026; abstract verified in part). Carbon-aware fine-tuning objective with a differentiable energy surrogate; not layer-wise inference allocation.
- **Dense Structural Compression of Transformers via Gauge-Correct Channel Removal** — arXiv:2609.07264, NeurIPS 2026 (verified abstract). Training-time structural compression to maximize inference utility per unit compute; not post-hoc mixed precision.
- Additional 2026 bit-allocation methods found but not read in full: FAMPWQ (arXiv:2608.24945), RAMP (arXiv:2603.17891), BitsMoE (arXiv:2606.00079), MODE (arXiv:2606.17118). All allocate bits under memory/size/cost budgets; none constrains measured energy or proves a finite-sample guarantee.

### 3.2 KILL check

**No 2024–2026 paper has the same estimand + guarantee.** The closest is SigmaQuant: same layer-wise $b_\ell$ object and a resource budget that *includes* energy, but its optimization targets memory/accuracy, its energy evaluation is post hoc, and it offers no finite-sample certificate, no measurement-noise model, and no non-separability analysis. LieQ and PolyQ constrain average bits; Towards Green AI measures but does not allocate; CTT applies a carbon-tax penalty to a pipeline ordering.

**Therefore: not a literal KILL. But the original formulation is a 6.5–7, not an 8+.** “Energy instead of bits” is a thin delta; the repair is mandatory.

### 3.3 Nearest defensible repair if a same-estimand paper appears

1. **Carbon-intensity-conditioned profiles.** Precompute $K$ compression profiles (e.g., three budgets); choose profile $p(t)$ from the grid carbon intensity $c(t)$ under a long-run carbon budget $\sum_t c(t)E_{p(t)}\le B_{\rm CO_2}$. The estimand is the carbon-constrained frontier over profiles, and the theorem is a threshold policy in $c(t)$ with a regret bound against a clairvoyant profile sequence. This is distinct from G1 (model cascade) because the action is compression, and from static G3 because the carbon signal is time-varying.
2. **Workload-conditioned allocation.** Prefill (compute-bound) and decode (memory-bound) have different optimal allocations; no static allocation is minimax-optimal. The theorem is a “no static optimum” lower bound plus a two-profile portfolio upper bound. This is a genuine functional change, not a new application.

Both repairs are more novel than the static knapsack and are the fallback if the pilot kills the static framing.

---

## 4. Theorem/bound shape, proof sketch, counterexample

### 4.1 Proposition 1 — the combinatorial problem is not the hard part

The additive problem is a **multiple-choice knapsack (MCKP)**:
$$
\max_{x\in C}\ \widehat A(x)
\quad\text{s.t.}\quad
\sum_{\ell=1}^{L}\widehat\mu_\ell(c_\ell)\le B'.
$$
With $L\le32$, $|C_\ell|\le10$, and an energy grid of $10^3$ bins, exact DP runs in
$$
O\big(L\cdot|C_\ell|\cdot B_{\rm bins}\big)\approx 2.8\times10^5
$$
transitions — milliseconds. Therefore:

1. The **integrality gap** of the LP relaxation is not the binding constraint; the exact integer optimum is computable. A theorem about the gap would be technically true but decision-irrelevant.
2. The LP dual is still useful: its optimal dual variable is the shadow price $\lambda^*$, which drives the noise bound.
3. A greedy $(1-1/e)$ theorem is unnecessary and may be wrong. The brief’s suggested theorem shape is misdirected for this instance size.

This is the first adversarial correction: **the interesting theory is statistical, not combinatorial.**

### 4.2 Theorem 1 — price of measurement noise (rate + matching lower bound)

**Assumptions.**

- **(A1) Additive energy:** $\mu(x,w)=\mu_0(w)+\sum_\ell\mu_\ell(c_\ell,w)$, with interaction residual $|\rho(x,w)|\le\rho_{\max}$.
- **(A2) Sub-Gaussian noise:** for each $(\ell,c)$, $E_{\ell,c}-\mu_\ell(c)$ is $\sigma_{\ell,c}$-sub-Gaussian, independent across repetitions and pairs. (Autocorrelation is handled by batch means; the effective $n$ replaces $n$.)
- **(A3) Validation accuracy:** $|\widehat A(x)-A(x)|\le\varepsilon_A$ for all evaluated $x$, w.p. $\ge1-\delta_A$.
- **(A4) Value-function Lipschitz:** $V(B)=\max_{\mu(x)\le B}A(x)$ satisfies $V(B)-V(B-\Delta)\le\bar\lambda\Delta$ for all $\Delta\ge0$. This holds if $V$ is concave; in general $\bar\lambda$ is the maximum dual price over the relevant range. If $V$ has a cliff, $\bar\lambda=\infty$; see §4.5.

**Statement.** Let
$$
\varepsilon_{\max}=\max_{x\in C}\sum_{\ell=1}^{L}\varepsilon_{\ell,c_\ell}(\delta),
\qquad
\widehat x=\arg\max_{x\in C_\delta(B)}\widehat A(x).
$$
Then, with probability at least $1-\delta-\delta_A$,

1. **(feasibility)** $\mu(\widehat x)\le B+\rho_{\max}$;
2. **(near-optimality)**
$$
A^*(B)-A(\widehat x)\ \le\ 2\bar\lambda\,\varepsilon_{\max}+2\varepsilon_A+\bar\lambda\,\rho_{\max}.
$$

**Proof sketch.**

*Feasibility.* By the sub-Gaussian tail and the union bound over the $\sum_\ell|C_\ell|$ pairs, $\mu(\widehat x)\le\widehat\mu(\widehat x)+\varepsilon(\widehat x)+\rho_{\max}\le B+\rho_{\max}$ with probability $\ge1-\delta$.

*Near-optimality.* Let $\Delta=2\varepsilon_{\max}+\rho_{\max}$ and let $x'$ be the oracle at budget $B-\Delta$. Then
$$
\widehat\mu(x')\le\mu(x')+\varepsilon_{\max}\le B-\Delta+\varepsilon_{\max}\le B-\varepsilon_{\max},
$$
so $\widehat\mu(x')+\varepsilon(x')\le B$; hence $x'$ is certified-feasible. Therefore
$$
\widehat A(\widehat x)\ge\widehat A(x')\ge A(x')-\varepsilon_A
=V(B-\Delta)-\varepsilon_A
\ge V(B)-\bar\lambda\Delta-\varepsilon_A.
$$
Using $A(\widehat x)\ge\widehat A(\widehat x)-\varepsilon_A$ gives the claimed bound. $\square$

**Lower bound (matching rate).** Consider a two-layer instance with a continuous relaxation of layer 2 (the lower bound applies a fortiori to the discrete action space). Layer 1 has two configurations with means $\mu$ and $\mu+\Delta$ and identical accuracy. Layer 2 has a continuum of configurations with accuracy linear in energy at rate $\lambda^*>0$. The budget is $B=\mu+e_0$. If the algorithm chooses the low-energy configuration when the true mean is high, it violates the budget by $\Delta$ (or must cut layer 2 by $\Delta$, losing $\lambda^*\Delta$). If it chooses the high-energy configuration when the true mean is low, it wastes $\Delta$ and loses $\lambda^*\Delta$. The observations are $N(\mu,\sigma^2/n)$ vs. $N(\mu+\Delta,\sigma^2/n)$. By Le Cam’s two-point lemma, the total variation is at most $\Delta\sqrt n/(2\sigma)$; choosing $\Delta=\sigma/\sqrt n$ makes the instances indistinguishable with constant error. Hence any algorithm suffers expected loss
$$
\Omega\!\left(\lambda^*\frac{\sigma}{\sqrt n}\right)
$$
or violates the budget with probability $\ge1/4$. This matches the upper bound up to the $\sqrt{\log(2L|C_\ell|/\delta)}$ factor. $\square$

**Corollary 1 (measurement allocation).** Minimize $\sum_\ell\lambda_\ell\varepsilon_\ell$ subject to $\sum_\ell n_\ell=N$. With $\varepsilon_\ell\propto\sigma_\ell/\sqrt{n_\ell}$, the optimum is
$$
n_\ell\ \propto\ (\lambda_\ell\sigma_\ell)^{2/3}.
$$
Measurements should be concentrated on layers with high shadow price and high variance, not spread uniformly. If a run of configuration $\ell$ costs $c_\ell$ energy, the constrained optimum becomes $n_\ell\propto(\lambda_\ell\sigma_\ell/c_\ell)^{2/3}$.

**Corollary 2 (joint validation/measurement design).** If the certificate width is dominated by both terms, minimize
$$
\bar\lambda\Big(\sum_\ell\varepsilon_\ell(n_\ell)+\varepsilon_A(n_{\rm val})\Big)
\quad\text{s.t.}\quad
c_E\sum_\ell n_\ell+c_A n_{\rm val}\le C_{\rm total}.
$$
The same water-filling gives $n_{\rm val}\propto(\bar\lambda/c_A)^{2/3}$ and $n_\ell\propto(\bar\lambda\sigma_\ell/c_E)^{2/3}$. This makes the experiment-design problem part of the estimand, not an afterthought. It also yields a practical diagnostic: if the validation term dominates, the bottleneck is accuracy evaluation, not energy measurement; spend the next GPU hour on more questions, not more repetitions.

**What the theorem does not say.** It does not certify the true accuracy $A(\widehat x)$ beyond $\varepsilon_A$; it does not solve the exploration problem of which configurations to add to $C$; it does not model time-varying carbon intensity; and it can be vacuous if $\varepsilon_A$ or $\rho_{\max}$ is large. The paper must report the certificate width and the dominant term honestly.

### 4.3 Theorem 2 — non-separability of layer-wise accuracy

**Claim.** For any layer-separable surrogate $A_{\rm sep}(x)=A_0+\sum_\ell a_\ell(c_\ell)$, there is a two-layer, two-configuration instance where the true accuracy and the surrogate rank the allocations differently, and an allocation that trusts the surrogate can lose $\Omega(1)$ accuracy.

**Construction.** Let each layer have configurations $H$ (high precision) and $L$ (low precision), with the true accuracy table

| | $H$ | $L$ |
|---|---|---|
| $H$ | 1.00 | 0.90 |
| $L$ | 0.90 | 0.10 |

The interaction is
$$
A(H,H)+A(L,L)-A(H,L)-A(L,H)=1.00+0.10-0.90-0.90=-0.70.
$$
Any additive interpolation from the other three cells predicts $A(L,L)=0.80$, but the true value is $0.10$. Any additive model has a sup-norm residual of at least $|\text{interaction}|/4=0.175$ on this table, and an allocation that trusts the interpolation can lose up to $0.70$ accuracy. The interaction cannot be detected from single-layer ablations; it requires a $2\times2$ factorial measurement. Estimating the interaction to error $\varepsilon$ requires $\Omega(1/\varepsilon^2)$ samples per cell (binomial variance).

**Remark (minimax extension).** The example extends to a policy-class lower bound: over the class of $2\times2$ interaction tables with $|\text{interaction}|\ge\gamma$, any layer-separable policy has worst-case regret at least $\gamma/4$, while an interaction-aware DP using the measured $2\times2$ table has regret $O(\sqrt{\log|C|/n})$ from estimation error. This is the sense in which separability is a load-bearing assumption, not a modeling convenience.

**Implication.** The knapsack formulation assumes separability. If $\iota$ is not negligible, the problem is a **quadratic knapsack** (NP-hard in general), and per-layer sensitivity proxies (Hessian trace, KL divergence, weight standard deviation) are insufficient. The paper must measure $\iota$ on the top-$k$ sensitive layer pairs and report the cost of ignoring it. This is a load-bearing assumption, not a technicality.

### 4.4 Counterexample — the brief’s submodular/greedy assumption fails

Three layers, binary actions: keep ($K$) or compress ($C$). Energy: $K$ costs 2 J, $C$ costs 1 J. Baseline all-$K$ is 6 J; the budget is 4 J, so exactly two layers must be compressed. Accuracy loss:

| Compressed set | Energy saved | Loss |
|---|---|---|
| {1} | 1 J | 0.01 |
| {2} | 1 J | 0.02 |
| {3} | 1 J | 0.02 |
| {1,2} | 2 J | **0.50** |
| {1,3} | 2 J | **0.50** |
| {2,3} | 2 J | 0.03 |
| {1,2,3} | 3 J | 0.51 |

A greedy by loss-per-joule picks layer 1 first (0.01, the smallest). After that, adding layer 2 or layer 3 costs 0.49, so it picks one and ends with loss 0.50. The optimal is {2,3} with loss 0.03. The marginal loss of adding a second compression after layer 1 is 0.49, much larger than the 0.02–0.03 standalone losses: the loss exhibits **increasing returns** (supermodularity on the relevant pairs), so the $(1-1/e)$ guarantee does not apply. Exact enumeration (or a DP with the pairwise terms) finds {2,3}. This is a concrete refutation of the brief’s suggested “greedy allocation with a $(1-1/e)$-type guarantee under submodular curves.” The assumption is not innocuous, and the paper should not build its theorem on it.

### 4.5 Counterexample — the Lipschitz assumption fails at a cliff

Suppose one layer has:

| Config | Accuracy | Energy |
|---|---|---|
| 4-bit | 0.90 | 1.0 J |
| 3-bit | 0.89 | 0.7 J |
| 2-bit | 0.50 | 0.6 J |

The value function drops by 0.39 when the budget crosses 0.7 J. A 0.05 J measurement error can flip the allocation from 3-bit to 2-bit, losing 0.39 accuracy. The local Lipschitz constant is effectively infinite; the price-of-noise bound with a finite $\bar\lambda$ is vacuous. This is realistic for small models at 2–3 bits. The repair is a **cliff-aware** constraint: require the certified energy of the 3-bit configuration to be at most $B-\delta_{\rm cliff}$, or use a **CVaR** objective that penalizes the worst-case accuracy over the noise distribution. The paper should plot accuracy vs. per-layer bits and test for discontinuities; if cliffs dominate, the linear-programming/knapsack framing is the wrong model, and a risk-averse formulation is required.

### 4.6 Which bound shape survives

| Shape | Verdict |
|---|---|
| Knapsack integrality gap | **Not decision-changing.** Exact DP is milliseconds at $L\le32$; the gap is a footnote, not a theorem. |
| LP relaxation / dual | **Useful as a certificate.** The dual gives $\lambda^*$, the price of a joule, which drives the noise bound. The LP is not the algorithm. |
| Submodular greedy $(1-1/e)$ | **Not defensible without strong assumptions.** Accuracy loss can be supermodular; the counterexample in §4.4 is decisive. Use exact DP or a MILP instead. |
| Price-of-noise rate + matching lower bound | **The defensible core.** It is a non-identity theorem, it changes the experiment design, and it is testable. |
| Non-separability lower bound | **The necessary caveat.** It tells the reader when the knapsack is the wrong model and what to measure instead. |

---

## 5. Dataset and energy protocol

### 5.1 Models

| Model | Params | Why | Fit on 6 GB |
|---|---|---|---|
| Qwen2.5-0.5B | 0.49B | Cheapest; fast iteration; ARC-Easy above chance | FP16 ~1 GB; 4-bit ~0.3 GB |
| Qwen2.5-1.5B | 1.54B | Main accuracy model; MMLU/ARC signal | FP16 ~3.1 GB; 4-bit ~0.9 GB |
| Pythia-410M | 0.41B | Independent architecture/tokenizer; replication | FP16 ~0.8 GB |
| Pythia-1.4B | 1.4B | Independent family at the 1.5B scale | FP16 ~2.8 GB; 4-bit ~0.8 GB |

All are public, non-gated, and small enough for the 3060. The 24 GB rental is optional for the 1.5B FP16 arm at batch 8 (which may be tight on 6 GB) and for more repetitions; at ~$0.3–0.5/h for ~20 h, it is within the $50 budget.

### 5.2 Data

- **MMLU** (`cais/mmlu`, test split, 14,042 questions, 57 subjects). Pilot: 200 stratified questions. Full: 2,000 stratified (≈35 per subject), with a held-out 1,000 for validation. Use 5-shot for the main table, 0-shot for the pilot; report both if time permits.
- **ARC-Easy / ARC-Challenge** (`allenai/ai2_arc`, 2,376 / 1,172 questions). Pilot: 200 / 100. Full: 1,000 / 500. 0-shot.
- **Energy prompt set:** 128 prompts sampled from the benchmark *validation* split (not the test questions), with fixed output length 64 tokens. Prompt lengths 128/512/2048; batch sizes 1 and 8. The full factorial is 18 workload cells; the pilot uses 3 (128/64, 512/64, 2048/16).
- **New dataset:** “CarbonQuant-4” — per-configuration energy, latency, accuracy for 4 models × 20–200 configurations × 3 workloads × 5 repetitions, released as CSV/Parquet with the paper. This is a reusable artifact and a dataset contribution, independent of whether the allocation theorem survives.

### 5.3 Energy measurement

- **Primary:** `nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits -lms 100`, integrated with the trapezoidal rule. Timestamp each sample with `time.time_ns()`.
- **Cross-check:** `codecarbon` if installed; a wall meter if available. Report both if they disagree by >10%.
- **Idle baseline:** 60 s of no workload; subtract. **Warm-up:** 3 runs discarded; **measured:** 5 runs per configuration. Report mean, SD, and CV.
- **Power cap / clocks:** set a fixed power cap (`nvidia-smi -pl`) if the laptop allows it; otherwise record the default and the clock variation. Randomize/interleave configuration order to balance thermal drift.
- **Units:** joules per query, joules per token, and joules per correct answer. Convert to gCO2e with the carbon intensity.
- **Honesty about the 3060:** laptop 3060 power readings may be coarse (0.1–1 W granularity, ~1 Hz). If the CV across 5 repetitions is >20%, the certification framing is not empirically testable on this GPU; use a wall meter or rent a 24 GB GPU. State this in the paper rather than hiding it.

### 5.4 Carbon intensity

- **Main experiment:** a fixed regional annual average from **Ember’s public CSV** (or EPA eGRID for US subregions). Record the exact year, region, and value at retrieval; do not hard-code from memory. Sanity check: the UK is roughly 0.2–0.3 kgCO2/kWh and India roughly 0.6–0.7 kgCO2/kWh, but use the current Ember release, not these approximations.
- **Conversion:** $\text{gCO}_2\text{e}=E\,[\text{J}]/3.6\times10^6\times \text{CI}\,[\text{g/kWh}]$. Example: 1 J/query × 1M queries = 1 MJ = 0.278 kWh; at 0.4 kg/kWh = 111 gCO2 per 1M queries.
- **Budget:** $B\in\{0.6,0.8,1.0\}\times E_{\rm FP16}$ per 1,000 queries, reported in both joules and gCO2e.
- **Dynamic extension (optional, for the carbon-specific claim):** UK National Grid ESO Carbon Intensity API (free, no key) or ENTSO-E (free registration), 30-min resolution, 30-day replay. Compute $\sum_t c_tE_t\le B_{\rm CO_2}$; compare a static profile to a threshold policy. No paid API.

### 5.5 Statistics and power

- **Primary outcome:** accuracy at matched measured energy $B$. Paired bootstrap over questions (1,000 resamples) for 95% CIs.
- **Secondary:** joules per correct answer; gCO2e per correct answer; frontier slope $\lambda^*$; certificate width and its dominant term.
- **Power:** for a paired accuracy difference with discordance rate $q\approx0.1$–$0.2$, the MDE at 80% power/α=0.05 is roughly $2.8\sqrt{q/N}$. At $N=200$, MDE ≈ 6–9 points; at $N=2{,}000$, ≈ 2–3 points; at $N=5{,}000$, ≈ 1.3–2 points. The pilot (200 questions) only detects large effects; the full study needs ≥2,000 questions per benchmark for a 3-point claim.
- **Energy:** with 5 repetitions and CV=10%, the 95% CI half-width is ≈9%; with 10 repetitions, ≈6%. The pilot can detect a 20% energy difference, not a 10% one; use 5 repetitions and report the CI.

### 5.6 Confounders and controls

- GPU clock/thermal drift → interleave configurations; record temperature and clocks.
- Background processes → kill them; record `nvidia-smi` processes.
- Model/page-cache warm-up → discard the first 3 runs.
- Kernel backend differences (bnb vs. torchao vs. GGUF) → fix one backend for the main comparison; report it.
- Prompt/output length and batch size → stratify; do not compare a 4-bit long-prompt run to an 8-bit short-prompt run.
- Accuracy evaluation order/seed → fixed seed and identical prompts across configurations.
- Carbon intensity source → fixed version and retrieval date; no mixing of annual averages and real-time signals in the same table.

---

## 6. Pilot — ≤2 hours on the 3060

**Important:** this note does **not** execute the pilot. The brief forbids GPU/model loads/downloads for this research task. The pilot is a plan for the team’s GPU session; models must be pre-cached.

### 6.1 Timeline

| Time | Step | Output |
|---|---|---|
| 0:00–0:15 | Environment check: `nvidia-smi`, `bitsandbytes`, `transformers`, `lm-eval`, model cache; record idle power, power granularity, power cap | `env.txt` |
| 0:15–0:50 | Energy microbenchmark: Qwen2.5-0.5B + Pythia-410M; FP16 / INT8 / NF4; 128 prompts × 64 tokens; 5 reps; interleaved order | `energy.csv`, CV, J/query |
| 0:50–1:20 | Accuracy: ARC-Easy 200 + MMLU 200; same 3 configurations; paired correctness | `acc.csv`, paired CIs, MDE |
| 1:20–1:45 | Sparsity: Qwen2.5-0.5B, 2:4 50% (Wanda), energy + accuracy; fit additive energy model with 80/20 held-out configurations | `energy_fit.csv`, $R^2$, interaction residual |
| 1:45–2:00 | Decision: exact DP under $B=0.8\times E_{\rm FP16}$; compare energy-aware vs. uniform NF4; check K1–K7 | one-page memo |

### 6.2 Commands sketch (not executed here)

```bash
# power log
nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu \
  --format=csv,noheader,nounits -lms 100 > power.csv &

# accuracy
python -m lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B \
  --tasks arc_easy,mmlu --limit 200 --batch_size 8 --output_path acc/

# energy (custom harness; fixed prompts, fixed output length, 5 reps)
python run_energy.py --model Qwen/Qwen2.5-0.5B --quant fp16,int8,nf4 \
  --prompts 128 --max_new_tokens 64 --reps 5 --out energy.csv

# additive fit + exact DP
python fit_energy.py --csv energy.csv --holdout 0.2
python dp_allocate.py --energy energy.csv --accuracy acc.csv --budget_frac 0.8
```

### 6.3 Kill criteria

- **K1 — no energy headroom.** The 95% CI of the FP16−NF4 energy difference includes zero, or the point estimate is <10% on both models. → KILL: the budget is not binding.
- **K2 — accuracy insensitivity.** $|\text{acc(NF4)}-\text{acc(FP16)}|<1$ point on both MMLU and ARC, and $|\text{acc(INT8)}-\text{acc(FP16)}|<1$ point. → KILL: there is nothing to trade.
- **K3 — measurement noise too high.** CV of J/query across 5 reps >20%. → Use a wall meter or rent a 24 GB GPU; if neither, KILL the certification framing.
- **K4 — non-additive energy.** Additive energy model $R^2<0.8$ on held-out allocations. → Pivot to end-to-end black-box allocation or a measurement paper; the per-layer knapsack is invalid.
- **K5 — energy ≈ latency.** Spearman $\rho(\text{energy},\text{latency})>0.95$ across configurations. → The energy constraint is redundant with latency; pivot to carbon-intensity-conditioned profiles (where carbon ≠ energy × constant) or KILL the “carbon-aware” claim.
- **K6 — no decision-changing gain.** The predicted accuracy gain of the energy-aware allocation over uniform NF4 at $B=0.8\times E_{\rm FP16}$ is <2 points, or below the pilot’s MDE. → Pivot to a measurement/benchmark paper or a larger model; do not claim a decision-changing method.
- **K7 — literature match.** A 2024–2026 paper with the same certified-energy-allocation theorem appears. → KILL or pivot to §3.3. None found as of 2026-11-08.

### 6.4 What the pilot can and cannot decide

- **Can decide:** whether there is energy headroom, whether accuracy is sensitive at this scale, whether the power measurement is reliable, whether additivity is a reasonable approximation, and whether energy is redundant with latency.
- **Cannot decide:** a 2-point accuracy effect (MDE ≈ 6–9 points at $N=200$), the full frontier, the carbon-intensity-conditioned extension, or the interaction matrix for all layer pairs. Those require the full study.
- **Decision rule:** if K1–K5 pass and the predicted K6 gain exceeds the **full-study** MDE (≈2–3 points at $N\ge2{,}000$), proceed to the full study. If K1–K4 pass but K6 fails, the project can still be a **certified measurement/benchmark** paper (the certificate and the dataset are contributions), but not the “decision-changing allocation” paper. If K5 fails, the static energy framing is dead; the carbon-conditioned profile repair is the only path to 8+.

---

## 7. Scores

### 7.1 As written (original knapsack/greedy G3)

| Factor | Weight | Score | Weighted |
|---|---:|---:|---:|
| Novelty | 0.25 | 6.5 | 1.625 |
| Method | 0.20 | 7.0 | 1.400 |
| Math | 0.15 | 6.5 | 0.975 |
| Experiments | 0.10 | 7.0 | 0.700 |
| Dataset | 0.10 | 7.5 | 0.750 |
| Q1 fit | 0.10 | 7.5 | 0.750 |
| Feasibility | 0.05 | 8.5 | 0.425 |
| Compute | 0.05 | 8.5 | 0.425 |
| **Weighted** | | | **7.05** |
| **Core** (Nov, Method, Math, Exp) | | | **6.75** |
| **Independence** | | | **7.0/10** |

**Why so low.** The object is crowded (SigmaQuant/LieQ/PolyQ/HAWQ-V3), the budget unit (joules vs. bits) is a thin delta, the combinatorial problem is trivial at this $L$, and the greedy/submodular theorem is not defensible. This is a solid 6.5–7 paper, not an 8+.

### 7.2 Repaired (certified carbon knapsack: noise rate + non-separability + exact DP dual)

| Factor | Weight | Score | Weighted |
|---|---:|---:|---:|
| Novelty | 0.25 | 8.0 | 2.000 |
| Method | 0.20 | 8.0 | 1.600 |
| Math | 0.15 | 8.0 | 1.200 |
| Experiments | 0.10 | 7.5 | 0.750 |
| Dataset | 0.10 | 8.0 | 0.800 |
| Q1 fit | 0.10 | 8.0 | 0.800 |
| Feasibility | 0.05 | 8.0 | 0.400 |
| Compute | 0.05 | 8.5 | 0.425 |
| **Weighted** | | | **7.975** |
| **Core** (Nov, Method, Math, Exp) | | | **7.875** |
| **Independence** | | | **7.5/10** |

**Rationale.**

- **Novelty 8.0:** the certified frontier $A^*_\delta(B)$ and the price-of-noise rate are new for this problem; layer-wise bit allocation itself is not.
- **Method 8.0:** robust DP, measurement allocation, interaction calibration, and a preregistered protocol. The method is rigorous but not architecturally novel.
- **Math 8.0:** matching upper/lower rate plus a non-separability counterexample and a cliff counterexample. The tools (Hoeffding, Le Cam, LP duality) are standard; the combination and the application are the contribution.
- **Experiments 7.5:** four small models, two benchmarks, measured energy, interaction study; the empirical gain may be modest and the accuracy MDE is a real constraint. Upside to 8.5 if the pilot shows a ≥3-point gain.
- **Dataset 8.0:** public benchmarks plus a new per-configuration energy/accuracy dataset (“CarbonQuant-4”), reusable beyond the paper.
- **Q1 fit 8.0:** `Sustainable Computing: Informatics and Systems`, `IEEE Transactions on Sustainable Computing`, `ACM TACO`, or `Journal of Systems and Software`; the paper has both a systems measurement and a theory component.
- **Feasibility 8.0:** 3–4 people, 4-5 months; the pilot is 2 h, the full study is 4–6 months; the 3060 is enough for 0.5B/1.5B, the 24 GB rental is optional.
- **Compute 8.5:** small models, inference-only, no pretraining; energy measurement is cheap; the main cost is accuracy evaluation, which is bounded.
- **Independence 7.5/10:** distinct object (layer-wise compression) from G1 (model cascade), G2 (retrieval depth), G4 (data selection), and G5 (reasoning strategies). It shares the “budgeted allocation” mathematics with G1/G4 and the energy measurement with G1/G2/G5, so it is not a 10. The static certification repair is orthogonal to those lanes, hence 7.5; the optional carbon-intensity-conditioned extension would lower this to ~7.0 because it overlaps G1’s time-varying carbon theme. Relative to the prior reasoning/calibration/RAG/watermark portfolio, it is independent; no overlap with those lanes.

### 7.3 Score sensitivity

| Scenario | Core | Weighted | Verdict |
|---|---:|---:|---|
| Original as written | 6.75 | 7.05 | Below bar; revise |
| Repaired, pilot passes K1–K5 but K6 borderline | 7.875 | 7.975 | Strong Q1; not 8.5 |
| Repaired, pilot shows ≥3-point gain at matched joules | 8.125 | 8.075 | GO; target 8.5 |
| Repaired + carbon-intensity-conditioned profiles with a positive gain | 8.25–8.5 | 8.3–8.5 | Best path to 8.5 |
| Pilot K1/K2/K5 fails | — | — | KILL or pivot to measurement paper |

---

## 8. Verdict and path

### 8.1 Verdict

**CONDITIONAL — GO only on the repaired framing; KILL the original knapsack/greedy version.**

- **Not a literal KILL:** no 2024–2026 paper has the same estimand + finite-sample guarantee. SigmaQuant is the closest, but it optimizes memory/accuracy and evaluates energy post hoc; LieQ/PolyQ constrain average bits; Towards Green AI measures but does not allocate; CTT uses a carbon-tax penalty, not a hard budget.
- **The original is a 6.5–7:** the combinatorial problem is exactly solvable at this scale, so an integrality-gap or greedy theorem is not decision-changing; “joules instead of bits” is a thin delta.
- **The repair is a credible Q1 paper:** certify the joule budget under noisy `nvidia-smi` measurements, bound the price of noise by the shadow price of a joule, and confront non-separability with a measured interaction residual. Core 7.875 / Weighted 7.975, with a path to 8.5.

### 8.2 What would change the verdict

**To GO (proceed to full study):**

1. K1: ≥10% energy headroom between FP16 and NF4 at matched workload, with a CI excluding zero.
2. K2: ≥2-point accuracy spread between at least two configurations on MMLU or ARC (otherwise the allocation has nothing to trade).
3. K3: CV ≤20% across 5 repetitions (or a wall meter); target ≤10% for the full study.
4. K4: additive energy $R^2\ge0.9$ on held-out allocations.
5. K5: Spearman(energy, latency) ≤0.85 (energy is not just latency).
6. K6: predicted gain ≥3 points at $B=0.8\times E_{\rm FP16}$, above the full-study MDE.
7. K7: literature recheck finds no same-estimand paper.

**To KILL:**

- Any of K1–K5 fails and cannot be fixed with a wall meter/rental.
- K6 fails with a well-powered full study (not just the pilot).
- A 2024–2026 paper with the same certified-energy-allocation theorem appears.
- The team cannot measure power reliably on the 3060, and the 24 GB rental is unavailable.

**To 8.5+:**

- The pilot/full study shows a ≥4-point accuracy gain at matched measured joules, and the interaction/cliff effects are measurable. The paper becomes “certified carbon knapsack with interaction-aware allocation,” with a real decision-changing result.
- Or the carbon-intensity-conditioned extension shows a positive gain over the best static profile on a 30-day replay; that is the most carbon-specific and most novel version.

### 8.3 Path

1. **Week 1:** run the ≤2 h pilot (K1–K7); write a one-page memo with the energy table, accuracy table, CV, $R^2$, and Spearman.
2. **Weeks 2–4:** if the pilot passes, build the full energy/accuracy dataset (4 models, 20–200 configurations, 3 workloads, 5 reps); fit the additive and pairwise-interaction models; compute the certificate widths.
3. **Weeks 5–8:** implement the exact DP + robust margin; compare uniform, bit-budget, energy-aware, and oracle allocations at matched measured energy; run the paired bootstrap and the power analysis.
4. **Weeks 9–12:** add the carbon-intensity-conditioned extension (UK/ENTSO-E replay) and the cliff-aware CVaR variant; test whether they beat the static allocation.
5. **Months 4–6:** write the paper; target `Sustainable Computing` or `IEEE Trans. Sustainable Computing`; release “CarbonQuant-4.”
6. **Decision gate:** if the pilot kills the static framing, do not spend months on it; pivot to the carbon-conditioned profiles (§3.3) or to a measurement/benchmark paper using the same dataset.

---

## 9. References and verification status

All abstract pages below were fetched on **2026-11-08** unless marked “listing verified only” (metadata came from an arXiv search listing, not the abstract page). “Abstract verified” means the arXiv abstract page was read; “venue from arXiv comments” means the venue claim is from the paper’s own comments/journal-ref field, not independently checked.

1. **SigmaQuant.** Q. Liu, P. Yu, M. Zapater, D. Atienza. “SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference.” arXiv:2602.22136 (v2, 3 Mar 2026). https://arxiv.org/abs/2602.22136 — **abstract verified**; preprint; venue unverified.
2. **LieQ.** H. Xiao, Q. Yang, D. Xie, W. Xu, Z. Su, R. Yang, W. Zhou, H. Liu, Z. Liu, N. Wong. “Exploring Layer-wise Information Effectiveness for Post-Training Quantization in Small Language Models.” arXiv:2508.03332 (v2, 27 Dec 2025). https://arxiv.org/abs/2508.03332 — **abstract verified**; preprint; venue unverified.
3. **PolyQ.** H. Oh, S. Jang, H. Chen, K. Nam, S. Yun, R. Masukawa, M. Imani. “PolyQ: Codesigning End-to-End Quantization Framework for Scalable Edge CPU LLM Inference.” arXiv:2607.14618; ICCAD 2026 (per arXiv comments). https://arxiv.org/abs/2607.14618 — **abstract verified**; venue from arXiv comments.
4. **Towards Green AI.** P. Basta. “Towards Green AI: Evaluating Energy Efficiency for Quantized LLM Inference.” arXiv:2610.17531 (13 Oct 2026). https://arxiv.org/abs/2610.17531 — **abstract verified**; preprint; venue unverified.
5. **Carbon-Taxed Transformers.** A. I. Alam, P. Roy, C. K. Roy, B. Roy, K. A. Schneider. “Carbon-Taxed Transformers: A Green Compression Pipeline for Overgrown Language Models.” arXiv:2604.25903; FSE 2026, Proc. ACM Softw. Eng. 3, Article FSE047; DOI 10.1145/3797075. https://arxiv.org/abs/2604.25903 — **abstract verified**; journal ref on arXiv page.
6. **MOSP.** M. J. Ali, S. Niar, E.-G. Talbi. “Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization.” arXiv:2607.22583; ICTAI 2026 (per arXiv comments). https://arxiv.org/abs/2607.22583 — **abstract verified**.
7. **KVTuner.** X. Li et al. “KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference.” arXiv:2502.04420; ICML 2025 (per arXiv comments). https://arxiv.org/abs/2502.04420 — **abstract verified**.
8. **HAWQ-V3.** Z. Yao et al. “HAWQV3: Dyadic Neural Network Quantization.” arXiv:2011.10680; ICML 2021. https://arxiv.org/abs/2011.10680 — **abstract verified**; precursor, outside 2024–2026.
9. **DSA.** X. Ning, T. Zhao, W. Li, P. Lei, Y. Wang, H. Yang. “DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation.” arXiv:2004.02164 (2020). https://arxiv.org/abs/2004.02164 — **abstract verified**; precursor; venue unverified.
10. **LAMP.** J. Lee, S. Park, S. Mo, S. Ahn, J. Shin. “Layer-adaptive sparsity for the Magnitude-based Pruning.” arXiv:2010.07611; ICLR 2021. https://arxiv.org/abs/2010.07611 — **abstract verified**; precursor.
11. **Carbon break-even.** S. Das, T. Joshi, K. Ghosh. “Can We Optimize the Performance-Carbon Emission Break-Even Point?: The Quest for Greener LLMs.” arXiv:2608.08744 (9 Aug 2026). https://arxiv.org/abs/2608.08744 — **abstract verified in part**; preprint.
12. **Dense Structural Compression.** J. A. Duersch, N. Es-Sebbani, N. Haas, Z. Bouraoui. “Dense Structural Compression of Transformers via Gauge-Correct Channel Removal.” arXiv:2609.07264; NeurIPS 2026 (per arXiv comments). https://arxiv.org/abs/2609.07264 — **abstract verified**.
13. **FAMPWQ.** G. Lee, J. Liu, J. Cheng, J. Wu. “FAMPWQ: Fisher Information-based Adaptive Mixed Precision Weight Quantization for Effective LLM Inference.” arXiv:2608.24945; EMNLP 2026 (per arXiv comments). — **listing verified only**; not read in full.
14. **RAMP.** A. S. Gautam, S. Jha. “RAMP: Reinforcement Adaptive Mixed Precision Quantization for Efficient On Device LLM Inference.” arXiv:2603.17891 (2026). — **listing verified only**.
15. **BitsMoE.** J. Zhao et al. “BitsMoE: Cost-Aware Bit Allocation in Spectral Space for MoE LLM Quantization.” arXiv:2606.00079 (2026). — **listing verified only**.
16. **MODE.** Y. Chen et al. “MODE: Modality-Decomposed Expert-Level Mixed-Precision Quantization for MoE Multimodal LLMs.” arXiv:2606.17118; EMNLP 2026 (per arXiv comments). — **listing verified only**.

**Search log.** Queries included: “energy-aware mixed-precision quantization LLM,” “layer-wise sparsity allocation budget LLM,” “carbon-aware quantization neural network inference,” “mixed-precision quantization knapsack bit-width allocation,” “energy budget bit-width neural network,” “carbon quantization large language model inference,” “layer-wise sparsity energy inference transformer,” “energy-aware pruning LLM budget,” “budgeted pruning approximation guarantee,” “submodular model compression resource allocation,” “mixed-precision quantization integer linear programming,” and exact-title lookups. Sources: arXiv API/search, arXiv abstract pages, keyless web search. Tavily was quota-limited; several keyless backends were degraded. The absence of a same-estimand paper is therefore **best-effort as of 2026-11-08**, not a proof of absence. Recheck before the full study.

---

*File: `green_computing/03_g3_quantization.md`. Ledger: `green_computing/ledger_g3.csv`.*
