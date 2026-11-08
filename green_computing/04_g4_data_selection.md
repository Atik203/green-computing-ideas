# G4 — Energy-normalized data selection for instruction tuning
### Adversarial audit, KILL of the stated formulation, and a defensible repair

**Date:** 8 November 2026 · **Track:** Green computing · **Budget:** CPU/web only; no GPU, no model loads, no downloads, no paid API.
**Portfolio update (8 Nov):** G3′ was killed by its pilot (NF4 energy loss, additive \(R^2\) fail, no feasible mix at \(0.8\times\)FP16). G4′ is now the **provisional green lead** and must stand alone; do not fold it into the killed G3′.
**Method of verification:** arXiv API metadata + abstracts + HTML full text (fetched 2026-11-08); Springer chapter page and abstract; no PDF-only claims are marked verified.
**Scope note:** this file writes only under `green_computing/`; `research_audit/round9` and `round10` were not opened.

---

## 0. Verdict box

| Object | Score (weighted / core) | Status |
|---|---|---|
| **G4 as written in the brief** — "select subset $S$ maximising quality gain per joule, submodularity + greedy $(1-1/e)$" | **6.25 / 5.88** | **KILL** |
| **G4′ — "certified joule-budgeted selection"** (repair developed below) | **7.78 / 7.50** | **CONDITIONAL GO**, gated by a 2-hour pilot |

**One-line reason for the kill.** The brief's formulation is not new at the functional level. `arXiv:2610.14028` (Sep 2026) already formalises $\max_S u(S)$ s.t. $\sum_{j\in S} p_j \le B$ with **per-sample prices** — a joules cost is the special case $p_j=e_j$ — and proves a subset-utility approximation guarantee; `arXiv:2410.16208` (ICLR 2025) already formalises a **cost-aware utility that budgets selection cost and training cost jointly** and already shows cheap selectors dominate; `arXiv:2601.23155` (ICLR 2026) already applies monotone submodular selection with a $(1-1/e)$ cardinality guarantee **to instruction tuning**. The $(1-1/e)$ step is Nemhauser–Wolsey–Fisher (1978) plus Sviridenko's knapsack extension (2004) — using it here is plumbing, not a theorem.

**One-line reason the repair survives.** None of the five closest papers buys **evidence**. They optimise a subset against a *known* validation signal. The moment the gain must be *certified* rather than asserted, the validation set becomes a budgeted resource, the search width $M$ enters the certificate, and a new decision appears: **certify or abstain**. That object — certified quality gain per joule — is unoccupied, and on 410M-class models its energy cost is the same order as the fine-tuning it is meant to justify.

---

## 1. Plain-language problem

Instruction tuning is the standard cheap adaptation step for an LLM, so "which 500 examples should I train on?" has become a crowded research area. Almost every method reports a quality gain and a *data* budget (examples, tokens, or FLOPs), and almost none reports the joules actually drawn — or the joules spent deciding *which* examples to use.

The green question is not "which subset is best" but **"given a fixed electricity budget, what can I honestly claim I bought?"** That includes the energy of scoring the pool, the energy of training, and the energy of the evaluation that proves the gain was real. This last term is the one nobody budgets, and it is large enough to reverse the decision.

---

## 2. Exact estimand / functional

### 2.1 The formulation in the brief (and why it is the wrong object)

$$
\max_{S\subseteq \mathcal{P}}\ \Delta Q(S)\quad \text{s.t.}\quad E(S)\le B,
\qquad\text{or}\qquad \max_{S}\ \frac{\Delta Q(S)}{E(S)} .
$$

**What is wrong with it, precisely:**

1. **It is the already-published formulation.** `2610.14028` Eq. (1) is exactly $\max_S u(S)$ s.t. $\sum_{j\in S}p_j\le B$ with $p_j>0$ per sample, plus a spectral guarantee (Theorem 1) and an APX-hardness result (Theorem 16). Setting $p_j=e_j$ turns their problem into ours with no new mathematics.
2. **The ratio form is not submodular.** $f(S)/E(S)$ with $f$ monotone submodular and $E$ modular is not submodular, so the $(1-1/e)$ guarantee does **not** transfer to "quality per joule"; the guarantee people would cite is for the cardinality-constrained problem, which is a different problem (see §4.5).
3. **$E(S)$ is treated as a known function.** In reality joules are measured with noise, depend on batching and padding, and include a fixed term.
4. **The evaluation energy that establishes $\Delta Q$ is omitted entirely.** This is the load-bearing omission (§4).

### 2.2 The repair: certified joule-budgeted selection

**Notation.**

| symbol | meaning |
|---|---|
| $\mathcal{P}$ | candidate pool, $|\mathcal{P}|=n$ |
| $\pi$ | selection procedure; returns $S_\pi\subseteq\mathcal{P}$ |
| $M$ | **search width**: number of candidate subsets whose validation performance the procedure inspects |
| $E_\pi(M)$ | energy of running the selector (scoring the pool + any retraining inside the loop) |
| $n_v$ | number of **paired** validation items used to estimate the gain |
| $c_v$ | joules per validation item (one forward pass, fixed length) |
| $a$ | fixed overhead (process start, model load, tokenisation, checkpointing) |
| $e_i$ | marginal training joules of example $i$ |
| $\Delta(S)$ | true quality gain of $S$ over the base model on the target distribution |

**Total energy (the object that must be $\le B$):**

$$
E_{\text{tot}}(S,\pi,n_v)\;=\;\underbrace{E_\pi(M)}_{\text{selection}}
\;+\;\underbrace{a+\sum_{i\in S}e_i}_{\text{training}}
\;+\;\underbrace{c_v\,n_v}_{\text{evidence}} .
$$

**Estimand.** With paired per-item differences $D_i=\ell_i(\text{base})-\ell_i(S)$ on $V$, $|V|=n_v$, and $\hat\Delta(S;V)=n_v^{-1}\sum_i D_i$:

$$
G_{\text{cert}}(B,\delta)\;=\;\max_{\substack{S,\pi,n_v}}\ \Big[\hat\Delta(S;V)-\operatorname{pen}(M,n_v,\delta)\Big]
\quad\text{s.t.}\quad E_{\text{tot}}\le B,
$$

with the guarantee $\Pr\{\Delta(\hat S)\ge G_{\text{cert}}\}\ge 1-\delta$, and $\hat S$ the argmax. If the feasible maximum is $\le 0$, the procedure **abstains**: it declines to claim a gain.

**Decision changed (three-way, not two-way):**
1. select with $\pi$ and **certify** a gain (report $G_{\text{cert}}$ and its joule cost);
2. train on a cheap/random subset and report an **uncertified** point estimate;
3. **abstain** — do not claim a selection gain at this budget (and, if $B<B_{\min}$, do not fine-tune for the purpose of a claimed gain at all).

Every paper in the closest-five list can only make decision 1 or 2. Decision 3 is what a green budget makes rational, and it has no counterpart in that literature.

---

## 3. Pre-emption audit — 5 closest at the functional level

All identifiers below were resolved through the arXiv API or the publisher page on 2026-11-08. "Verified" means I read the abstract (and, where noted, full-text HTML) and the stated capability is present there.

| # | Paper | ID / venue | Functional (what it optimises) | Guarantee | Verified | Same estimand? | Same guarantee? |
|---|---|---|---|---|---|---|---|
| 1 | **Compute-Constrained Data Selection** (Yin & Rush) | arXiv:2410.16208 · **ICLR 2025** | $\max$ training gain with **selection cost + training cost** both inside one compute budget; sweeps training tokens, model size, selection compute | Empirical compute-optimality; no theorem, no certificate | **yes** (abstract + HTML §3, App. B) | **Yes** — cost-aware utility including selection cost | No (no finite-sample guarantee) |
| 2 | **Pick the Mixture, Not the Samples** (Zhang, Wang, Li) | arXiv:2610.14028 · 2026 | $\max_S u(S)$ s.t. $\sum_{j\in S}p_j\le B$, **per-sample prices**; domain mixture then within-domain ranking | Spectral gap bound (Thm 1); APX-hardness (Thm 16) | **yes** (abstract + HTML Eq. 1, App. A.11) | **Yes** — our joule budget is $p_j=e_j$ | **Yes**, at the optimisation level (approximation guarantee) |
| 3 | **SPICE** (Chang et al.) | arXiv:2601.23155 · **ICLR 2026** | Monotone submodular Fisher-information selection for instruction tuning; $\varepsilon$-decomposition for gradient conflict | **$(1-1/e)$ under cardinality budget** | **yes** (abstract) | Partly — quality, not joules | **Yes** — the $(1-1/e)$ the brief proposes |
| 4 | **QSGS** (Zhao et al.) | arXiv:2611.04059 · 2026-11-02 | Submodular graph-cut selection; argues pointwise scoring fails at tight budgets; 0.5% data budget | Submodular set-function guarantee; no energy axis | **yes** (abstract) | Partly — quality at tight budgets | No (no cost/joule guarantee) |
| 5 | **Balancing Accuracy and Energy: Optimal Subset Size Selection** (Haddaji, Brun, Prabhu) | LNCS 16424, 2026 · doi:10.1007/978-3-032-18494-8_11 | Accuracy **per unit energy** vs subset size; energy-scaling laws; stratified **random** subsets | Empirical; no selection method, no guarantee | **yes** (Springer abstract) | **Yes** — the green estimand (interior optimum) | No |

**Context rows (not closest, but load-bearing for the verdict):**

- `arXiv:2510.16806` — **CADS**, compute-budget-aware bilevel selection, budget measured in *sample usages*; verified abstract. Its finding that "no algorithm consistently outperforms random across budgets" independently supports the abstention logic.
- `arXiv:2503.01807` — **Large-Scale Data Selection for Instruction Tuning** (Ivison et al.), up to 2.5M samples from 5.8M pools: many methods **fall short of random** and degrade with larger pools; RDS+ wins. Verified abstract.
- `arXiv:2602.18584` — **GIST** (ICML 2026): selection by coupled-subspace gradient alignment, 25% of the compute time of the baseline. Verified abstract.
- `arXiv:2605.15691` — **SEED**: weighted independent set on a similarity graph. Verified abstract.
- `arXiv:2610.38315` — **LLM-PowerTrace**: 100 ms per-GPU power traces for fine-tuning/inference. Verified abstract. Measurement infrastructure, not a competitor.
- `arXiv:2608.08744` — carbon-surrogate regulariser inside fine-tuning (not data selection). Verified abstract.

### 3.1 Kill analysis

**KILL #1 — the brief's functional.** Rows 1 and 2 jointly cover it: row 1 owns "selection cost + training cost under one budget", row 2 owns "per-sample prices + budget + approximation guarantee". A joule-valued $p_j$ is a change of units, and a reviewer at an ML venue will say so in the first paragraph.

**KILL #2 — the brief's theorem shape.** Row 3 owns $(1-1/e)$ for instruction tuning. The guarantee is classical (Nemhauser–Wolsey–Fisher 1978; Sviridenko 2004 for knapsack). There is no new mathematics in restating it.

**KILL #3 — the green novelty.** Row 5 already shows empirically that accuracy per unit energy has an **interior optimum** in subset size (5% MNIST, 24% CIFAR-10), which is the most eye-catching empirical prediction the brief's framing would have made.

**What is *not* covered.** Rows 1–5 all assume the quality signal is known and noiseless. None of them asks how much energy must be spent to **establish** that a subset is better, none of them treats the validation set as a budget line, and none of them can say "abstain". Row 1 counts selection FLOPs; it does not count the statistical evidence needed to trust the result, and its budget is FLOPs, not joules.

**Therefore:** the original idea is at most a 6 by the brief's own rule ("same estimand and guarantee ⇒ at most 6"). The repair below is the nearest defensible move: **keep the green budget, change the object from "quality" to "certified quality", and let the certificate consume the budget.**

---

## 4. Theorem / bound shape

### 4.1 Assumptions

- **A1 (paired, bounded/sub-Gaussian differences).** Validation items are drawn i.i.d. from the target distribution; $D_i=\ell_i(\text{base})-\ell_i(S)$ are independent sub-Gaussian with variance proxy $\sigma^2$ (bounded $[0,1]$ for accuracy; empirically $\hat\sigma$ is reported).
- **A2 (pre-registered candidate family).** The selector inspects $M$ candidate subsets. For pointwise selectors $M$ is the number of distinct subsets scored; for adaptive/greedy selectors $M$ is the number of *decisions* that touch $V$ (see §4.4, Corollary 3b).
- **A3 (additive energy).** $E_{\text{tot}}=E_\pi(M)+a+\sum_{i\in S}e_i+c_v n_v$, with $a,e_i,c_v$ measured (§5). Batching/padding residual $\Delta_{\text{batch}}$ is measured and bounded; if it exceeds 10%, the budget is shrunk to $B-\Delta_{\text{batch}}^{\max}$.
- **A4 (non-trivial gain).** $\Delta(S)>0$ for at least one feasible $S$; otherwise the question is vacuous and the answer is abstain.

### 4.2 Theorem 1 — validity of the certified objective

For any (possibly data-dependent) choice among $M$ candidates,
$$
\Pr\Big\{\forall j\le M:\ \Delta(S_j)\ \ge\ \hat\Delta(S_j;V)-\sigma\sqrt{\tfrac{2\log(2M/\delta)}{n_v}}\Big\}\ \ge\ 1-\delta .
$$

**Proof sketch.** Hoeffding's inequality gives, for each fixed $j$, $\Pr\{\Delta(S_j)<\hat\Delta_j-t\}\le e^{-n_v t^2/(2\sigma^2)}$. Set $t=\sigma\sqrt{2\log(2M/\delta)/n_v}$ and union-bound over $j=1,\dots,M$. The bound is uniform over the $M$ candidates, so it survives the argmax — this is the step that a naive "report the best validation score" pipeline omits. With unknown $\sigma$, replace $t$ by the empirical-Bernstein penalty $\hat\sigma\sqrt{2\log(4M/\delta)/n_v}+7\log(4M/\delta)/(3(n_v-1))$ (Maurer–Pontil); the rate in $n_v$ is unchanged. $\square$

**Why this is not the trivial "use a test set" remark.** The bound is uniform over the *search*, not over one hypothesis. Its width therefore depends on how widely the selector searched, which is a design variable under the energy budget.

### 4.3 Theorem 2 — the certification tax (a dominance condition)

Let $\Delta^*_M=\max_{j\le M}\Delta(S_j)$ be the true best gain among the first $M$ candidates. On the $1-\delta$ event of Theorem 1, the **certification slack** of a width-$M$ search is bounded by
$$
\Delta^*_M-G_{\text{cert}}(M)\;\le\;\sigma\sqrt{\tfrac{2}{n_v}}\Big(\sqrt{\log\tfrac{2M}{\delta}}+\sqrt{\log\tfrac{2}{\delta}}\Big).
$$
Therefore, for $M_2>M_1$, the wider search can be **certified** as an improvement only if
$$
\Delta^*_{M_2}-\Delta^*_{M_1}\;>\;\underbrace{\sigma\sqrt{\tfrac{2}{n_v}}\Big(\sqrt{\log\tfrac{2M_2}{\delta}}-\sqrt{\log\tfrac{2M_1}{\delta}}\Big)}_{\text{certification tax }\tau}.
$$
So a wider search is certified-better **only if its true quality gain exceeds $\tau$**. $\tau$ grows only logarithmically in $M$: the price is paid as $n_v\propto\log(2M/\delta)$, not as $n_v\propto M$.

**Proof sketch.** Under the Theorem 1 event, $G_{\text{cert}}(M)=\hat\Delta_{\max}-\operatorname{pen}(M)\le\Delta(S_{\hat j})\le\Delta^*_M$. For the lower side, let $j^*=\arg\max_j\Delta(S_j)$; then $\Delta^*_M-\hat\Delta_{\max}\le\Delta^*_M-\hat\Delta_{j^*}$, and a single-candidate Hoeffding bound gives $\Delta^*_M-\hat\Delta_{j^*}\le\sigma\sqrt{2\log(2/\delta)/n_v}$ with probability $1-\delta/2$. Add $\operatorname{pen}(M)=\sigma\sqrt{2\log(2M/\delta)/n_v}$ and subtract the two widths. $\square$

**Why this is the honest form.** The theorem does not say wide search is bad; it says wide search is *unprovable* unless it buys more than $\tau$. At $\sigma=0.39$, $n_v=6{,}308$, $\delta=0.05$, going from $M=1$ to $M=10^7$ costs $\tau\approx1.8$ pp — the same order as the entire selection gain the field usually reports. At $\sigma=0.08$ the same move costs $\tau\approx0.4$ pp.

**Consequence (the number that matters).** With $\sigma=0.39$ (paired accuracy differences at ~45% accuracy and 0.7 inter-model correlation), $\Delta=2$ pp, $\delta=0.05$:

| search width $M$ | required $n_v$ | projected $E_{\text{eval}}$ | ratio to $E_{\text{train}}(500)$ |
|---|---|---|---|
| $1$ | 2,805 | 26 kJ | 1.9× |
| $10^2$ | 6,308 | 58 kJ | 4.2× |
| $10^7$ (greedy over 20k pool, $k=500$) | 15,063 | 139 kJ | 10.0× |

Projection: Pythia-410M, 512 tokens, $2N$ FLOPs/item forward, $6N$/token train, 5 TFLOP/s effective, 110 W. **These are FLOP-based projections, not measurements — the pilot replaces them.** With a low-variance metric ($\sigma=0.08$ nats, $\Delta=0.03$ nats) the same certificate needs $n_v\approx 60$–$150$, i.e. under 1.5 kJ. The choice of metric therefore decides whether certification is affordable at all.

### 4.4 Theorem 3 — the abstention floor

Certifying a gain $\Delta$ at width $M$ requires $n_v\ge 2\sigma^2\log(2M/\delta)/\Delta^2$, hence
$$
B_{\min}(\Delta,M)\;=\;\min_{k}\Big[E_\pi(M)+a+\bar e\,k+\frac{2c_v\sigma^2\log(2M/\delta)}{\Delta^2}\Big].
$$
If $B<B_{\min}(\Delta,M)$ for the smallest sensible $M$ (usually $M=1$, i.e. no search), **no certified gain of size $\Delta$ is purchasable within $B$**, and the joule-rational action is to abstain from the claim (or to spend the joules on training without selection and report an uncertified estimate). $B_{\min}$ is increasing in $\sigma$ and decreasing in $\Delta$; it is *convex* in $\Delta^{-1}$, so the cost of certifying small effects explodes.

**Proof sketch.** Rearrange the Hoeffding penalty for $n_v$; substitute into $E_{\text{tot}}$; minimise over $k$. $\square$

**Corollary 3a (metric asymmetry).** The decision-relevant metric (accuracy) has $\sigma\approx0.39$; a variance-reduced metric (paired NLL) has $\sigma\approx0.05$–$0.08$. Certification on accuracy costs $(0.39/0.08)^2\approx24\times$ more validation energy than certification on NLL. **A field that reports 2 pp accuracy gains on 400-item benchmarks is reporting effects it cannot afford to certify.**

**Corollary 3b (adaptivity tax).** For a greedy selector that scores every (example, position) pair, $M\approx nk$; the certificate therefore needs $\log(nk)$ rather than $\log k$, a factor $\approx2\times$ in $n_v$ for $n=20$k, $k=500$. Adaptive selectors pay this in addition to their larger $E_\pi$.

### 4.5 Why the $(1-1/e)$ route dies (and should be stated as a negative result)

The tempting theorem is: $f$ is monotone submodular, $E$ is modular, so cost-benefit greedy gives $(1-1/e)$. Three problems.

1. It is classical (Sviridenko 2004; Khuller–Moss–Naor 1999) and already used in this exact application by SPICE.
2. The ratio objective $f(S)/E(S)$ is **not** submodular, so the guarantee does not apply to "quality per joule" — the very thing the brief asks for.
3. Cardinality-proxy selection can be **arbitrarily bad** in the joule objective. Fix any $k$ chosen without reference to the cost vector. Explicit two-cluster instance: $w=(0.30,1.00)$, $\beta=(0.01,0.004)$, $e=(1,20/50/100)$, $U=200$. A pointwise quality ranker picks the expensive cluster (marginal $0.004>0.003$); the joule-optimal solution excludes it (density $2\times10^{-4}$ vs $3\times10^{-3}$ at $e_2=20$, falling to $4\times10^{-5}$ at $e_2=100$). Every size-$k$ set is either infeasible or worse; the ratios (computed, not estimated) are:

| $e_2$ | pointwise proxy gain | joule-optimal gain | ratio |
|---|---|---|---|
| 20 | 0.0392 | 0.2594 | **6.6×** |
| 50 | 0.0159 | 0.2594 | **16.3×** |
| 100 | 0.0080 | 0.2594 | **32.6×** |
| 500 | 0.0000 (no example affordable) | 0.2594 | **unbounded** |

So the correct statement is an **impossibility for cost-blind cardinality rules**: their competitive ratio against the joule objective is unbounded in the cost dispersion. That is a defensible "sign/bound/impossibility" result — but it is not *new* either: it is the cost-heterogeneity version of the pointwise-fails argument already made by QSGS and SEED. It belongs in the paper as a motivating lemma, not as the headline.

### 4.6 Counterexamples (load-bearing, not decorative)

**C1 — clustered validation items break Theorem 1.** Instruction pools contain near-duplicates and templated prompts; if validation items cluster, the effective sample size is $n_{\text{eff}}=n_v/(1+(m-1)\rho)$ with intra-cluster correlation $\rho$. With $m=5$, $\rho=0.4$, $n_{\text{eff}}=0.38\,n_v$: the certificate is ~1.6× too narrow and under-covers. **Fix:** cluster-robust penalty (replace $n_v$ by $n_{\text{eff}}$) and report the design effect. **Pilot check:** compute $\rho$ on the validation set before trusting any bound.

**C2 — heavy-tailed per-item differences.** A few items (very long outputs) dominate the mean; sub-Gaussianity fails and Hoeffding is invalid. **Fix:** empirical Bernstein plus 1% trimming, and report the sensitivity of $G_{\text{cert}}$ to trimming. If the answer moves by more than the effect, the certificate is not decision-grade.

**C3 — the stopping rule of Theorem 2 is only local.** If $\Delta^*_M$ jumps at large $M$ (a rare high-value cluster reachable only by wide search), $G_{\text{cert}}$ can rise again after a plateau. Then the correct policy is two-stage search (coarse screen, then deep screen), and a single global $M^*$ does not exist. **This is the honest failure mode of the repair:** the theorem gives a local dominance condition, not a global optimum.

**C4 — the whole thing can be vacuous.** If $\sigma$ is large and $\Delta$ small, $B_{\min}$ exceeds any budget a small team has, and the paper's own conclusion is "certified green selection is infeasible at 410M scale". That is a publishable negative result, but it is a different paper from the one the brief imagined.

---

## 5. Dataset and energy protocol

### 5.1 Data

| pool | size | role |
|---|---|---|
| Alpaca (`tatsu-lab/alpaca`) | 52k | short, templated; low $e_i$, high redundancy |
| Dolly 15k (`databricks/databricks-dolly-15k`) | 15k | 7 native task types; long outputs; high $e_i$ |
| FLAN v2 subset (`allenai/flan_v2`, task-stratified) | 50k | task diversity; medium $e_i$ |

Working pool: 10k Alpaca + 6k Dolly + 4k FLAN = 20k, stratified by native task type. Held-out evaluation: 2,000 instruction items disjoint from the pool (paired NLL), plus ARC-Easy (400), HellaSwag (400), PIQA (300) for the decision-relevant secondary metric.

### 5.2 Models and training

- **Primary:** Pythia-410M (ungated, deterministic, no licence friction). **Replication:** Qwen2.5-0.5B (ungated). Optional 1.5B on the 24 GB rental for the camera-ready.
- LoRA $r=16$, $\alpha=32$, dropout 0.05, targets q,k,v,o,gate,up,down; bf16; batch 8 × grad-accum 2; lr $2\times10^{-4}$ cosine, 3% warmup; 1–3 epochs; max length 512.
- **Packing off** for the per-example energy accounting (padding is the confound); packing on in a separate arm to measure $\Delta_{\text{batch}}$.

### 5.3 Energy measurement

1. `nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 100` in a side process; integrate $P(t)$ by trapezoid over the run window; subtract the idle baseline measured for 60 s immediately before.
2. **Fixed overhead $a$:** regress measured joules on padded tokens over five run lengths (0, 125, 250, 500, 1000 examples, 1 epoch); $a$ is the intercept, $\kappa$ the slope. Report $R^2$ and residual SD.
3. **Cross-check:** `codecarbon` in offline mode (no API) on three runs; report the ratio and its CI. If `codecarbon` is unavailable, say so.
4. **Laptop-3060 honesty clause:** mobile GPUs often expose coarse or rounded `power.draw`, and some drivers update at ~1 Hz. Validate against a wall meter on three runs if one is available; otherwise report the NVML-based figure as a *proxy* with its calibration error, and never present it as a metered value. If `power.draw` is unavailable, fall back to `utilization.gpu × TDP` and label it as such.
5. **Carbon:** fixed regional average from a public source (Ember/IEA annual grid factor; report the exact kgCO$_2$e/kWh and year). Per the brief, temporal variation is out of scope; state that a fixed average cannot capture diurnal variation and therefore the carbon numbers are location-comparative only.
6. **Report separately:** $E_\pi$, $E_{\text{train}}$, $E_{\text{val}}$, and $E_{\text{tot}}$. The whole thesis depends on the third term being visible.

### 5.4 Statistics

- Paired design (same items for base and tuned model) — this is what makes $\sigma$ small enough to certify anything.
- $\sigma$ estimated per run; $\delta=0.05$; $M$ logged explicitly by the selector wrapper (every validation query increments $M$).
- Report $G_{\text{cert}}$, $n_v$, $M$, $\tau$, $B_{\min}$, and the design effect from C1. A result without these five numbers is not admissible under the repair.

---

## 6. Pilot — ≤ 2 hours on the 3060

**Purpose:** decide whether the certification term is (i) measurable, (ii) large enough to change the decision. Not to produce a paper result.

| time | step | runs | output |
|---|---|---|---|
| 0–10 min | CPU simulation of Theorems 1–3 (numpy): verify the $\sqrt{\log M/n_v}$ law, calibrate the design for the real runs | 0 | $M$ grid, $n_v$ grid, predicted $\tau$ |
| 10–30 min | energy calibration: 5 LoRA runs at 0/125/250/500/1000 examples, 1 epoch | 5 | $a$, $\kappa$, $R^2$, repeat CV |
| 30–85 min | 8 runs at $k=500$: random ×2 seeds, BM25 ×2, embedding/RDS+ ×2, one gradient selector on a 2k subsample ×2 | 8 | paired $D_i$ per run |
| 85–105 min | evaluate all 8 checkpoints on a shared $V$ (nested sizes 125/250/500/1000) and on the 2k test set | 8×4 evals | $\hat\Delta$, $\sigma$, optimism vs $M$ |
| 105–120 min | fit $\tau$, $B_{\min}$, design effect; compute go/no-go | 0 | decision table |

Total: 13 training runs, ~90 min wall-clock with tokenisation and evaluation overhead.

**Bootstrap trick (keeps it under 2 h):** the 8 checkpoints are fixed; bootstrap the per-item differences to simulate many candidate subsets, then take maxima over $M\in\{1,2,4,8,\dots,1024\}$. This estimates the optimism term without 1,024 training runs. It is a pilot-grade estimate, not a substitute for the real sweep.

### Kill criteria (each kills a different leg)

- **K1 — measurement.** Energy fit $R^2<0.90$, or repeat-run CV of joules $>15\%$: joules are not certifiable at this scale → **KILL the joule-budget claim** (the fallback is a FLOP-budget paper, which is row 1's territory — do not write it).
- **K2 — effect.** Optimism at $M=8$, $n_v=500$ is $<0.5$ pp: the winner's-curse term is negligible → **KILL the search-width story**; the paper collapses to a measurement note.
- **K3 — power.** Required $n_v$ for $\Delta=1$ pp exceeds 5,000 paired items at $\sigma$ measured: no affordable certificate at 410M scale → **KILL**, or escalate to a 1.5B model where effects are larger (but then compute cost rises and the green claim weakens).
- **K4 — signal.** $\Delta(k=500\text{ selected})$ vs random is $<0.5$ pp on the paired NLL: there is nothing to certify → **KILL** (and note that this replicates `2503.01807`).

**Pass condition:** K1–K4 all pass, and the projected $E_{\text{val}}/E_{\text{train}}$ at the certified $n_v$ is $\ge1$. That last condition is the paper's headline; if it is $<1$, the "price of proof" claim is not true at this scale.

---

## 7. Scores

Rubric: Novelty 25%, Method 20%, Math 15%, Experiments 10%, Dataset 10%, Q1 fit 10%, Feasibility 5%, Compute 5%. Core = mean(Novelty, Method, Math, Experiments). Independence is separate.

| factor | weight | G4 (original) | G4′ (repair) | justification for G4′ |
|---|---|---|---|---|
| Novelty | .25 | 5.5 | **8.0** | New functional: certified gain per joule with evidence as a budget line. No direct pre-emption found; adjacent to CCDS (cost) and MtS (prices) but neither buys evidence. |
| Method | .20 | 6.0 | **7.5** | Paired design + uniform certificate + measured energy. Sound but not technically deep; the hard part is measurement discipline. |
| Math | .15 | 6.0 | **8.0** | Validity theorem, certification tax, abstention floor, adaptivity corollary. Rates and a sign result; ingredients classical, combination and object new. |
| Experiments | .10 | 6.0 | **6.5** | 410M/0.5B only; effects small; but the energy/evidence measurement is the contribution and is robust. Risk: K2/K4. |
| Dataset | .10 | 7.0 | **7.0** | Public pools; no new dataset. Mitigation: release the paired-evaluation + joule-annotated harness as an artifact. |
| Q1 fit | .10 | 6.5 | **8.0** | Green venues reward energy accounting, carbon numbers, and an explicit "don't spend" recommendation. |
| Feasibility | .05 | 8.0 | **9.0** | 13-run pilot; full study ~120 GPU-hours; no 7B requirement for the core claim. |
| Compute | .05 | 8.5 | **9.5** | 410M LoRA fits a 6 GB 3060; evaluation is forward-only. |

**Arithmetic.**

- Original: $0.25(5.5)+0.20(6.0)+0.15(6.0)+0.10(6.0)+0.10(7.0)+0.10(6.5)+0.05(8.0)+0.05(8.5)$
  $=1.375+1.200+0.900+0.600+0.700+0.650+0.400+0.425=\mathbf{6.25}$.
  Core $=(5.5+6.0+6.0+6.0)/4=\mathbf{5.875}$.
- Repair: $0.25(8.0)+0.20(7.5)+0.15(8.0)+0.10(6.5)+0.10(7.0)+0.10(8.0)+0.05(9.0)+0.05(9.5)$
  $=2.000+1.500+1.200+0.650+0.700+0.800+0.450+0.475=\mathbf{7.775}$.
  Core $=(8.0+7.5+8.0+6.5)/4=\mathbf{7.50}$.

**Independence: 7/10** (revised after reading the sibling files present on 2026-11-08).
- vs **G1** (carbon-budgeted cascade, itself killed at 5.8): different layer (training vs inference), different estimand, different math object. Shares only the "finite-sample guarantee" style — 7/10 pairwise.
- vs **G2** (retrieval depth): disjoint — 9/10.
- vs **G3′** (certified quantisation frontier): **6/10 pairwise** — this is the real overlap. Both repairs use a "certified budget under noise" device; G3′ certifies an inference-time accuracy frontier under *energy-measurement* noise, G4′ certifies a training-time gain under *evaluation* noise with evidence as a budget line. Different objects, same statistical move. Do not run both as separate papers.
- vs **G5** (CPCA benchmark): both measurement-flavoured; G5 measures inference joules per correct answer, G4′ measures training/evidence joules per certified gain — 8/10.

Portfolio-level: **7/10**. It is the only idea in the set whose headline number is an *abstention threshold*, but it is not independent enough from G3′ to justify a second "certified" paper.

---

## 8. Verdict

**Original G4: KILL.** Score 6.25 weighted / 5.88 core, consistent with the brief's rule. The functional is covered by `2410.16208` + `2610.14028`, the guarantee by `2601.23155`, and the green empirical claim by the Haddaji et al. chapter. Writing it would produce a paper that a reviewer can pre-empt with three citations.

**G4′ (certified joule-budgeted selection): CONDITIONAL GO.** 7.78 weighted / 7.50 core — below the 8 bar, and honestly so, because Experiments is capped by the 410M scale and by the real risk that the optimism term is small.

**What would move it to 8+ (and make it a GO):**
1. Pilot shows $\sigma_{\text{paired}}\le0.10$ on the NLL metric and $\ge1.5$ pp optimism at $M=8$, $n_v=500$ → Experiments 8.0, Novelty 8.5 → weighted ≈ 8.05.
2. The measured $E_{\text{val}}/E_{\text{train}}$ at the certified $n_v$ is $\ge2$ → the headline "proof costs more than the thing it proves" is real and quotable.
3. A design-effect check (C1) shows $\rho<0.2$, so the certificate is not undermined by near-duplicates.

**What would kill it outright:** K1 (joules unmeasurable on the laptop 3060), K2 (optimism negligible), or K3 (certification unaffordable at any useful effect size). In the last case the salvage is a negative result — "no certified green selection is affordable at 410M scale" — which is publishable in a green venue but is a different, smaller paper. Do not let the salvage be presented as the original idea.

**Strategic note.** G4 is the weakest of the five directions on current evidence, not because the question is bad but because 2025–2026 saturated the *selection* half of it. If the pilot fails, redistribute the effort to G2/G3. If the pilot passes, do **not** run G4′ as a second certified-budget paper alongside G3′ (7.975 weighted): the statistical device is the same and a 3–4 person team cannot defend two of them. The better move is to fold G4′'s distinctive asset — the measured separation of selection, training and evidence joules — into G3′ as its training-side section, and keep the abstention threshold as the combined paper's policy contribution. Standalone, G4′ is the most *distinctive* green paper in the portfolio because it is the only one that can tell a practitioner not to spend; portfolio-wise, it is the one to merge.

---

## 9. References

**Closest five (verified 2026-11-08).**
1. J. O. Yin, A. M. Rush. *Compute-Constrained Data Selection.* arXiv:2410.16208; ICLR 2025.
2. L. Zhang, J. Wang, B. Li. *Pick the Mixture, Not the Samples: Combinatorial Data Selection for LLMs.* arXiv:2610.14028, 2026.
3. P. Chang et al. *SPICE: Submodular Penalized Information-Conflict Selection for Efficient Large Language Model Training.* arXiv:2601.23155; ICLR 2026.
4. Z. Zhao, Y. Kawanishi, H. Zheng, M. E. Papka, Y. Yan. *Query-based Submodular Data Selection for Efficient Instruction Tuning.* arXiv:2611.04059, 2026.
5. O. Haddaji, O. Brun, B. Prabhu. *Balancing Accuracy and Energy: An Empirical Study of Optimal Subset Size Selection.* LNCS 16424, 2026. doi:10.1007/978-3-032-18494-8_11.

**Context.**
6. H. Ivison et al. *Large-Scale Data Selection for Instruction Tuning.* arXiv:2503.01807, 2025.
7. *Computational Budget Should Be Considered in Data Selection.* arXiv:2510.16806, 2025.
8. G. Min et al. *GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry.* arXiv:2602.18584; ICML 2026.
9. *SEED: Targeted Data Selection by Weighted Independent Set.* arXiv:2605.15691, 2026.
10. *LLM-PowerTrace: A Dataset of Time-Resolved GPU Power Traces for AI Compute.* arXiv:2610.38315, 2026.
11. *Can We Optimize the Performance-Carbon Emission Break-Even Point? The Quest for Greener LLMs.* arXiv:2608.08744, 2026.
12. X. Wang, C. Na, E. Strubell, S. Friedler, S. Luccioni. *Energy and Carbon Considerations of Fine-Tuning BERT.* arXiv:2311.10267, Findings of EMNLP 2023.

**Theory.**
13. G. L. Nemhauser, L. A. Wolsey, M. L. Fisher. An analysis of approximations for maximizing submodular set functions — I. *Mathematical Programming* 14, 1978.
14. M. Sviridenko. A note on maximizing a submodular set function subject to a knapsack constraint. *Operations Research Letters* 32(1), 2004.
15. S. Khuller, A. Moss, J. Naor. The budgeted maximum coverage problem. *Information Processing Letters* 70(1), 1999.
16. A. Maurer, M. Pontil. Empirical Bernstein bounds and sample-variance penalization. *COLT* 2009.
17. J. Hoffmann et al. Training compute-optimal large language models. arXiv:2203.15556, 2022.

**Status of claims.** Every ID, venue, and capability statement in §3 is verified at abstract level; §3's HTML-derived claims (Eq. 1 of `2610.14028`, App. B of `2410.16208`) are verified at full-text level. The energy numbers in §4.3/§4.4 and the gap ratios in §4.5 are **computed projections on this CPU**, not measurements; they are labelled as such and the pilot replaces them. The Haddaji et al. numerical results (5%, 24%) are from the publisher abstract and are marked unverified beyond that.
