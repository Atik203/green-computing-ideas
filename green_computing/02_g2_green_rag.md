# G2 — Energy-Optimal Retrieval Depth (Green RAG)

**Date:** 8 November 2026  
**Status:** **CONDITIONAL** (pre-pilot)  
**Scores:** Core **7.0** · Weighted **7.35** · Independence **3/5**  
**Closest paper:** Pan et al., *Towards Automatically Optimizing Retrieval Augmented AI Systems* (NeurIPS 2025 MLForSys Workshop; OpenReview `an17DUVvuS`)  
**One-line verdict:** Not a KILL — nobody has the exact estimand plus guarantee — but adaptive-\(k\) and RAG energy profiling are crowded, so the paper must be a **certification theory for \(k^\star\)**, not another "we measured the energy curve and picked \(k\)" study.

---

## 0. Verdict first

The naive version of G2 — "measure joules per correct answer for several \(k\), report the minimum, call it energy-optimal retrieval depth" — is a **6/10 measurement paper** in November 2026. Pan et al. already show configurations within \(\le 3\%\) accuracy differing by up to **20.2×** in energy, RAGMark already sweeps retrieval depths with per-stage power, and adaptive-\(k\) has at least four 2025–2026 methods. The naive version is not an 8.

The defensible version is narrower and stronger:

1. **An exact optimal-depth identity.** At an interior optimum, the depth-elasticity of energy equals the accuracy elasticity:
   \[
   \frac{k^\star J'(k^\star)}{J(k^\star)}=\frac{k^\star A'(k^\star)}{A(k^\star)}.
   \]
   For affine energy, the left side is the variable retrieval/context energy share. This is a scale-free, testable diagnostic: at \(k^\star\), the context-side energy share is pinned to how fast accuracy is still improving.

2. **A price-of-static-depth bound.** For any static \(k\), there is a heterogeneous query distribution on which the best static depth is arbitrarily worse than a query-conditional policy, at equal accuracy. Adaptivity is not cosmetic; it is necessary.

3. **A certification-cost threshold.** Finding \(k^\star\) itself costs energy. Calibration is worth it only if the deployment has enough queries:
   \[
   N>\frac{C_{\mathrm{cert}}}{A(k^\star)\bigl[E(k_0)-E(k^\star)\bigr]}.
   \]
   For small deployments, the greenest policy is **not to calibrate**. This is the most distinctive, decision-changing result.

The empirical study then tests these three claims, including a **causal manipulation of the retrieval/generation energy split**: add a fixed per-query energy tax (raises \(a\)) and a per-passage energy tax (raises \(b\)), and check that \(k^\star\) moves in the predicted direction without changing the reader's accuracy. That is a genuine test of the theory, not another curve.

**Verdict: CONDITIONAL.** Proceed to the 2-hour pilot only with the certification framing. If the pilot shows a flat or boundary \(k^\star\), kill the depth story and fold the measurement into G5.

---

## 1. Plain-language problem

Retrieval-augmented generation spends energy twice: once to fetch and score passages, and again to put those passages through the reader's context window. Retrieving more passages can raise accuracy by covering more evidence, but it also raises prefill energy and can distract the reader. The green question is not "what \(k\) maximizes accuracy?" but "what \(k\) minimizes **joules per correct answer**, and is it worth the energy to find out?"

The decision changes in three places: (i) a deployed RAG service chooses a per-query retrieval depth; (ii) an engineer decides whether to run an offline calibration sweep at all; (iii) a hardware/platform team decides whether to move retrieval or generation to a different energy or carbon regime.

---

## 2. Exact estimand, functional, and decision

### 2.1 Notation

| Symbol | Meaning |
|---|---|
| \(q\sim P\) | query from a task distribution |
| \(k\in\mathcal{K}=\{0,1,\dots,K_{\max}\}\) | retrieval depth (number of passages) |
| \(J_k(q)\ge 0\) | measured joules for query \(q\) at depth \(k\) |
| \(C_k(q)\in\{0,1\}\) | answer correctness |
| \(A(k)=\mathbb{E}[C_k]\) | accuracy at depth \(k\) |
| \(J(k)=\mathbb{E}[J_k]\) | mean joules per query at depth \(k\) |
| \(E(k)=J(k)/A(k)\) | **joules per correct answer** |
| \(A_{\min}\) | pre-registered accuracy floor |
| \(k^\star=\arg\min_{k\in\mathcal{K}:A(k)\ge A_{\min}}E(k)\) | energy-optimal feasible depth |

The primary estimand is the ratio of expectations, not the expectation of per-query ratios:
\[
E(k)=\frac{\mathbb{E}[J_k(q)]}{\mathbb{E}[C_k(q)]}
=\frac{\text{total joules over the query population}}{\text{total correct answers over the query population}}.
\]
This is the quantity a service operator actually pays per correct answer. Report the per-query ratio distribution as a secondary diagnostic; it is not the estimand.

### 2.2 Why the ratio needs a correctness gate

A ratio can be gamed by failing cheaply. Consider a distribution with easy queries (probability \(1-\varepsilon\)) answerable at \(k=0\), and hard queries (probability \(\varepsilon\)) answerable only at \(k=K\). With \(J(0)=a\) and \(J(K)=a+bK\):
\[
E(0)=\frac{a}{1-\varepsilon},\qquad E(K)=a+bK.
\]
For \(\varepsilon<bK/(a+bK)\), \(E(0)<E(K)\): the unconstrained ratio prefers **no retrieval**, because it accepts errors on the hard queries. That is not a green RAG policy; it is a policy that hides its errors in the denominator.

Therefore the decision is always **correctness-gated**:
\[
k^\star=\arg\min_{k\in\mathcal{K}}\;E(k)\quad\text{subject to}\quad A(k)\ge A_{\min}.
\]
Pre-register \(A_{\min}\). Two defensible choices: (i) non-inferiority to the current deployment, \(A_{\min}=A(k_0)\); (ii) a fraction of the best achievable accuracy, \(A_{\min}=0.98\max_k A(k)\). Report both; the conclusions must not depend on the choice.

### 2.3 Energy decomposition and the retrieval/generation split

For a query at depth \(k\):
\[
J_k(q)=J^{\mathrm{ret}}_k(q)+J^{\mathrm{pre}}_k(q)+J^{\mathrm{dec}}_k(q)+J^{\mathrm{fix}}(q).
\]
Here \(J^{\mathrm{ret}}\) is BM25/index/fetch, \(J^{\mathrm{pre}}\) is the prefill of the query plus \(k\) passages, \(J^{\mathrm{dec}}\) is decoding, and \(J^{\mathrm{fix}}\) is the amortized model/idle cost. With passage length \(\ell\) tokens, query length \(T_q\), and output length \(T^{\mathrm{out}}_k\):
\[
J^{\mathrm{ret}}_k=j_r k+o(k),\qquad
J^{\mathrm{pre}}_k=j_p(T_q+\ell k),\qquad
J^{\mathrm{dec}}_k=j_d T^{\mathrm{out}}_k.
\]
The affine approximation is
\[
J(k)=a+bk,\qquad
a=J^{\mathrm{fix}}+j_pT_q+j_dT^{\mathrm{out}}_0,\qquad
b=j_r+j_p\ell+j_d\frac{\partial T^{\mathrm{out}}}{\partial k}.
\]
If output length is independent of \(k\), \(b=j_r+j_p\ell\) is the **per-passage energy**. If more context shortens answers, \(b\) is smaller; if it lengthens them, \(b\) is larger. Measure \(T^{\mathrm{out}}_k\); do not assume it.

Define the **context-side share**
\[
S(k)=\frac{J^{\mathrm{ret}}(k)+J^{\mathrm{pre}}(k)}{J(k)}
\]
and the **retrieval-only share** \(S_r(k)=J^{\mathrm{ret}}(k)/J(k)\). The context-side share is the relevant green lever in GPU RAG because prefill of retrieved tokens usually dominates fetch. Report both; conflating them is a common measurement error.

### 2.4 Decision

The paper studies three nested decisions:

1. **Static depth:** choose one \(k\) for a population, \(k^\star_{\mathrm{static}}=\arg\min_{k:A(k)\ge A_{\min}}E(k)\).
2. **Query-conditional depth:** choose \(k(q)\) from query features, \(k^\star(q)=\arg\min_{k:A(k\mid q)\ge A_{\min}}E(k\mid q)\). The benchmark is the oracle policy that knows \(k^\star(q)\).
3. **Whether to calibrate:** choose between a default depth \(k_0\) and paying \(C_{\mathrm{cert}}\) joules to estimate \(k^\star\). This is the meta-decision that most green-RAG papers ignore.

The theory below covers all three.

---

## 3. Closest five papers at the functional level (2024–2026)

### 3.1 The five

| # | Paper | ID | Verified | Functional overlap | What it does **not** have |
|---|---|---|---|---|---|
| P1 | Pan, Arabzadeh, Jacob, Kazhamiaka, Choukse, Zaharia, *Towards Automatically Optimizing Retrieval Augmented AI Systems* | OpenReview `an17DUVvuS`; NeurIPS 2025 MLForSys Workshop | **Verified** via OpenReview API (title, authors, abstract). PDF blocked by Cloudflare; full text unverified. | Maps the joint accuracy–energy landscape of RAG workflows; configurations within \(\le 3\%\) accuracy differ up to **20.2×** in energy; predicts accuracy from configuration knobs to prune the space. | No \(E(k)=J/\text{correct}\), no \(k^\star\), no query-conditional policy, no minimax/certification guarantee. Workshop paper, FinanceBench/FRAMES, cloud models. |
| P2 | Hwang, *Query Complexity–Aware Adaptive Retrieval for Cost-Efficient Retrieval-Augmented Generation* | OpenReview `ObkC4BCAFt` (direct upload, 28 Jan 2026) | **Verified** via OpenReview page/abstract. **Unrefereed** direct upload. | Adapts retrieval depth and effective context from an estimated query-difficulty signal; comparable accuracy to larger fixed-\(k\) baselines at lower retrieval/token cost. | Cost is retrieval/token cost, not joules; no energy measurement; no \(k^\star\); no theorem or guarantee. |
| P3 | Hashemi, Rühle, Rajmohan, *Cost-Aware Retrieval-Augmentation Reasoning Models with Adaptive Retrieval Depth* | **arXiv:2510.15719** (17 Oct 2025) | **Verified** via arXiv abs. | Dynamically adjusts the length of the retrieved document list; cost-aware advantage function for RL; 7 QA datasets; ~16–20% lower latency, ~5% higher EM. | Latency/cost, not energy; no \(k^\star\) or finite-sample guarantee; reasoning-model setting, not a 6 GB reader. |
| P4 | Feric, Taherin, Ren, Wang, Dy, Kaeli, *RAGMark: A Comprehensive Framework for Benchmarking Retrieval-Augmented Generation Systems* | **arXiv:2609.05760** (4 Sep 2026) | **Verified** via arXiv abs. | Modular benchmark with per-stage latency, GPU utilization, memory, power, TTFT, throughput, quality; sweeps five RAG workloads across retrieval depths, model scales, reranking, compression, vector DBs; reports up to 66% lower energy from context reduction. | A benchmark, not an optimizer; no \(k^\star\), no guarantee; multi-GPU small-scale environment, not a laptop 3060. |
| P5 | Song, Fang, Li, Xia, Wang, *Tail-Aware Adaptive-k: Query-Adaptive Context Selection for Retrieval-Augmented Generation* | **DOI 10.1007/978-3-032-37667-1_12**; ECML PKDD 2026, LNCS LNAI 16944, pp. 193–209 | **Verified** via Springer/Crossref. | Training-free adaptive \(k\) via knee detection + EVT tail testing; under mild monotone-likelihood-ratio assumptions yields a stable query-adaptive cutoff; near-oracle F1 within 2–3% on WebQuestions, 2WikiMultiHopQA, MuSiQue. | Objective is retrieval quality/tokens, not joules per correct answer; no energy measurement; the theorem is about cutoff stability, not energy optimality; no minimax/certification. |

### 3.2 Why these five, and not the obvious others

- **Taguchi, Maekawa, Bhutani (EMNLP 2025, DOI 10.18653/v1/2025.emnlp-main.1017)** is the strongest adaptive-\(k\) baseline: a single-pass similarity threshold, up to 10× fewer tokens, no tuning. It is a near miss rather than a top-5 threat because it has no energy measurement and no guarantee. It must be a baseline in the experiments.
- **Ruf & Detyniecki, *The Cost of Context* (arXiv:2610.19819)** characterizes context-scaling energy and the prefill/decode disparity. It is a component of the split, not a depth optimizer.
- **Zhao, *When Is Additional Retrieval Worthwhile?* (arXiv:2610.37358)** is a cost-aware two-action retrieval policy (dense+rerank vs hybrid), with a pre-registered threshold and a null utility result. It is close in spirit but not a \(k\)-sweep and not energy.
- **Anand et al., *One Size Does Not Fit All!* (arXiv:2609.17709)** routes retriever/generator configurations by query performance prediction; latency, not joules, and not depth.
- **Gao, Wang, Yu, *Token Arena* (arXiv:2605.00300)** already uses **joules per correct answer** as a headline composite, at endpoint granularity. This is the clearest evidence that the *metric* is not new. It does not optimize retrieval depth.
- **Kashaniyan & Jannesari, *ATEST* (arXiv:2610.25519)** is energy-aware adaptive test-time scaling with an online stopping rule. It is the closest *adaptive energy stopping* paper, but it allocates samples, not passages, and has no formal guarantee.
- **Khan, N J, Soni, *GreenFaith-RAG* (DOI 10.1109/icati69399.2026.11604335)** is an energy-monitored, confidence-aware RAG framework. Metadata verified via Crossref; abstract only partially verified (paywalled). It routes queries/models, not retrieval depth; functionally closer to G1.
- **Huo, Parry, Ganguly, Clarke, *Optimal Re-Ranking Depth* (DOI 10.1145/3805712.3809953, SIGIR 2026)** formulates query-specific optimal reranking depth and shows it is predictable. It is about reranking, not retrieval depth, and has no energy analysis. A reviewer may still cite it, so address it.
- **Dhakal, Pokharel, Adhikari, *Energy-Efficient RAG: A Systematic Review* (DOI 10.59324/ejaset.2026.4(5).11)** already lists adaptive retrieval and context management as energy levers, with 20–60% savings. It is a survey, not a method, but it kills the claim that "adaptive retrieval for energy" is unexplored.

### 3.3 KILL check: same estimand + same guarantee?

| Paper | Estimand | Guarantee |
|---|---|---|
| P1 Pan et al. | energy per configuration/query vs accuracy | none (predictive regressors) |
| P2 Hwang | retrieval/token cost vs accuracy | none |
| P3 Hashemi et al. | latency/cost vs EM | none |
| P4 RAGMark | per-stage energy/latency vs quality | none |
| P5 TAA-\(k\) | retrieval quality vs \(k\) | cutoff stability under monotone likelihood ratio |

**We found no 2024–2026 paper with \(E(k)=\mathbb{E}[J_k]/\mathbb{E}[C_k]\) plus a correctness gate, a certified \(k^\star\), and a finite-sample/minimax guarantee.** The closest is P1 on the outcome space and P5 on the decision variable, but neither combines them. Therefore: **not a KILL** (subject to a re-run of the search before submission).

**Nearest defensible repair.** If P1 were extended to report \(E(k)\) over \(k\), the empirical contribution would collapse. The repair is not to measure harder; it is to move the contribution to the certification theory: the identity, the price-of-static-depth bound, and the calibration-cost threshold. That is the recommended framing now, not a post-hoc fallback. If the team is unwilling to prove those results, the idea should be killed or merged into G5.

---

## 4. Theorem/bound shape and proof sketch

### 4.1 Theorem 1 — optimal-depth identity, unimodality, comparative statics

**Assumptions.**
- **(A1) Noisy coverage:** \(A(k)=1-(1-A_0)(1-p)^k\), with \(A_0=A(0)\in[0,1)\), \(p\in(0,1]\). Each additional passage independently supplies the needed evidence with probability \(p\), given the reader can use it. This is a stylized model, not a claim about all retrievers; it yields testable predictions.
- **(A2) Affine energy:** \(J(k)=a+bk\), \(a,b>0\).

**Theorem.** Let \(E(k)=J(k)/A(k)\) on \(k\in\mathcal{K}=\{0,\dots,K_{\max}\}\).

1. **Unimodality.** \(E\) is strictly decreasing then strictly increasing; the minimizer \(k^\star\) is unique (possibly at a boundary).
2. **Identity.** If \(0<k^\star<K_{\max}\), then
   \[
   bA(k^\star)=(a+bk^\star)A'(k^\star)
   \quad\Longleftrightarrow\quad
   \frac{k^\star J'(k^\star)}{J(k^\star)}=\frac{k^\star A'(k^\star)}{A(k^\star)}.
   \]
   For affine \(J\), the left side is \(bk^\star/(a+bk^\star)\), the variable energy share at the optimum.
3. **Comparative statics.** \(k^\star\) is nondecreasing in \(a\) and nonincreasing in \(b\).
4. **Boundary.** If \(A_0>0\), \(k^\star=0\) iff
   \[
   \frac{b}{a}\ge \frac{(1-A_0)\bigl(-\ln(1-p)\bigr)}{A_0}.
   \]
   If \(A_0=0\), \(k^\star>0\) unless \(\mathcal{K}=\{0\}\).

**Proof sketch.** Let \(s=1-p\), \(L=-\ln s>0\), \(B=1-A_0\). Then \(A(k)=1-Bs^k\) and \(A'(k)=BLs^k\). Write
\[
E'(k)=\frac{N(k)}{A(k)^2},\qquad
N(k)=b\bigl(1-Bs^k\bigr)-(a+bk)BLs^k.
\]
Differentiate:
\[
N'(k)=bBLs^k-\bigl[bBLs^k-(a+bk)BL^2s^k\bigr]
=(a+bk)BL^2s^k>0.
\]
So \(N\) is strictly increasing and crosses zero at most once, from negative to positive. Hence \(E'\) has the sign of \(N\): \(E\) decreases then increases. The interior first-order condition is \(N(k^\star)=0\), which rearranges to \(bA=(a+bk)A'\), and dividing by \(JA\) gives the identity. For the comparative statics, \(N_a=-A'<0\) and \(N_k>0\), so \(\partial k^\star/\partial a=-N_a/N_k>0\). At \(N=0\), \(N_b=A-kA'=(a/b)A'>0\), so \(\partial k^\star/\partial b=-N_b/N_k<0\). The boundary condition is \(N(0)\ge 0\). \(\square\)

**What the identity buys.** It is scale-free: multiplying all energy costs by a constant leaves \(k^\star\) unchanged. It is also a diagnostic: if the measured context-side share at \(k^\star\) is much larger than the measured accuracy elasticity, the deployment is over-retrieving; if much smaller, it is under-retrieving. This is a stronger, more falsifiable claim than "the curve is U-shaped."

**Caveat.** The comparative static in \(a\) holds holding \(A\) fixed. A larger reader raises \(a\) but also changes \(A_0\) and the shape of \(A\); the net effect on \(k^\star\) is ambiguous. The clean test is the **energy-tax manipulation** in §5.4, which changes \(a\) or \(b\) without changing the reader.

### 4.2 Corollaries — split, scale, carbon

**Corollary 1 (split).** Write \(a=a_g+a_{r0}\) and \(b=b_g+b_r\), where subscripts \(g\) and \(r\) denote generation-side and retrieval-side energy. Holding the other terms fixed, \(k^\star\) increases with \(a_g\) and \(a_{r0}\), and decreases with \(b_g\) and \(b_r\). Intuition: expensive fixed generation makes retrieval more attractive; expensive per-passage retrieval makes it less attractive.

**Corollary 2 (scale invariance).** If \(J\mapsto\lambda J\) for \(\lambda>0\), then \(E\mapsto\lambda E\) and \(k^\star\) is unchanged. Hardware that scales retrieval and generation equally does not change the energy-optimal depth; hardware that changes the **split** does. This is why a CPU-vs-GPU comparison is scientifically interesting and a faster GPU with the same energy profile is not.

**Corollary 3 (carbon).** If retrieval and generation have carbon intensities \(\kappa_r,\kappa_g\) (gCO\(_2\)e/J), replace \(a,b\) by
\[
a^\kappa=\kappa_r a_{r0}+\kappa_g a_g,\qquad
b^\kappa=\kappa_r b_r+\kappa_g b_g.
\]
The same theorem applies. If \(\kappa_r=\kappa_g\), carbon-optimal \(k\) equals energy-optimal \(k\); carbon changes the decision only when retrieval and generation have different carbon intensities. This is an honest limitation of most "carbon-aware RAG" claims: a constant grid factor is a unit conversion, not a new optimization.

### 4.3 Theorem 2 — price of static depth (impossibility/lower bound)

**Theorem.** For any \(K\ge 1\), there exists a query distribution \(P_\varepsilon\) such that the best static depth meeting accuracy \(A_{\min}=1\) is \(k_s=K\), the oracle query-conditional policy also meets \(A=1\), and
\[
\frac{E(k_s)}{E(\text{oracle})}
=\frac{a+bK}{a+\varepsilon bK}
\;\xrightarrow[\varepsilon\to 0]{}\;
1+\frac{bK}{a}.
\]
For any \(M>1\), choosing \(a<bK/(M-1)\) (e.g., \(a=bK/M\)) makes the ratio exceed \(M\) for \(\varepsilon\) small enough. Hence no static depth is minimax-optimal over heterogeneous queries, and the price of ignoring heterogeneity is unbounded.

**Proof.** Let a fraction \(1-\varepsilon\) of queries be "easy": correct at \(k=0\) and at \(k=K\), with energy \(a\) and \(a+bK\) respectively. Let a fraction \(\varepsilon\) be "hard": incorrect at \(k=0\), correct at \(k=K\), with the same energies. The oracle uses \(k=0\) on easy and \(k=K\) on hard; its total energy per query is \((1-\varepsilon)a+\varepsilon(a+bK)=a+\varepsilon bK\), and its accuracy is 1. The static policy \(k=K\) uses \(a+bK\) on every query and also has accuracy 1. The ratio follows. \(\square\)

**Why it matters.** The population-level \(E(k)\) can prefer \(k=0\) even when \(k=0\) is catastrophically wrong on a minority of queries (the calculation in §2.2). The accuracy gate removes that pathology, but the static-vs-oracle gap remains. This is the formal justification for query-conditional depth and for reporting the oracle gap in the experiments.

### 4.4 Theorem 3 — certification rate and the cost of calibration

**Setup.** For each \(k\), estimate \(E(k)\) from \(n\) i.i.d. queries by the plug-in ratio \(\hat E(k)=\sum_i J_k(q_i)/\sum_i C_k(q_i)\). Let \(k^\star\) be the best feasible depth and \(\Delta_k=E(k)-E(k^\star)>0\) for \(k\ne k^\star\). Let \(\sigma_k^2\) be the delta-method variance of \(\hat E(k)\):
\[
\sigma_k^2=\frac{\operatorname{Var}(J_k)}{A_k^2}
+\frac{J_k^2\operatorname{Var}(C_k)}{A_k^4}
-\frac{2J_k\operatorname{Cov}(J_k,C_k)}{A_k^3}.
\]

**Theorem (rate).** Any algorithm that identifies \(k^\star\) with probability at least \(1-\delta\) needs
\[
n\;\ge\;\max_{k\ne k^\star}\frac{2\sigma_k^2}{\Delta_k^2}\log\frac{1}{2\delta}
\]
queries in the worst case. Conversely, successive elimination with valid \((1-\delta)\) confidence intervals whose half-widths are below \(\Delta_k/2\) for each surviving \(k\) identifies \(k^\star\) with probability at least \(1-\delta\) using
\[
n=O\!\left(\sum_{k\ne k^\star}\frac{\sigma_k^2}{\Delta_k^2}\log\frac{K}{\delta}\right)
\]
queries.

**Proof sketch.** The lower bound is the two-point Le Cam bound applied to the hardest alternative \(k\ne k^\star\): distinguish two instances whose ratio means differ by \(\Delta_k\), with per-observation variance \(\sigma_k^2\). The upper bound is a union bound over \(K\) confidence intervals plus the standard successive-elimination argument. The ratio variance is the delta-method expression above. \(\square\)

**Corollary (calibration cost).** Let \(C_{\mathrm{cert}}\) be the measured energy of the calibration run, \(k_0\) the default depth, and \(N\) the number of deployment queries. If \(A(k^\star)\ge A(k_0)\), the amortized energy per correct answer with calibration is
\[
E_{\mathrm{cal}}(N)=E(k^\star)+\frac{C_{\mathrm{cert}}}{N A(k^\star)}.
\]
Calibration beats the default iff
\[
N>\frac{C_{\mathrm{cert}}}{A(k^\star)\bigl[E(k_0)-E(k^\star)\bigr]}.
\]
**Proof.** Algebra: \(E_{\mathrm{cal}}(N)<E(k_0)\) iff \(C_{\mathrm{cert}}/(NA(k^\star))<E(k_0)-E(k^\star)\). \(\square\)

**Why this is the distinctive result.** It turns the energy measurement into a decision: for a small deployment, the joules spent finding \(k^\star\) exceed the joules saved by using it. The paper should report \(N^\star\) for each dataset and hardware configuration. A reviewer cannot get this from Pan et al., RAGMark, or any adaptive-\(k\) paper.

### 4.5 Counterexample — distractor-induced non-unimodality

The unimodality theorem depends on monotone coverage. Real retrievers violate it when passages distract the reader (Liu et al., *Lost in the Middle*, TACL 2024, arXiv:2307.03172; Shi et al., *Large Language Models Can Be Easily Distracted by Irrelevant Context*, arXiv:2302.00093).

Take \(a=1\), \(b=0.02\), \(\mathcal{K}=\{0,1,2,4\}\), and
\[
A(0)=0.6,\quad A(1)=0.9,\quad A(2)=0.7,\quad A(4)=1.0.
\]
Then
\[
E(0)=1.667,\quad E(1)=1.133,\quad E(2)=1.486,\quad E(4)=1.080.
\]
\(E\) has a **local minimum at \(k=1\)** and a **lower global minimum at \(k=4\)**. A greedy policy that stops at the first increase stops at 1 and misses 4. The coverage assumption fails at \(k=2\) because the second passage is a distractor; the fourth passage includes the gold evidence and overrides it.

**Implications.**
- The identity in Theorem 1 holds at smooth local minima but cannot distinguish local from global. The certification algorithm must search globally, or the method must first test monotonicity of \(A(k)\) and remove distractors by reranking/compression.
- In the discrete setting, the first-order condition becomes a bracketing inequality; report it as a diagnostic, not a proof of global optimality.
- An adversarial retriever can make \(A(k)\) arbitrary, so no finite-sample guarantee is possible without assumptions. The lower bound is instance-dependent and can be arbitrarily large as \(\Delta_k\to 0\). This is an honest impossibility caveat, not a flaw in the rate result.

### 4.6 What is actually new, and what is standard

**New.**
- The estimand \(E(k)=J(k)/A(k)\) with a correctness gate and a query-conditional \(k^\star(q)\).
- The optimal-depth identity and its use as a split diagnostic.
- The price-of-static-depth bound at equal accuracy.
- The certification-cost threshold \(N^\star\), which makes calibration itself an energy decision.
- The causal energy-tax manipulation as a test of the comparative statics.

**Standard.**
- The bandit/successive-elimination machinery in Theorem 3.
- The delta-method variance for a ratio.
- The idea of adaptive \(k\) itself (Taguchi et al.; Hwang; Hashemi et al.; TAA-\(k\)).
- The joules-per-correct-answer metric (Token Arena).

The paper must not overclaim the standard parts. The contribution is the **decision theory around \(k^\star\)**, not the discovery that retrieval depth affects accuracy.

---

## 5. Dataset and measurement protocol

### 5.1 Data

| Dataset | Use | Size | Retrieval corpus | Answer scoring |
|---|---|---|---|---|
| **SQuAD v1.1 dev** | controlled pilot; in-domain extractive QA | 300 sampled questions (from 10,570) | all unique dev contexts, chunked to 80-word passages; BM25 top-32 | normalized EM; F1 secondary |
| **Natural Questions (NQ-open)** | open-domain generalization | 300 sampled questions | DPR Wikipedia 100-word passages (truncated to 80 words for the pilot), or a 50k-passage public sample; BM25 top-32 | normalized EM; F1 secondary |
| **HotpotQA (distractor or fullwiki)** | optional multi-hop/distractor stress test | 300 questions | provided 10-paragraph distractor set or fullwiki | EM/F1 |

**Answer normalization.** Lowercase, strip articles and punctuation, collapse whitespace. Primary correctness \(C=1\) iff normalized EM \(=1\); sensitivity \(C=1\) iff token F1 \(\ge 0.5\). Pre-register the primary; report both. For generative readers, use a fixed prompt, greedy decoding, and extract the first line after `Answer:`.

**Corpus limitation.** A 50k-passage NQ sample has a lower retrieval ceiling than the full DPR corpus. Report retrieval recall@32 on the sample and the full-corpus recall if available; do not compare absolute accuracy to full-corpus papers. The pilot uses SQuAD's closed corpus to isolate the depth effect; the main experiment uses NQ for external validity.

### 5.2 Reader and retriever

- **Pilot reader:** Qwen2.5-1.5B-Instruct, 4-bit GGUF (Q4_K_M) via `llama-cpp-python` with CUDA offload; max 64 new tokens; context 4096; batch size 1. Fallback: `deepset/roberta-base-squad2` (extractive, 125M) for a high-accuracy, low-variance curve.
- **Main reader:** Qwen2.5-1.5B and Qwen2.5-3B (4-bit), plus one extractive reader, to test generalization of the split.
- **Retriever:** BM25 via `rank_bm25`; lowercased alphanumeric tokenization; index built once; top-32; depth \(k\in\{0,1,2,4,8,16,32\}\). No dense retriever in the pilot; add a cross-encoder reranker only as a secondary condition (it changes the split).
- **Prompt:** fixed system prompt ("Answer the question using only the passages. If the passages do not contain the answer, answer `unknown`."), fixed few-shot examples if any, fixed max tokens. Do not tune the prompt per \(k\).

### 5.3 Energy measurement

**GPU (primary).**
- `pynvml` at 50 ms; record `power.draw`, `utilization.gpu`, `memory.used`, `temperature.gpu`, `clocks.sm`.
- Cross-check with `nvidia-smi --query-gpu=power.draw.average --format=csv,noheader,nounits -lms 100`.
- Integrate power over the query wall time; call `torch.cuda.synchronize()` (or the llama.cpp equivalent) before and after; subtract the idle baseline measured for 60 s before and after the run.
- Report joules per query, joules per correct answer, and the idle-subtracted vs raw values.

**CPU retrieval (secondary).**
- `pyRAPL` (Intel) or `amd_energy` hwmon (AMD) if available; otherwise `turbostat`/`perf` or a wall meter on a 30-query subset.
- If unavailable, report GPU-only energy and a lower bound on retrieval energy; state the boundary explicitly. Do not silently omit CPU energy.

**Whole system (cross-check).**
- A plug-in wall meter (e.g., Kill A Watt) on a 30-query subset per condition; compare to GPU+CPU integration; report the calibration factor and its variance. This is the honest answer to "laptop 3060 power readings are coarse."

**`codecarbon`.** Use if installed as a cross-check, not primary; its CPU estimate is TDP-based and its GPU estimate uses NVML. Report both.

**Protocol controls.**
- Warm-up 20 queries; discard.
- Randomize the order of \(k\) within each question block to avoid thermal/clock drift.
- Fixed power mode if possible; record GPU clocks and temperature; if clocks throttle, report the throttle fraction.
- No concurrent jobs; batch size 1; screen off; AC power.
- Repeat 3 blocks of 20 identical queries; compute the coefficient of variation (CV). The minimum detectable effect is \(2\times\) CV; if the observed effect is smaller, report "not measurable" rather than a spurious \(k^\star\).

### 5.4 Causal manipulation of the split

This is the key test of Corollary 1, and it distinguishes the paper from a correlational measurement study.

- **Fixed energy tax (\(a\)).** Before generation, run a fixed dummy GPU computation (e.g., a calibrated matmul or memory copy) that consumes \(+\Delta_a\) joules but does not change the prompt or answer. Prediction: \(k^\star\) **increases**.
- **Per-passage energy tax (\(b\)).** For each retrieved passage, run a small dummy computation (or duplicate a fixed per-passage preprocessing cost) consuming \(+\Delta_b\) joules per passage, without changing the text. Prediction: \(k^\star\) **decreases**.
- Calibrate \(\Delta_a,\Delta_b\) to 10–30% of the baseline \(a,b\). Run 100–150 questions per condition. Check the sign, not just significance.
- If the signs are wrong, the affine/split model is wrong; the paper must revise the theory or kill the causal claim.

This manipulation is feasible on a 3060 and is a much stronger contribution than another hyperparameter sweep.

### 5.5 Carbon intensity

- **Primary:** Ember *Yearly Electricity Data* (public CSV), country-year average for the study region. Record the exact vintage and factor (gCO\(_2\)e/kWh).
- **Cross-check:** EPA eGRID subregion annual average (US) or EEA (EU), depending on location.
- **Optional time-varying:** UK National Grid Carbon Intensity API (free) for GB; not used for \(k^\star\) because a constant factor does not change the argmin.
- **Reporting:** carbon per correct answer \(=E(k)\times\kappa\); report the retrieval/generation carbon split and the vintage. State clearly that a constant \(\kappa\) is a unit conversion; the carbon-optimal \(k\) differs only under Corollary 3 (different \(\kappa_r,\kappa_g\)).

### 5.6 Statistics and pre-registration

- **Primary endpoint:** \(E(k)\) at \(k\in\{0,1,2,4,8,16,32\}\), gated at \(A_{\min}=A(k_0)\).
- **Effect size:** \(\Delta_{\mathrm{rel}}=[\min(E(0),E(K_{\max}))-E(k^\star)]/E(k^\star)\).
- **Uncertainty:** 1,000 bootstrap resamples over questions; 95% percentile CIs for \(E(k)\), \(A(k)\), \(S(k)\), and \(k^\star\); report the bootstrap probability that each \(k\) is optimal.
- **Model fit:** nonlinear least squares for \(A(k)=1-(1-A_0)(1-p)^k\); report \(R^2\), residuals, and a monotonicity test. If the monotone-coverage model is rejected, report the counterexample regime and use a global search.
- **Identity test:** estimate \(S_{\mathrm{var}}(k)=kJ'(k)/J(k)\) and \(\varepsilon_A(k)=kA'(k)/A(k)\) by finite differences (or fitted affine/coverage models); test equality at \(k^\star\) with a bootstrap CI. A large residual falsifies the affine/coverage model.
- **Multiplicity:** one primary trend test (Jonckheere–Terpstra or a mixed model with question random effects); all other tests exploratory.
- **Pre-registration:** freeze \(A_{\min}\), the \(k\) grid, the reader, the prompt, the correctness rule, and the kill criteria before the pilot. Publish the analysis script.

---

## 6. Pilot: \(\le 2\) hours on the 3060

### 6.1 Design

| Item | Value |
|---|---|
| Questions | 300 SQuAD dev (primary); optional 100 NQ-open if the corpus is ready |
| Depths | \(k\in\{0,1,2,4,8,16,32\}\) (7) |
| Reader | Qwen2.5-1.5B-Instruct Q4_K_M, 64 new tokens, greedy |
| Retriever | BM25, top-32, 80-word passages |
| Runs | 2,100 generations (300 × 7) |
| Repeats | 3 × 20 identical queries for noise/drift |
| Energy | NVML 50 ms; idle baseline; GPU-only primary; CPU retrieval if `pyRAPL` available |
| Time | ~80–110 min including setup and analysis |

### 6.2 Timeline

| Time | Activity |
|---|---|
| 0:00–0:20 | Load model, build BM25 index, smoke test 5 queries |
| 0:20–0:30 | Idle baseline 60 s; 20 warm-up queries; clock/temperature check |
| 0:30–1:15 | Main sweep: 300 questions × 7 depths; randomized \(k\) order |
| 1:15–1:30 | Noise block: 3 × 20 repeated queries; drift check |
| 1:30–1:50 | Compute \(E(k)\), \(A(k)\), \(S(k)\), bootstrap CIs, \(k^\star\), identity residual |
| 1:50–2:00 | Decision memo against the kill criteria |

If the 1.5B reader is slower than expected, fall back to 200 questions × 6 depths (1,200 queries) or the extractive reader; the pilot is a falsification test, not the final experiment.

### 6.3 Kill criteria (pre-registered)

| Code | Criterion | Action |
|---|---|---|
| **K1** | Flat curve: \(\Delta_{\mathrm{rel}}<0.10\) | KILL the depth story; fold measurement into G5 |
| **K2** | Boundary: \(k^\star\in\{0,K_{\max}\}\) on both datasets | Extend \(K_{\max}\) once; if still boundary, KILL |
| **K3** | Uncertifiable: 95% CI of \(E(k^\star)\) overlaps the second-best \(k\), or \(P(k^\star\text{ optimal})<0.8\) | Fix noise/\(n\); if persists, KILL |
| **K4** | Split negligible: \(S(k^\star)<0.05\) and the curve is flat | The lever is generation, not depth; KILL/reframe |
| **K5** | Unstable: \(k^\star\) spans \(>1\) depth across bootstrap or datasets | Pivot to query-conditional policy; do not claim a population \(k^\star\) |
| **K6** | No adaptive gain: oracle gain \(<15\%\) over the best static \(k\) | KILL the adaptive-policy component |
| **K7** | Noise: CV of repeated identical runs \(>20\%\), or minimum detectable effect \(>\) observed effect | Fix measurement (more repeats, wall meter); do not interpret the curve |

**Any of K1–K4 or K6 firing is a kill/reframe.** K5 and K7 are fixable measurement problems, not automatic kills.

### 6.4 Go criteria

| Code | Criterion |
|---|---|
| **G1** | Interior \(k^\star\) with \(\Delta_{\mathrm{rel}}\ge 0.20\) on at least one dataset |
| **G2** | Bootstrap \(P(k^\star\text{ optimal})\ge 0.8\) and CI excludes adjacent depths |
| **G3** | \(S(k^\star)\in[0.10,0.70]\), and \(S(k)\) varies with \(k\) |
| **G4** | Oracle query-conditional gain \(\ge 0.20\); a learned policy captures \(\ge 50\%\) of it |
| **G5** | Replication on a second reader or dataset |

### 6.5 Expected outcomes and interpretation

- **Most likely:** a shallow interior \(k^\star\) around 2–8 on SQuAD, with \(A(k)\) saturating and \(J(k)\) rising; \(\Delta_{\mathrm{rel}}\) 10–30%; \(S(k^\star)\) 10–40% on GPU. If so, the pilot supports the framing but the effect is modest; the main experiment must show a larger effect or a robust adaptive gain.
- **If \(k^\star=0\):** the reader is strong enough that retrieval does not pay for itself at the accuracy gate; this is a legitimate green result ("don't retrieve"), but it kills the depth-optimization story. Reframe as a negative result or move to G1.
- **If \(k^\star=K_{\max}\):** retrieval is still improving accuracy at 32 passages; extend the grid or conclude the lever is context compression, not depth.
- **If the curve is non-monotone:** the distractor counterexample is live; report it and switch to a global search/reranking condition.

---

## 7. Scores

Rubric: Novelty 25%, Method 20%, Math 15%, Experiments 10%, Dataset 10%, Q1 fit 10%, Feasibility 5%, Compute 5%. Core = mean of Novelty, Method, Math, Experiments. Independence is separate (1–5).

| Factor | Weight | Score | Rationale |
|---|---:|---:|---|
| **Novelty** | 25% | **7.0** | Exact estimand + guarantee absent; P1 profiles energy/accuracy; adaptive-\(k\) crowded; the identity, price-of-static bound, and certification-cost threshold are new. The metric itself is not new (Token Arena). As framed as a measurement paper: 6.0. |
| **Method** | 20% | **7.5** | Ratio estimand with correctness gate; explicit energy decomposition; successive elimination; causal energy-tax manipulation; pre-registered protocol. Weakness: one GPU, small reader, single retriever in the pilot. |
| **Math** | 15% | **7.5** | Unimodality, identity, comparative statics, price-of-static bound, minimax rate. Correct and decision-relevant, but the bandit tools are standard; the novelty is the framing and the calibration-cost corollary. |
| **Experiments** | 10% | **6.0** | Pilot-scale; two datasets, one or two readers; limited external validity. Needs HotpotQA and a second hardware/split condition for a Q1 experimental section. |
| **Dataset** | 10% | **7.0** | NQ/SQuAD are public and standard but saturated; add HotpotQA and a distractor stress set. The closed-corpus SQuAD pilot is a strength for isolation, a weakness for realism. |
| **Q1 fit** | 10% | **8.0** | Green AI + IR + efficiency; fits Q1 journals (IP&M, ACM TOIS, IEEE T-SUSC) and top IR/NLP conferences (SIGIR, EMNLP). The energy/carbon angle is timely. |
| **Feasibility** | 5% | **8.5** | 3060, public data, 6–7 months; the main risk is measurement noise, not compute. |
| **Compute** | 5% | **9.0** | CPU/3060; 1.5B–3B Q4; no large models; \(\le \$70\) cash. |

**Core** \(=(7.0+7.5+7.5+6.0)/4=\) **7.0**.  
**Weighted** \(=0.25(7.0)+0.20(7.5)+0.15(7.5)+0.10(6.0)+0.10(7.0)+0.10(8.0)+0.05(8.5)+0.05(9.0)=\) **7.35**.

**Independence: 3/5.** G2 shares the joules-per-correct-answer estimand with G5 and the energy-measurement harness/carbon source with G1. The intervention (\(k\)) and the certification theorem are distinct, but the ideas are not fully independent. If the team runs G1 and G5, reuse the harness (good for feasibility) and explicitly separate the estimands in the write-up.

**Sensitivity.** As framed as "measure \(E(k)\) and pick \(k^\star\)": Novelty 6.0, Method 7.0, Math 6.0, Experiments 6.0, Dataset 7.0, Q1 8.0, Feasibility 8.5, Compute 9.0 → Core **6.25**, Weighted **6.75**. The certification framing is worth roughly +0.75 Core / +0.58 Weighted; without it, the idea is below the 8+ bar.

---

## 8. Verdict and what would change it

### Verdict: CONDITIONAL

**Proceed** to the 2-hour pilot with the certification framing. **Do not** proceed with the naive "measure the curve" framing.

The idea is not a KILL because we found no 2024–2026 paper with the same estimand and guarantee. It is not a GO because the neighborhood is crowded and the current evidence is pre-pilot: the exact metric exists (Token Arena), adaptive \(k\) exists (Taguchi; Hwang; Hashemi; TAA-\(k\)), and RAG energy profiling exists (Pan; RAGMark). The only defensible 8+ contribution is the **decision theory of \(k^\star\)**: the identity, the price of static depth, and the cost of certification.

### What would move it to GO

1. The pilot finds an interior \(k^\star\) with \(\Delta_{\mathrm{rel}}\ge 0.20\) and a bootstrap CI that excludes neighbors (G1–G2).
2. The context-side share at \(k^\star\) is between 10% and 70%, and the energy-tax manipulation moves \(k^\star\) in the predicted direction (G3 + causal test).
3. The oracle query-conditional policy beats the best static \(k\) by \(\ge 20\%\) at equal accuracy, and a simple learned policy captures at least half of that gain (G4).
4. Replication on a second reader or dataset (G5).
5. The calibration-cost threshold \(N^\star\) is small enough to be practically relevant (e.g., \(N^\star<10^4\) queries) or large enough to be an interesting negative result (calibration is not worth it for typical deployments).

### What would KILL it

- A 2026–2027 paper that proves a minimax lower bound for energy-optimal retrieval depth with a matching policy (same estimand + guarantee). Re-run the search before submission.
- The pilot shows a flat curve (\(\Delta_{\mathrm{rel}}<0.10\)), a boundary \(k^\star\), or an unmeasurable split (\(S<5\%\)).
- The adaptive gain over the best static \(k\) is \(<10\%\); the complexity is not justified.
- The energy-tax manipulation has the wrong sign; the affine/split model is falsified.
- Measurement noise (CV \(>20\%\)) cannot be reduced within the 3060 budget.

### Path to 8+

The 8+ version is a **theory + causal measurement** paper:

1. **Theorem 1** (identity + unimodality + comparative statics) as the core.
2. **Theorem 2** (price of static depth) as the necessity argument.
3. **Theorem 3** (certification rate + calibration-cost threshold) as the decision-changing result.
4. **Causal energy-tax experiment** testing the comparative statics.
5. **Pilot/main experiment** on NQ/SQuAD (+HotpotQA), two readers, BM25, NVML + wall-meter cross-check, Ember carbon factor.
6. **Negative results reported honestly**: if \(k^\star\) is at a boundary or the split is negligible, say so; that is still a useful green-RAG result.

If the team cannot commit to Theorems 2–3 and the causal manipulation, the honest verdict is **KILL** or **merge into G5**.

---

## 9. Adversarial review

**Objection 1: "This is just adaptive-\(k\) with a green metric."**  
Response: adaptive-\(k\) papers optimize quality/tokens or latency. G2's contribution is the energy-optimality identity, the static-depth lower bound, and the certification-cost threshold. If the paper only reports the curve, the objection wins; hence the framing.

**Objection 2: "The metric is not new; Token Arena already reports joules per correct answer."**  
Response: correct. The metric is a reporting layer; the new object is \(E(k)\) with a correctness gate and a certified \(k^\star\), plus the decision of whether to calibrate. Cite Token Arena and do not claim the metric.

**Objection 3: "The ratio can be gamed by failing cheaply."**  
Response: the correctness gate \(A_{\min}\) and the equal-accuracy comparison in Theorem 2. Report both the unconstrained and gated curves; the gated one is the decision.

**Objection 4: "The coverage model is unrealistic."**  
Response: it is a stylized model that yields testable predictions; the empirical curve and the monotonicity test are the falsification. The counterexample shows exactly what happens when it fails. The theory does not claim universal coverage.

**Objection 5: "Laptop 3060 energy measurement is too noisy."**  
Response: idle subtraction, 50 ms NVML, repeated blocks, bootstrap CIs, wall-meter cross-check on a subset, and a pre-registered minimum detectable effect. If the effect is below the noise floor, the pilot kills the idea; that is a feature, not a bug.

**Objection 6: "Carbon is just a constant multiplier."**  
Response: true for a single grid factor. Corollary 3 states when carbon changes \(k^\star\) (different carbon intensities for retrieval vs generation). The paper's primary estimand is joules; carbon is reported transparently, not oversold.

**Objection 7: "The closest paper already has the energy-accuracy landscape."**  
Response: P1 maps configurations and predicts accuracy; it does not define \(E(k)\), does not certify \(k^\star\), and does not decide whether calibration is worth its energy. The repair is to make those the contribution, not to compete on the landscape.

**Objection 8: "The bandit lower bound is standard."**  
Response: yes; the paper should say so. The new part is the ratio variance, the correctness-gated feasible set, and the calibration-cost corollary. The theorem is a tool, not the headline.

---

## Appendix A — Search and verification log

**Sources used:** OpenReview API (`api2.openreview.net`), arXiv API (`export.arxiv.org`), Crossref REST API, OpenAlex API, ACL Anthology, SpringerLink, IEEE Xplore metadata, donsetch web search/fetch.

**Queries (representative):** "energy per correct answer retrieval-augmented generation"; "joules per correct answer"; "energy-optimal retrieval depth"; "adaptive retrieval depth energy"; "RAG energy consumption configurations"; "green RAG energy"; "retrieval depth energy"; "energy-aware RAG"; "carbon per correct answer RAG"; title searches for each candidate.

**What was verified.**
- P1: title, authors, abstract, venue, OpenReview ID via the OpenReview API. Full PDF not read (Cloudflare).
- P2: title, author, abstract, submission dates via the OpenReview page. Peer-review status unverified (direct upload).
- P3, P4, near-miss arXiv papers: title, authors, abstract, submission date via arXiv abs pages.
- P5: title, authors, abstract, venue, pages, DOI via SpringerLink and Crossref.
- Taguchi et al.: title, authors, abstract, venue, pages, DOI via ACL Anthology and Crossref.
- GreenFaith-RAG: title, authors, venue, DOI via Crossref and IEEE Xplore; abstract only partially visible (paywall). Label: metadata verified, abstract unverified.
- Optimal Re-Ranking Depth: title, authors, venue, DOI, abstract via Crossref/OpenAlex.
- Energy-Efficient RAG survey: title, authors, venue, DOI, abstract via Crossref.

**What was not verified.**
- Full texts of P1, P2, GreenFaith-RAG, and the survey (paywalls/Cloudflare).
- Exact \(k\) grids used by P1 and P4 (abstracts mention retrieval depths/configurations; full texts not read).
- Any 2027 paper that may appear after this search (the search is current to 8 November 2026).

**Recommendation:** re-run the arXiv/OpenReview/Crossref search immediately before submission, with the exact phrases "energy per correct answer," "joules per correct answer," "energy-optimal retrieval depth," and "certified retrieval depth."

---

## References

1. Pan, M., Arabzadeh, N., Jacob, M., Kazhamiaka, F., Choukse, E., Zaharia, M. *Towards Automatically Optimizing Retrieval Augmented AI Systems.* NeurIPS 2025 MLForSys Workshop. OpenReview `an17DUVvuS`. https://openreview.net/forum?id=an17DUVvuS
2. Hwang, M. *Query Complexity–Aware Adaptive Retrieval for Cost-Efficient Retrieval-Augmented Generation.* OpenReview direct upload, 2026. `ObkC4BCAFt`. https://openreview.net/forum?id=ObkC4BCAFt
3. Hashemi, H., Rühle, V., Rajmohan, S. *Cost-Aware Retrieval-Augmentation Reasoning Models with Adaptive Retrieval Depth.* arXiv:2510.15719, 2025. https://arxiv.org/abs/2510.15719
4. Feric, Z., Taherin, A., Ren, B., Wang, Y., Dy, J., Kaeli, D. *RAGMark: A Comprehensive Framework for Benchmarking Retrieval-Augmented Generation Systems.* arXiv:2609.05760, 2026. https://arxiv.org/abs/2609.05760
5. Song, Z., Fang, J., Li, K., Xia, T., Wang, C. *Tail-Aware Adaptive-k: Query-Adaptive Context Selection for Retrieval-Augmented Generation.* ECML PKDD 2026, LNCS LNAI 16944, pp. 193–209. DOI 10.1007/978-3-032-37667-1_12.
6. Taguchi, C., Maekawa, S., Bhutani, N. *Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k.* EMNLP 2025, pp. 20105–20130. DOI 10.18653/v1/2025.emnlp-main.1017.
7. Ruf, B., Detyniecki, M. *The Cost of Context: Profiling the Energy Footprint of Input Tokens in Large Language Models.* arXiv:2610.19819, 2026. https://arxiv.org/abs/2610.19819
8. Zhao, Z. *When Is Additional Retrieval Worthwhile? A Cost-Aware Study of Evidence Gain and Answer Reliability in RAG.* arXiv:2610.37358, 2026. https://arxiv.org/abs/2610.37358
9. Anand, N., Santra, P., Basuchowdhuri, P., Ganguly, D., Bhatia, S. *One Size Does Not Fit All! Dynamic Retriever and Generator Selection for RAG.* arXiv:2609.17709, 2026. https://arxiv.org/abs/2609.17709
10. Gao, Y., Wang, M., Yu, Y. L. *Token Arena: A Continuous Benchmark Unifying Energy and Cognition in AI Inference.* arXiv:2605.00300, 2026. https://arxiv.org/abs/2605.00300
11. Kashaniyan, M., Jannesari, A. *ATEST: Adaptive Resource-Aware Test-Time Scaling for LLM Inference.* arXiv:2610.25519, 2026. https://arxiv.org/abs/2610.25519
12. Khan, F. S., N J, J., Soni, P. *GreenFaith-RAG: A Confidence-Aware and Energy-Monitored Retrieval-Augmented Generation Framework for Sustainable AI Systems.* IEEE ICATI 2026. DOI 10.1109/icati69399.2026.11604335.
13. Huo, S., Parry, A., Ganguly, D., Clarke, C. L. A. *Optimal Re-Ranking Depth.* SIGIR 2026, pp. 3799–3804. DOI 10.1145/3805712.3809953.
14. Dhakal, A., Pokharel, P., Adhikari, S. *Energy-Efficient Retrieval-Augmented Generation: A Systematic Review of Retrieval, Reranking, Caching, and Context-Management Strategies.* European Journal of Applied Science, Engineering and Technology, 2026. DOI 10.59324/ejaset.2026.4(5).11.
15. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. arXiv:2307.03172.
16. Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E., Schärli, N., Zhou, D. *Large Language Models Can Be Easily Distracted by Irrelevant Context.* ICML 2023. arXiv:2302.00093.
