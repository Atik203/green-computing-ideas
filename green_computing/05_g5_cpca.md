# G5 — Carbon per Correct Answer (CPCA): adversarial review and repair

**Date:** 8 November 2026  
**Track:** Green computing, separate from `research_audit/round9` and `round10`  
**Verdict:** **KILL** the CPCA leaderboard as stated (weighted 4.63, core 4.00). **CONDITIONAL** on repair R1, a budget-indexed carbon-effectiveness decision layer (weighted 7.03, core 7.00), subject to a two-week gate.

> **One line.** The estimand is already owned. Token Arena defines \(J_{\mathrm{CA}}(e)=j_eT_e/A_e\), exactly joules per correct answer; Yang & Armour measure joules per correct answer across 86 checkpoints and reasoning modes; Mamun et al. compare CoT and multi-sampling energy; Jin et al. compare direct, majority-vote, and reasoning-token test-time compute on accuracy–energy frontiers; Kashaniyan & Jannesari compare candidate-generation schedules with measured GPU energy. A fifth CPCA table is incremental. The nearest defensible repair is not another table: it is the carbon-effectiveness frontier with a break-even carbon price, paired failure-inclusive estimation, and an explicit statement that a scalar CPCA ranking is not decision-sufficient.

---

## 1. Plain-language problem

A team can answer a question directly, with chain-of-thought (CoT), with self-consistency (SC), or with debate. Each choice uses a different amount of energy and gets a different fraction of answers right. The tempting summary is **joules per correct answer**: total energy divided by the number of correct answers. That number is easy to report and easy to rank.

The problem is that the ranking is not the deployment decision. A cheap, low-accuracy strategy can have the best joules per correct answer and still be the wrong choice when the service must hit an accuracy floor. A verbose, high-accuracy strategy can have a worse joules per correct answer and still be the right choice when the carbon budget is generous or when an extra correct answer is worth more than the extra energy. “Which strategy is greenest per correct answer?” therefore has no budget-free answer. The benchmark must report the **frontier** and the **price at which the choice changes**, not a single league table.

This is not a philosophical objection. It changes what a green-AI benchmark should measure, how many queries it needs, and what a practitioner should do with the result.

---

## 2. Exact estimand, functional, and decision

### 2.1 Setup

Let \(q\sim Q\) be a benchmark query. A **strategy** \(s\) is a stochastic policy that maps \(q\) to a sequence of model calls, retries, and optional tool calls, and then to a final answer. For each run, observe

\[
E_{s,q}\ge 0 \quad\text{(total joules consumed by the run)},\qquad
Y_{s,q}\in\{0,1\}\quad\text{(final answer correct)}.
\]

Both are random: decoding is stochastic, and the policy may stop, retry, or call a tool depending on intermediate output. The pair \((E_{s,q},Y_{s,q})\) is the primitive observation. Failed attempts are part of \(E_{s,q}\); they are not discarded.

Define

\[
\mu_s=\mathbb E_Q[E_{s,q}],\qquad A_s=\mathbb E_Q[Y_{s,q}],\qquad
\boxed{\mathrm{CPCA}_s=R_s=\frac{\mu_s}{A_s}}\quad\text{(joules per correct answer)}.
\]

With a fixed grid carbon intensity \(\mathrm{CI}\) (gCO\(_2\)e per kWh),

\[
R_s^{\mathrm{CO2}}=R_s\cdot \frac{\mathrm{CI}}{3.6\times 10^6}\quad\text{(gCO}_2\text{e per correct answer)}.
\]

Because \(\mathrm{CI}\) is a constant within a region and hardware, \(R^{\mathrm{CO2}}\) is a monotone rescaling of \(R\). **Carbon adds no ranking information in a single-region, single-hardware study.** The scientific content is the energy and accuracy measurement; carbon is a reporting unit and a policy scale. Any claim of “carbon-aware” novelty must come from spatial/temporal carbon-intensity variation or a local-vs-remote comparison, neither of which is in the original G5 scope. This is the first reason the original framing overclaims.

### 2.2 Channel decomposition

Energy is additive by measurement boundary:

\[
E_{s,q}=E^{\mathrm{model}}_{s,q}+E^{\mathrm{tok}}_{s,q}+E^{\mathrm{retry}}_{s,q}
+E^{\mathrm{tool}}_{s,q}+E^{\mathrm{idle}}_{s,q}.
\]

A practical, identifiable version uses per-call measurements and a reference model \(m_0\):

\[
E_{s,q}=\sum_{c\in\mathcal C_{s,q}}\Big[a_{m_c}+b_{m_c}p_c+d_{m_c}o_c+e_{\mathrm{tool}}(c)\Big]
+E^{\mathrm{retry}}_{s,q}+E^{\mathrm{idle}}_{s,q}+\varepsilon_{s,q},
\]

where \(p_c\) and \(o_c\) are prefill and decode token counts, \(a_m\) is the per-call launch/state cost, and \(b_m,d_m\) are joules per prefill/decode token. Define

\[
E^{\mathrm{tok}}=\sum_c\big(b_{m_0}p_c+d_{m_0}o_c\big),\qquad
E^{\mathrm{model}}=\sum_c\Big[a_{m_c}+(b_{m_c}-b_{m_0})p_c+(d_{m_c}-d_{m_0})o_c\Big].
\]

The split is a **reference-model accounting convention**, not a physical separability claim: model choice and token count are collinear in a GPU kernel. Report the reference model and the fitted coefficients. The retry channel is the energy of attempts after the first that fail the policy’s verifier; the tool channel is measured tool energy (for a local calculator or retriever, CPU/GPU energy; for a remote API, exclude and report a lower bound). Idle energy is the measured baseline between queries. The residual \(\varepsilon\) is measurement error and is reported, not hidden.

**Why this decomposition matters.** A strategy can be green for two very different reasons: fewer tokens, or a cheaper model at the same tokens, or fewer retries, or cheaper tools. A single CPCA number cannot tell a practitioner which lever to pull. The decomposition also exposes a common error: comparing strategies at different token budgets and attributing the whole difference to “reasoning.”

### 2.3 Estimator

The natural estimator is the **paired plug-in ratio**

\[
\hat R_s=\frac{\bar E_s}{\bar Y_s}
=\frac{\frac1n\sum_{i=1}^n E_{s,q_i}}{\frac1n\sum_{i=1}^n Y_{s,q_i}},
\]

with the same queries \(q_1,\dots,q_n\) used for every strategy. Pairing is essential: query difficulty drives both energy and correctness, and unpaired sampling inflates the variance of every comparison.

The influence function is

\[
\psi_{s,i}=\frac{E_{s,i}-R_sY_{s,i}}{A_s},
\qquad
\sqrt n(\hat R_s-R_s)\;\Rightarrow\;\mathcal N\!\left(0,\sigma_s^2\right),
\quad
\sigma_s^2=\operatorname{Var}(\psi_{s}).
\]

This follows from \(\sqrt n(\hat R_s-R_s)=\sqrt n(\bar E_s-R_s\bar Y_s)/\bar Y_s+o_p(1)\) and \(\bar Y_s\to A_s\). The estimator is consistent but biased at \(O(1/n)\); the bias is negligible relative to the confidence interval at the sample sizes planned here. The plug-in variance is

\[
\hat\sigma_s^2=\frac1n\sum_{i=1}^n\big(\hat\psi_{s,i}-\bar{\hat\psi}_s\big)^2,
\qquad
\hat\psi_{s,i}=\frac{E_{s,i}-\hat R_sY_{s,i}}{\bar Y_s}.
\]

For a paired contrast \(\Delta=R_s-R_t\), use

\[
\hat\Delta=\hat R_s-\hat R_t,\qquad
\widehat{\operatorname{Var}}(\hat\Delta)
=\frac1n\widehat{\operatorname{Var}}\big(\hat\psi_{s,i}-\hat\psi_{t,i}\big).
\]

This automatically accounts for the covariance between energy and correctness and for the paired dependence. Do **not** add channel variances: the channels are correlated, and the total variance must be computed from the influence function of total energy.

For a single ratio, a Fieller interval is more reliable than the delta method when \(A_s\) is near 0:

\[
\left\{R:\;(\bar E_s-R\bar Y_s)^2
\le z_{1-\alpha/2}^2\,\frac{\widehat{\operatorname{Var}}(E_s-RY_s)}{n}\right\}.
\]

Use a paired cluster bootstrap (10,000 replicates, resampling queries) as the primary interval and Fieller/delta as checks. If the denominator confidence interval includes 0, report the ratio as unbounded rather than printing a finite number.

### 2.4 Decision rules

The original G5 asks “which strategy is greenest per correct answer?” The deployable questions are these three.

**D1 — Accuracy-constrained minimum carbon.**

\[
s^\star(A_{\min})=\arg\min_{s:\,A_s\ge A_{\min}} R_s.
\]

This is the “I must hit at least \(A_{\min}\) accuracy; use the least energy per correct answer” rule.

**D2 — Carbon-budgeted maximum accuracy.**

\[
s^\star(B)=\arg\max_{s:\,\mu_s\le B} A_s,
\]

where \(B\) is a per-query average carbon budget (J, or gCO\(_2\)e after conversion). This is the “I have a carbon budget; buy as many correct answers as possible” rule.

**D3 — Carbon-price net benefit.**

\[
s^\star(\lambda)=\arg\max_{s}\;\mathrm{NCB}_s(\lambda),
\qquad
\mathrm{NCB}_s(\lambda)=\lambda A_s-\mu_s,
\]

where \(\lambda\ge 0\) is the **carbon shadow price**: joules (or gCO\(_2\)e) the deployer is willing to pay for one additional correct answer. This is the Lagrangian dual of D2. It is the most useful rule for reporting because the break-even price is interpretable.

**Certification rule (pairwise).** Strategy \(s\) is certifiably greener than \(t\) at margin \(\delta_R>0\) and level \(\alpha\) if

\[
\mathrm{UCB}_{1-\alpha}(R_s-R_t)<-\delta_R
\quad\text{and}\quad
\mathrm{LCB}_{1-\alpha}(A_s-A_t)\ge -\delta_A,
\]

using Holm correction across the \(K-1\) comparisons. For D3, replace the ratio test with the paired net-benefit difference

\[
D_i(\lambda)=\lambda(Y_{s,i}-Y_{t,i})-(E_{s,i}-E_{t,i}),
\]

and certify if \(\mathrm{LCB}_{1-\alpha}(\bar D(\lambda))>\delta_{\mathrm{NCB}}\). The net-benefit parameterization is regular even when \(A_s=A_t\), whereas the incremental ratio is undefined there; this is why D3 is the preferred primary rule.

**Break-even carbon price.** For two adjacent frontier strategies with \(A_s>A_t\) and \(\mu_s>\mu_t\),

\[
\lambda^\star_{s,t}=\frac{\mu_s-\mu_t}{A_s-A_t}
\]

is the price above which the more accurate/more expensive strategy wins. Report \(\lambda^\star\) with a Fieller or bootstrap confidence interval. If it includes 0 or is unbounded, report “not identified” and fall back to the net-benefit decision.

**Carbon-effectiveness acceptability curve (CEAC).** For a grid of \(\lambda\), bootstrap the paired data and compute the argmax of \(\mathrm{NCB}_s(\lambda)\) in each replicate. The fraction of replicates in which \(s\) is optimal is \(\mathrm{CEAC}_s(\lambda)\). This is the uncertainty-aware version of the league table: it reports the probability that each strategy is the right deployment at each carbon price.

### 2.5 Sample size and minimum detectable effect

For a paired contrast with per-query standard deviation \(\sigma_\Delta\), the sample size for power \(1-\beta\) and two-sided level \(\alpha\) is

\[
n\ge \frac{\sigma_\Delta^2\big(z_{1-\alpha/2}+z_{1-\beta}\big)^2}{\delta^2}.
\]

Equivalently, the minimum detectable effect at \(n\) is

\[
\delta_{\min}=\big(z_{1-\alpha/2}+z_{1-\beta}\big)\frac{\sigma_\Delta}{\sqrt n}.
\]

For a relative effect on CPCA, use \(\delta=\epsilon R\) and \(\sigma_\Delta\) from the paired influence-function differences. The pilot estimates \(\sigma_\Delta\); the full study sizes \(n\) from it. A lower bound of the same order follows from Le Cam’s two-point method: no test can distinguish two net-benefit distributions separated by \(\delta\) with error \(\alpha\) using \(o(\sigma_\Delta^2/\delta^2)\) paired samples. The practical implication is that a benchmark with \(n=100\) and a 25% effect is usually underpowered for ratio comparisons; report the MDE rather than a bare ranking.

---

## 3. The failure-inclusion bias (a measurement proposition)

Many energy studies report energy per **successful** run, \(E^+=E[E\mid Y=1]\), because failed runs are inconvenient. This is not CPCA. By the law of total expectation,

\[
\mu=A\,E^+ +(1-A)E^-,
\qquad
R=\frac{\mu}{A}=E^+ +\frac{1-A}{A}E^-,
\]

where \(E^-=E[E\mid Y=0]\). The success-conditioned number underestimates CPCA by

\[
\Delta_{\mathrm{bias}}(s)=\frac{1-A_s}{A_s}E^-_s.
\]

The bias is strategy-dependent: it is large when accuracy is low and failed runs are expensive, and small when failures are cheap. A low-accuracy strategy with cheap failures can look better than it is; a high-accuracy strategy with expensive failures can look worse. A ranking reversal \(R_s<R_t\) but \(E^+_s>E^+_t\) is possible whenever

\[
0<E^+_s-E^+_t<\Delta_{\mathrm{bias}}(t)-\Delta_{\mathrm{bias}}(s).
\]

This is not a hypothetical. Reasoning strategies differ in accuracy by 10–40 points on small models, and failed CoT/SC runs can be long and expensive. Any benchmark that drops failed attempts can reverse the ranking it claims to measure. **Protocol rule: every attempt, including failed retries and parse failures, is metered and included in the numerator; accuracy is the final correctness rate.** This is cheap to implement and is a genuine methodological contribution relative to per-success reporting.

Token Arena and Yang & Armour already divide total energy by accuracy, so they are not guilty of this particular bias; the proposition is aimed at the broader literature and at any new benchmark that measures only successful generations. It is a repair requirement, not a claim that the closest papers are wrong.

---

## 4. Closest five papers (2024–2026), functional level

Search run 8 November 2026: arXiv abstract/full text, exact-title checks, no paid API. “Verified” means the paper’s content was read; venue status is stated separately. The ledger `ledger_g5.csv` carries the full metadata.

| # | Paper | ID / year | What it already owns | What it lacks relative to G5 | Status |
|---|---|---|---|---|---|
| 1 | **Token Arena: A Continuous Benchmark Unifying Energy and Cognition in AI Inference** — Gao, Wang, Yu | arXiv:2605.00300, 2026 | The exact estimand: \(J_{\mathrm{CA}}(e)=j_eT_e/A_e\), joules per correct answer, plus dollars per correct answer and endpoint fidelity; 78 endpoints, 12 model families, 33 providers; 6.2× spread in modeled joules per correct answer | Energy is **modeled** from TDP × throughput, not measured; unit is the endpoint, not the reasoning strategy; no CoT/SC/debate axis; no retries/tool decomposition; no confidence interval, no budget decision, no CE frontier | Verified (full text); venue unverified |
| 2 | **Measuring the Inference Energy Efficiency of Open-Weight LLMs across Benchmarks, Reasoning Modes, and Hardware** — Yang, Armour | arXiv:2610.14389, 2026 | Measured GPU energy per correct answer across 86 checkpoints, 134 configs, 7 benchmarks, 6 hardware; 33× spread among >80% AIME models; thinking on/off costs 3.7× energy; token count misorders 18–24% of model pairs | Reasoning mode is only thinking on/off, not CoT vs SC vs debate; no retry/tool decomposition; no CPCA confidence interval or budget rule; deliberately avoids a score-per-energy ratio | Verified (full text); venue unverified |
| 3 | **Evaluating the Environmental Impact of using SLMs and Prompt Engineering for Code Generation** — Mamun, Nath, Uddin, Deb | arXiv:2604.02776, 2026 | Six prompting strategies (including CoT and multi-sampling) across 11 SLMs (1B–34B), HumanEval+/MBPP+, Pass@1 with energy (kWh), carbon (kgCO₂eq), latency; finds CoT near-optimal and multi-sampling disproportionate | Code-generation domain only; no CPCA estimator, confidence interval, or decision rule; no budget/CE frontier; a single point per strategy, not a frontier | Verified (abstract and details); venue unverified |
| 4 | **Sample Count Is Not Enough: Candidate-Generation Strategy Shapes the Energy and Performance of LLM Test-Time Scaling** — Kashaniyan, Jannesari | arXiv:2609.19499, 2026 | Compares four candidate-generation schedules at fixed \(N=8\) with measured GPU energy; Phi-3-mini and Qwen2.5-1.5B on GSM8K/SciQ, A100/V100; serial calls use 4.64–4.86× the energy of one batched call | The axis is scheduling, not CoT/SC/debate; no accuracy denominator, no CPCA, no confidence interval, no budget decision | Verified (abstract + venue); SC26 EESP workshop |
| 5 | **The Energy Cost of Reasoning: Analyzing Energy Usage in LLMs with Test-time Compute** — Jin, Wei, Brooks | arXiv:2505.14733, 2025 | Direct vs majority-vote (parallel sampling) vs reasoning-token TTC on Qwen2.5 1.5B–32B, A100, 10 runs; accuracy–energy Pareto frontiers; RT can use 113.48× the energy; tokens for correct vs incorrect queries | No CPCA estimator/confidence interval; no CoT vs SC vs debate taxonomy; no budget/CE decision; no failure-inclusion analysis; one model family | Verified (full text); venue unverified |

### 4.1 KILL statement

**Same estimand.** Token Arena Eq. (1) is CPCA with a modeled \(j_e\). Yang & Armour report measured GPU energy per correct answer, including reasoning modes, and explicitly discuss the 33× spread. There is no new estimand to define.

**Same empirical question.** Mamun et al. compare CoT and multi-sampling energy and accuracy; Jin et al. compare direct, majority-vote, and reasoning-token strategies; Kashaniyan & Jannesari compare generation schedules with measured energy. The question “which reasoning strategy is greenest?” is already answered at several levels, on public benchmarks, with measured energy.

**Same benchmark shape.** Token Arena is a continuous leaderboard; Yang & Armour is a 134-configuration benchmark; ML.ENERGY is a measurement tool and leaderboard. A new CPCA leaderboard would be a fifth data point, not a new functional.

**Decision.** **KILL the original G5.** The same estimand is already published, which puts the ceiling at 6; the actual weighted score is 4.6 because the carbon framing is a constant rescaling and the empirical comparison is crowded. Do not spend six months producing another table.

### 4.2 Adjacent work (not among the closest five, but relevant)

- **CEBench** (arXiv:2407.12797, 2024): cost-effectiveness toolkit for LLM pipelines. Cost is dollars, not energy; no carbon, no reasoning strategies, no uncertainty. It establishes that “cost-effectiveness” is an acceptable framing, but not the CPCA frontier.
- **The Efficiency Frontier** (arXiv:2605.23071, LMIAT 2026): cost–performance frontier for context management, token cost, no energy. Useful precedent for a frontier paper, but not a competitor on the estimand.
- **ML.ENERGY** (arXiv:2505.06371, NeurIPS D&B 2025 Spotlight): measurement infrastructure and leaderboard; no accuracy denominator, no strategy comparison. Use it as the measurement baseline, not as a competitor.
- **TokWatt** (arXiv:2610.39846, 2026): 134 published inference-energy estimates harmonized across five boundary components; useful for carbon conversion and boundary reporting, but no accuracy/CPCA.
- **AI Energy Score** (Hugging Face / Salesforce consortium): standardized energy per query at batch size 1; no accuracy denominator; not a reasoning-strategy benchmark. Use its protocol as a sanity check for the measurement boundary.

---

## 5. Repair R1: CPCA-CEA (carbon-effectiveness analysis)

### 5.1 What changes

Do not ask “which strategy has the lowest CPCA?” Ask:

1. What is the **carbon-effectiveness frontier** of (energy per query, accuracy) for the strategy set?
2. For a deployment carbon budget \(B\), which strategy maximizes accuracy?
3. For a carbon shadow price \(\lambda\), which strategy maximizes net carbon benefit?
4. What is the **break-even carbon price** between adjacent frontier strategies, with uncertainty?
5. Which strategies are **dominated** and should never be deployed, even if their CPCA looks respectable?

The output is a frontier, a staircase, a CEAC, and a break-even table — not a single ranking.

### 5.2 The non-identity result: CPCA rank is not carbon-effectiveness rank

Let \(\mathcal P=\{(\mu_s,A_s)\}_{s\in\mathcal S}\). Let \(\mathcal F\) be the upper concave envelope of \(\mathcal P\) (the upper convex hull if fleet-level mixtures/routing are allowed). A point below \(\mathcal F\) is dominated: there is a mixture (or a single point, if one exists) with no more energy and no less accuracy.

**Proposition 1 (frontier sufficiency).** For D1, D2, and D3 with average budgets and feasible mixtures, every optimal strategy lies on \(\mathcal F\); any point below \(\mathcal F\) is never strictly optimal.

*Proof sketch.* If \(p=(\mu,A)\) is below \(\mathcal F\), there is a convex combination \(p'=(\mu',A')\) with \(\mu'\le\mu\) and \(A'\ge A\), strict in at least one coordinate. For D1, \(R'=\mu'/A'\le\mu/A\). For D2, \(p'\) is feasible whenever \(p\) is and has \(A'\ge A\). For D3, \(\lambda A'-\mu'\ge\lambda A-\mu\). Hence \(p\) is not strictly optimal. \(\square\)

**Proposition 2 (the CPCA minimum is on the frontier, but the ranking is not).** Let \(R_{\min}=\min_s\mu_s/A_s\) and \(s_0\in\arg\min R_s\). At \(\lambda=R_{\min}\), \(\mathrm{NCB}_{s_0}=R_{\min}A_{s_0}-\mu_{s_0}=0\), while for every \(s\), \(\mathrm{NCB}_s=R_{\min}A_s-\mu_s\le0\) because \(A_s\le R_{\min}\mu_s\). Hence \(s_0\) is on \(\mathcal F\) and is optimal at \(\lambda=R_{\min}\). But the CPCA order of the **other** strategies need not be the frontier order: a lower-CPCA strategy can be below \(\mathcal F\) (dominated), while a higher-CPCA strategy can be a frontier vertex and optimal at a higher \(\lambda\).

*Constructive example* (illustrative, plausible for a small-model suite):

| Strategy | \(\mu\) (J/query) | \(A\) | \(R=\mu/A\) (J/correct) | Frontier status |
|---|---:|---:|---:|---|
| Direct | 100 | 0.30 | 333 | frontier (low \(\lambda\)) |
| CoT-short (cap 128) | 200 | 0.40 | 500 | **dominated** |
| CoT-long (cap 512) | 400 | 0.70 | 571 | frontier (mid \(\lambda\)) |
| SC \(k=3\) | 900 | 0.75 | 1200 | frontier (high \(\lambda\)) |

CPCA ranks Direct < CoT-short < CoT-long < SC3. But the chord from Direct to CoT-long at \(\mu=200\) gives \(A=0.30+\frac{100}{300}(0.40)=0.433>0.40\). A fleet that routes one third of queries to CoT-long and two thirds to Direct gets 43.3% accuracy at 200 J/query, dominating CoT-short. Under fleet-level routing, CoT-short is never optimal for any \(\lambda\ge0\). The **second-greenest** strategy in the CPCA table is undeployable at every carbon price, while the third-greenest is optimal above \(\lambda=750\) J/correct. A CPCA league table would recommend the wrong strategy.

The dominance claim uses fleet-level routing, which is the natural setting for an average carbon budget. Without routing, the weaker statement still holds: CoT-short is budget-optimal only on the narrow interval \(B\in[200,400)\), Direct is optimal below 200, and CoT-long is optimal above 400; the CPCA ranking provides no budget index and therefore cannot tell the deployer which interval applies. The strong version (a dominated CPCA runner-up) is a fleet-level result; the weak version (budget-dependent ranking) needs no mixture assumption. Both are decision-changing; report which one the deployment supports.

The break-even prices are:

\[
\lambda^\star_{\text{Direct}\to\text{CoT-long}}
=\frac{400-100}{0.70-0.30}=750\ \text{J/correct},
\qquad
\lambda^\star_{\text{CoT-long}\to\text{SC3}}
=\frac{900-400}{0.75-0.70}=10{,}000\ \text{J/correct}.
\]

At \(\lambda<750\), Direct wins; at \(750<\lambda<10{,}000\), CoT-long wins; above 10,000, SC3 wins. The CPCA table says Direct is always greenest, which is true only for \(\lambda<750\). The decision-relevant output is the **interval of \(\lambda\)** in which each strategy is optimal, not its rank.

**Proposition 3 (failure-inclusion bias).** As in §3: \(R_s=E^+_s+\frac{1-A_s}{A_s}E^-_s\), so success-conditioned accounting is biased downward by a strategy-dependent amount; ranking reversals are possible. Proof: law of total expectation. \(\square\)

**Proposition 4 (rate).** Certifying a net-benefit difference \(\delta\) at level \(\alpha\) with power \(1-\beta\) requires \(n=\Theta(\sigma_\Delta^2(z_{1-\alpha/2}+z_{1-\beta})^2/\delta^2)\) paired queries; no test can do better than \(\Omega(\sigma_\Delta^2/\delta^2)\) by Le Cam’s two-point bound. Proof sketch: standard Gaussian shift; the paired net-benefit difference is a bounded-variance mean. \(\square\)

### 5.3 What the repaired benchmark reports

For every strategy \(s\) and benchmark \(b\):

1. \(\hat\mu_s,\hat A_s,\hat R_s\) with paired bootstrap CIs; \(n\), MDE, and power.
2. Channel decomposition: model, prefill/decode tokens, retries, tools, idle; covariance matrix and reference model.
3. Frontier membership: probability that \(s\) is on the upper envelope/hull under the bootstrap; probability that \(s\) is dominated.
4. Budget staircase \(A^\star(B)\) for \(B\) on a grid, with confidence bands.
5. CEAC\(_s(\lambda)\) and break-even \(\lambda^\star_{s,t}\) with CIs.
6. The benchmark’s own energy and carbon cost (the evaluation is not free).

This is a reusable artifact: a protocol, a per-query schema, an analysis script, and a leaderboard that reports frontier probabilities rather than a single rank. It is a measurement/decision benchmark, not another accuracy table.

### 5.4 Novelty claim (honest)

The cost-effectiveness frontier, ICER, net-benefit regression, and CEAC are mature in health economics. The novelty is the **domain transfer plus the CPCA-specific results**: (i) the exact demonstration that CPCA rank is not budget/frontier rank for reasoning strategies; (ii) the failure-inclusion bias proposition for reasoning traces; (iii) a paired, failure-inclusive, channel-decomposed measurement protocol; (iv) the break-even carbon price as the headline output. This is a 6.5–7 novelty, not an 8: the mathematics is imported, and the empirical question is crowded. The paper’s contribution is that it stops the field from reporting a misleading scalar.

---

## 6. Measurement plan

### 6.1 Benchmarks and models

| Component | Choice | Why |
|---|---|---|
| Math | GSM8K test, fixed random sample of 300 items (seed recorded) | Multi-step reasoning; small models span a useful accuracy range; exact-match grading |
| Science MCQ | ARC-Challenge test, stratified 300 | Non-math reasoning; 4-way choice; exact-match letter |
| Knowledge | MMLU test, stratified 500 across subjects | Broad coverage; checks whether reasoning gains generalize beyond math |
| Models | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct (fp16); Phi-3-mini-3.8B or Qwen2.5-3B in 4-bit on a 24 GB rental | Fit the 6 GB 3060 at 1.5B; 3B only with quantization/rental; two sizes test scaling |
| Strategies | Direct; CoT-short (cap 128); CoT-long (cap 512); SC \(k=3,5\); debate 2 agents × 2 rounds; CoT + local calculator; CoT + verify-retry | Spans tokens, samples, retries, tools; the pilot uses the first four |
| Serving regimes | Batch 1 interactive (primary); batch 8 saturated (secondary) | CPCA depends on batching (Yang & Armour; ML.ENERGY); report both |
| Seeds | Full study: 3 seeds per item for stochastic strategies; greedy for direct/CoT | Separates sampling noise from item difficulty; the pilot uses 1 run plus a 20-item replication |

All benchmarks are public and downloadable; no paid API. Use exact-match scoring with a fixed answer extractor; log parse failures as incorrect and report the parse rate. Do not use an LLM judge (energy and variance cost, and judge bias).

### 6.2 Energy instrumentation

Primary:

```bash
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu \
  --format=csv,noheader,nounits -lms 100
```

- Sample at 100 ms; integrate power over the query interval with the trapezoid rule; synchronize CUDA before the start and after the end of each query.
- Measure a 60 s idle baseline before and after each arm; subtract it for net energy; report gross and net.
- Log per-call boundaries (prefill start/end, decode start/end) to attribute tokens vs model time.
- Run one query at a time for the per-query measurement (no concurrent requests); the batched regime is a separate condition with its own per-request allocation by token share.
- Warm up for 5 minutes; monitor clocks/temperature; randomize arm order; pause on thermal throttling.

Cross-check: `codecarbon` if installed. It measures a different boundary (CPU+RAM+GPU), so do not mix its numbers with NVML; report both and explain the difference. If NVML `power.draw` is unavailable or quantized too coarsely on the laptop 3060, use a wall meter or a desktop GPU; kill the measurement if neither is available (see §7).

### 6.3 Carbon intensity

Use a **fixed annual regional average** from a public dataset, not a real-time API:

- Ember annual electricity data (public CSV) for the study country/region; record year, region, and value.
- Sensitivity: low 100, mid 400, high 700 gCO\(_2\)e/kWh. Since carbon intensity is a scalar, these change the scale, not the within-region ranking; state that explicitly.
- PUE = 1.0 for local hardware; if using a rental cloud, use the provider’s published PUE or 1.2 as a sensitivity.
- Report J/correct as the primary metric and gCO\(_2\)e/correct as the policy conversion. Do not claim carbon-aware scheduling from a constant carbon intensity.

### 6.4 Measurement boundary and validity

**Boundary:** GPU board power (NVML) for discrete GPUs; GPU+HBM for superchips; exclude host CPU, RAM, display, power-supply losses, cooling, and network. This is the standard accelerator boundary used by ML.ENERGY and Yang & Armour. Report it explicitly; do not call it “total system energy.”

**Threats and mitigations:**

| Threat | Mitigation |
|---|---|
| Coarse laptop power readings | 100 ms sampling, 60 s idle baseline, repeat calibration; compare NVML to codecarbon; kill if CV > 0.35 |
| Thermal throttling | 5 min warm-up; randomized arm order; monitor clocks/temp; pause between arms |
| Background GPU use | Close other apps; record `nvidia-smi` processes; run one model at a time |
| Batching confound | Separate batch-1 and batch-8 conditions; never compare across regimes |
| Tokenization differences | Log token counts from the tokenizer; report tokens/query and tokens/correct |
| Answer parsing | Fixed extractor; log parse failures; report parse rate; count as incorrect |
| Contamination | Use test splits; report exact split and seeds; do not tune prompts on test |
| Stochastic decoding | 3 seeds; paired by item; report seed variance |
| Hardware specificity | Two GPU types if possible (3060 + rental 24 GB); report absolute J and relative rankings |
| Carbon constant | Report J first; carbon-intensity sensitivity; no carbon-ranking claim |

### 6.5 Power analysis

From the pilot, estimate \(\sigma_\Delta\) for the primary contrast (Direct vs CoT-long) and for the frontier break-even. Target: 80% power to detect a 10% relative difference in CPCA, or a break-even \(\lambda\) with a confidence-interval width < 30% of its point estimate. If the pilot's \(\sigma_\Delta\) implies \(n>1000\) per arm, the 3060-only design is underpowered; either use a paired design with more items, reduce the number of arms, or move to a 24 GB rental for the full study.

---

## 7. Pilot: ≤2 hours on the RTX 3060 6 GB

### 7.1 Goal

The pilot is not a mini-leaderboard. It answers four questions:

1. **Can the 3060 measure energy well enough** to resolve a 15–20% CPCA difference?
2. **Does failure-inclusive CPCA differ from success-conditioned \(E^+\)** in ranking?
3. **Is there a real frontier configuration** in which the CPCA runner-up is dominated and a higher-CPCA strategy is optimal at a plausible \(\lambda\)?
4. **What is \(\sigma_\Delta\)** for the full-study power calculation?

### 7.2 Design

- **Model:** Qwen2.5-1.5B-Instruct, fp16, batch 1, HF Transformers (no vLLM on Windows).
- **Benchmark:** GSM8K 50 items + ARC-Challenge 30 items = 80 queries, fixed random sample (seed recorded); ARC stratified by answer letter. If the measured throughput is above ~25 output tokens/s, extend to 120 queries (80 GSM8K + 40 ARC) using the spare time.
- **Arms:** Direct; CoT-short (max 128 new tokens); CoT-long (max 512); SC \(k=3\) with CoT-long samples and majority vote. This gives four points and enough geometry to test the frontier; it also spans the token channel.
- **Seeds:** 1 run per item for all arms; 20 items repeated with a second seed on Direct and CoT-long only, to estimate within-item variance without blowing the budget.
- **Schedule (target 115 min):**

| Time | Activity |
|---|---|
| 0:00–0:10 | Load model, warm up 5 min, measure idle baseline, verify `nvidia-smi` sampling |
| 0:10–0:25 | 10-query smoke test; check answer extraction, token logging, per-query energy integration |
| 0:25–1:15 | Main run: 80 queries × 4 arms, randomized arm order, per-query JSONL |
| 1:15–1:30 | Second-seed replication: 20 items × 2 arms (Direct, CoT-long) |
| 1:30–1:50 | Analysis: \(\hat\mu,\hat A,\hat R\), paired bootstrap confidence intervals, frontier, CEAC, \(E^+\) comparison, MDE |
| 1:50–2:00 | Score against kill criteria; write one-page memo |

If the 1.5B model averages more than ~35 s/query across the four arms, drop to 60 queries (40 GSM8K + 20 ARC); preserve the four arms, because the geometry is the point. If SC3 is too slow, use \(k=3\) only on the 30 ARC items and direct/CoT on all 80; report the imbalance. Do not add debate or tools in the pilot; they are full-study arms.

### 7.3 Outputs (per-query schema)

```text
query_id, benchmark, arm, seed, tokens_prefill, tokens_decode,
energy_J_gross, energy_J_net, latency_s, power_mean_W, power_max_W,
correct, parse_fail, retry_count, tool_calls, timestamp
```

Summary outputs: \(\hat\mu,\hat A,\hat R\) with 95% confidence intervals; \(E^+\) and the failure-inclusion gap; frontier plot with the upper envelope; CEAC on \(\lambda\in[0,2R_{\min}]\); break-even \(\lambda^\star\) for adjacent frontier points; \(\sigma_\Delta\); MDE at the pilot's \(n\); total pilot energy and gCO\(_2\)e.

### 7.4 Kill criteria

| # | Criterion | Threshold | Action if triggered |
|---|---|---|---|
| K1 | Power measurement unusable | `power.draw` missing, or per-query energy CV > 0.35 after averaging | Kill the 3060-only design; require a wall meter or desktop GPU |
| K2 | Accuracy degenerate | All arms < 15% or > 90% on the pooled suite | Change benchmark/model; do not interpret ratios near 0 or 1 |
| K3 | Underpowered | 95% confidence-interval width > 40% of \(\hat R\) at the pilot's \(n\), or MDE > 25% | Kill the fixed-n pilot; redesign with more items or a rental GPU |
| K4 | No decision change | All four arms lie on one frontier segment; CPCA order equals budget-optimal order for all \(B\); no \(E^+\) rank reversal | Downgrade to a measurement note; the repair has no empirical bite |
| K5 | No trade-off | SC3 uses > 8× Direct energy for < 5 pp accuracy gain, and CoT-long is dominated by Direct | The strategy set is uninteresting; change models/benchmark |
| K6 | Measurement overhead | Instrumentation adds > 20% wall time or > 10% energy | Simplify logging; re-run the calibration |
| K7 | Seed instability | The repeated Direct-vs-CoT-long ordering flips between the two seeds on the 20-item replication | Increase seeds/items; do not report a ranking |

**Pass condition:** K1–K3 and K5–K7 pass, and K4 shows at least one of: (a) a CPCA runner-up that is dominated on the frontier, (b) a budget \(B\) at which the CPCA-min is not optimal, or (c) a success-conditioned rank reversal. If K4 shows none, the original G5 is dead and the repair is not worth a six-month project.

---

## 8. Scores

Rubric: Novelty 25%, Method 20%, Math 15%, Experiments 10%, Dataset 10%, Q1 fit 10%, Feasibility 5%, Compute 5%. Core = mean of Novelty, Method, Math, Experiments. Independence is separate (0–10), not part of the weighted score.

| Factor | Original G5: CPCA leaderboard | Repair R1: CPCA-CEA |
|---|---:|---:|
| Novelty | 3.0 | 6.5 |
| Method | 5.0 | 7.5 |
| Math | 3.0 | 7.5 |
| Experiments | 5.0 | 6.5 |
| Dataset | 6.0 | 6.5 |
| Q1 fit | 5.0 | 7.0 |
| Feasibility | 8.5 | 8.0 |
| Compute | 8.0 | 7.5 |
| **Core** | **4.00** | **7.00** |
| **Weighted** | **4.63** | **7.03** |
| **Independence** | **8/10** | **7.5/10** |

**Why the original is low.** Same estimand as Token Arena; same measured quantity as Yang & Armour; same strategy comparison as Mamun/Jin/Kashaniyan; carbon is a constant multiplier in a single-region study; no theorem. It is a solid empirical paper at 4.6, not a Q1 green-computing contribution at the 8+ bar.

**Why the repair is 7, not 8.** The decision mathematics is imported from health economics; the empirical question is crowded; the carbon framing remains a rescaling unless the study adds spatial/temporal carbon-intensity variation or local-vs-remote comparison. The contribution is real but is a methodological correction, not a new theory. It is a strong CONDITIONAL, not a GO.

**Why Independence is 7.5.** The green_computing track is separate, but the repair shares measurement infrastructure with G2 (Green RAG) and the budget/routing framing with G1 (cascade). It must not duplicate G1's policy optimization; its object is measurement and decision certification, not routing.

---

## 9. Verdict and what would change it

**Verdict: KILL G5 as stated; CONDITIONAL on R1.**

- **KILL** the CPCA leaderboard: the estimand is published (Token Arena), the measured version is published (Yang & Armour), and the strategy comparison is published (Mamun; Jin; Kashaniyan). A new table is not a new functional.
- **CONDITIONAL** on R1: the carbon-effectiveness frontier, failure-inclusive paired estimation, break-even carbon price, and CEAC. This is the nearest defensible repair, and it changes the decision from “rank strategies” to “choose a frontier point at a budget/price.”
- **Two-week gate:** before committing, run the pilot in §7 and the literature re-check (a new CEAC/CPCA paper could appear between Nov 2026 and the project start). If K4 shows a dominated CPCA runner-up or a budget-dependent reversal, and K1–K3 pass, proceed to a full study. If not, stop and keep G2/G5 as the green track.
- **What would upgrade to GO (8+):** a frontier configuration with a robust dominated runner-up, a break-even \(\lambda\) confidence interval narrower than 30%, a success-conditioned reversal, and a clear Q1 venue fit (ACM JCSS, IEEE T-SUSC, or NeurIPS D&B). Without all four, keep it at CONDITIONAL.
- **What would kill R1:** all arms collinear/on one frontier segment; CPCA confidence intervals too wide at \(n=300\); no strategy-level effect beyond Direct; or a new paper that already applies CEAC to LLM energy.

**Path:** (1) two-week pilot and re-check; (2) if pass, 6–8 weeks to build the measurement harness and run 2 models × 3 benchmarks × 6 arms; (3) 4 weeks for the frontier/CEAC analysis and robustness; (4) 6 weeks writing; (5) target ACM JCSS or IEEE T-SUSC, with a NeurIPS D&B benchmark track as a fallback. Budget: 3060 local for the pilot; a few hours of 24 GB rental for the 3B/7B replication; no paid API.

---

## Appendix A — Proof sketches

### A.1 Influence function

\[
\sqrt n(\hat R-R)
=\frac{\sqrt n(\bar E-R\bar Y)}{\bar Y}
=\frac{1}{\bar Y}\cdot\frac1{\sqrt n}\sum_i(E_i-RY_i)+o_p(1).
\]

Since \(\bar Y\to A\) and \(E_i-RY_i\) has mean zero and variance \(\sigma^2=\operatorname{Var}(E-RY)\), the limit is \(\mathcal N(0,\sigma^2/A^2)\). The influence function is \(\psi_i=(E_i-RY_i)/A\). \(\square\)

### A.2 Failure-inclusion bias

\[
\mu=A E^+ +(1-A)E^-
\;\Longrightarrow\;
R=\frac{\mu}{A}=E^+ +\frac{1-A}{A}E^-.
\]

The success-conditioned estimator targets \(E^+\), so its bias is \(\frac{1-A}{A}E^-\). The reversal condition follows by subtracting the two biases. \(\square\)

### A.3 Frontier sufficiency and the CPCA-min

For any \(s\), \(A_s\le R_{\min}\mu_s\) by definition of \(R_{\min}\). At \(\lambda=R_{\min}\), \(\mathrm{NCB}_s=R_{\min}A_s-\mu_s\le0=\mathrm{NCB}_{s_0}\). Hence \(s_0\) maximizes a linear objective at \(\lambda=R_{\min}\) and lies on the upper hull. If a point is below the upper hull, a convex combination has \(\mu'\le\mu\) and \(A'\ge A\); the three decision objectives cannot strictly prefer the dominated point. \(\square\)

### A.4 Rate lower bound (sketch)

The paired net-benefit difference \(D_i(\lambda)\) is a mean-zero bounded-variance random variable under the null. Two-point Le Cam: to distinguish \(\mathbb P\) and \(\mathbb Q\) with \(|\mathbb E_{\mathbb P}D-\mathbb E_{\mathbb Q}D|=\delta\) and \(\operatorname{Var}\le\sigma^2\), any test with error \(\alpha,\beta\) needs \(n\gtrsim\sigma^2(z_{1-\alpha}+z_{1-\beta})^2/\delta^2\). The matched upper bound is the usual \(z\)-test sample size. \(\square\)

---

## Appendix B — Verification and ledger notes

- All five closest papers were checked against their arXiv abstract pages on 8 November 2026; Token Arena, Yang & Armour, and Jin et al. were also checked against the full HTML for the exact estimand, sample sizes, and measured/modeled distinction.
- Token Arena’s formula was read directly from §3.1: \(J_{\mathrm{CA}}(e)=j_eT_e/A_e\).
- Yang & Armour’s 33× and 3.7× figures were read from the abstract and §1/§6.
- Mamun et al.’s six strategies, 11 models, and HumanEval+/MBPP+ were read from the abstract.
- Kashaniyan & Jannesari’s four schedules and 4.64–4.86× energy figures were read from the abstract; the SC26 EESP workshop venue is in the journal reference.
- Jin et al.’s Base/MV/RT setup, 10 runs, and 113.48× figure were read from the full text.
- Venue status for preprints is “unverified” unless a journal reference or acceptance comment is present. This is deliberate: the IDs and content are verified; the publication venue is not claimed.
- No paid API, GPU, model load, or download was used in this review. The pilot is a plan, not an executed experiment.

---

*File: `05_g5_cpca.md` · Ledger: `ledger_g5.csv` · Track: green_computing.*
