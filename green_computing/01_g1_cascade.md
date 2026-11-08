# G1 — Carbon-Budgeted Model Cascade

**Date:** 8 November 2026 · **Track:** green computing · **Status:** adversarial formulation + literature kill test.

> **Verdict: KILL (as briefed).** The functional is jointly occupied by GAR (carbon-aware routing with accuracy and p95-latency constraints, arXiv:2605.11603) and C3PO / Calibrate-Then-Delegate (budget-constrained cascades with finite-sample conformal guarantees, arXiv:2511.07396 / arXiv:2604.14251). Carbon is a multiplier on cost; with a constant grid factor the problem is C3PO with a different unit. The only residual that is not a rescaling is **settled vs forecast carbon**: a policy that spends the forecast budget with a 5% safety margin still violates a 5% per-query carbon cap on **27.0%** of UK 2025 half-hours, not 5% (measured below from 17,476 full-year public forecast/actual pairs). That repair is defensible but scores **5.8/10**, below the 8+ bar. Recommend killing G1 and reallocating to G2/G3 unless the team explicitly wants a safe applied paper.

---

## 0. Plain-language problem

A user asks a question. The system can answer with a small local model, a medium local model, or a large remote model. Each option buys quality with energy, latency, and carbon; the carbon per joule depends on the grid at that moment, and the grid is known only through a forecast. The desired policy maximises answer quality subject to a per-query carbon budget and a latency SLA.

The problem is well posed and practically important. It is also now a crowded formulation: one paper already does carbon-aware routing under accuracy and tail-latency constraints, and two others already do budget-constrained cascades with distribution-free overspend guarantees. The carbon version adds a stochastic multiplier, not a new decision problem — unless the multiplier's **forecast error** is treated as the core object.

---

## 1. Formalisation

### 1.1 Primitives

- Query \(x \in \mathcal{X}\) arrives at time \(t\). Latent difficulty \(z\); observable features \(\phi(x)\) (task id, prompt length, embedding).
- Model set \(\mathcal{M}=\{m_1,\dots,m_K\}\), ordered by capability. A **stage** is a pair \((m_k, r)\) where \(r\in\mathcal{R}\) is the serving region; the pilot uses \(K=3\) in one region, the full study adds a remote tier.
- Correctness \(Y_k(x)\in\{0,1\}\), with \(p_k(x)=\Pr(Y_k=1\mid x)\) and \(p_1\le p_2\le p_3\).
- Energy \(E_k(x)\ge 0\) joules, random through output length, prefill length, batching, and phase; latency \(T_k(x)\) seconds.
- Confidence \(s_k\in[0,1]\), a statistic of the stage-\(k\) output. \(\mathcal{F}_k=\sigma(x,s_1,\dots,s_k,E_1,\dots,E_k)\).
- Carbon intensity \(\hat\gamma(t)\) (forecast, available at decision time) and \(\gamma(t)\) (settled, available after the fact), in gCO\(_2\)e/kWh.
- Carbon of a run: \(C_k=\gamma(t_k)\,E_k/3.6\times10^{6}\) grams. (Operational, scope 2; embodied carbon excluded and reported separately.)

### 1.2 Policy class

A cascade policy \(\pi\) is a stopping time \(\tau\in\{1,\dots,K\}\), adapted to \((\mathcal{F}_k)\), plus the final answer \(\hat y_\tau\). The policy chooses **stop or escalate** at each stage; it is not a one-shot argmin over models. This is the only structural difference from GAR, which selects one model per query.

Quality \(Q(\pi)=\mathbb{E}[Y_\tau]\). Carbon \(C(\pi)=\sum_{k=1}^{\tau}C_k\). Latency \(T(\pi)=\sum_{k=1}^{\tau}T_k\).

### 1.3 Three budget semantics (the distinction that matters)

| Semantics | Constraint | Who uses it |
|---|---|---|
| Mean | \(\mathbb{E}[C(\pi)]\le B\) | Wiesner et al. (annual) |
| Rolling-window cumulative | \(\sum_{i=t-w}^{t}C_i\le B_w\) (dual update, no proof) | GAR |
| Per-query chance | \(\Pr(C(\pi)>b)\le\alpha\) | C3PO, Calibrate-Then-Delegate |
| Settled cumulative | \(\Pr(\forall t\le T:\sum_{i\le t}C_i>B_t)\le\alpha\) | **not occupied with a guarantee for LLM cascades** |

The brief's "hard budget" is ambiguous between these four. They have different feasible sets, different estimators, and different optimal policies. Most of the existing literature silently uses a mean or a rolling window.

### 1.4 Estimand / functional

\[
Q^\star(b,\alpha,L)=\sup_{\pi}\ \mathbb{E}[Y_\tau]
\quad\text{s.t.}\quad
\Pr\big(C(\pi)>b\big)\le\alpha,\;\;
\Pr\big(T(\pi)\le L\big)\ge 1-\beta .
\]

The estimand is the **budget-feasible quality frontier** and the policy that attains it. The decision it changes: at each stage, stop or escalate; and, at deployment, whether to promise a per-query carbon cap at all.

Equivalently, the deliverable is the pair \(\big(Q^\star(b,\alpha,L),\ \pi^\star\big)\), not a single accuracy number.

### 1.5 What carbon adds over cost — the wedge

C3PO assumes a fixed, query-independent per-model cost \(c_k\). Carbon is

\[
C_k=\gamma(t)\cdot E_k(x)/3.6\times10^{6},
\]

so the carbon-to-cost ratio is

\[
R_k(x,t)=\frac{C_k}{c_k}=\frac{\gamma(t)\,e_k(x)}{3.6\times10^{6}\,\tilde c_k},
\]

where \(e_k(x)\) is joules per unit of billed cost and \(\tilde c_k\) is the price per unit. The problem collapses to C3PO iff \(R_k(x,t)\) is constant across \(k,x,t\). Write the two wedges separately:

\[
\log R_k(x,t)=\log\gamma(t)+\log\frac{e_k(x)}{\tilde c_k}+\text{const},
\qquad
R_\gamma(t)=\frac{\gamma(t)}{\hat\gamma(t)} .
\]

The first is the **objective wedge** (does carbon routing differ from cost routing?); the second is the **constraint wedge** (is the guarantee valid against settled carbon?). \(\operatorname{Var}(\log R)\) decomposes into a grid term \(\operatorname{Var}_t(\log\gamma)\), a price–energy term \(\operatorname{Var}_k(\log(e_k/\tilde c_k))\), and a length/phase term \(\operatorname{Var}_x(\log e_k(x))\), plus covariance terms.

The first three are already exploited by GAR and by price-aware routers. **The fourth — the forecast error \(R_\gamma\) — is the only one that changes the guarantee**, because it makes the constraint bind against a random variable that the policy does not observe at decision time. The pilot measures all four components (Section 5); if the constraint wedge is small, the idea is C3PO with a green label.

---

## 2. Closest five papers at the functional level (2024–2026)

| # | Paper | ID | Venue status | Functional | What it already owns |
|---|---|---|---|---|---|
| 1 | **GAR: Carbon-Aware Routing for LLM Inference via Constrained Optimization** — Sheshanarayana, Pal, Sinha, Dasgupta | arXiv:2605.11603 | arXiv v1 12 May 2026; ICLR 2026 **withdrawn** (OpenReview `wVd99lgt4j`) — verified | Carbon-aware per-request routing; accuracy floor + p95-latency SLO; rolling carbon budget; primal-dual | The **estimand**: minimise CO\(_2\) subject to quality and tail-latency constraints. Explicitly: "A full regret analysis … is left for future work." |
| 2 | **C3PO: Optimized LLM Cascades with Probabilistic Cost Constraints for Reasoning** — Valkanas, Pal, Rumiantsev, Zhang, Coates | arXiv:2511.07396 | arXiv preprint, no verified venue | Cascade under \(\Pr(\text{cost}>C^\star)\le\alpha\); conformal quantile filter; PAC-Bayes generalisation | The **guarantee shape**: finite-sample, distribution-free probability of budget overspend for a cascade. Costs are fixed and query-independent. |
| 3 | **Calibrate-Then-Delegate: Safety Monitoring with Risk and Budget Guarantees via Model Cascades** — Pona, Kazemi, Hosseini, Du, Watson, Simeone, Paoletti | arXiv:2604.14251 | arXiv preprint, no verified venue | Cascade with finite-sample guarantees on cost **or** safety risk; streaming decisions; delegation-value probe | The **guarantee machinery**: calibrated threshold + multiple testing for a budget/risk constraint in a cascade. |
| 4 | **Carbon-Aware Quality Adaptation for Energy-Intensive Services** — Wiesner, Grinwald, Weiss, Wilhelm, Khalili, Kao | arXiv:2411.19058 | **e-Energy'25** (DOI 10.1145/3679240.3734614; ACM landing page bot-blocked, verified via arXiv comment + DOI) | Adjust the fraction of requests served by quality tiers to stay within an **annual carbon budget**; forecast-based multi-horizon optimisation | The **carbon-budget + quality** objective at service level. Also shows the annual budget is a different (easier) problem than a per-query cap. |
| 5 | **Choosing to Be Green: Advancing Green AI via Dynamic Model Selection** — Cruciani, Verdecchia | arXiv:2509.19996 | Green-Aware 2025 workshop (verified) | Green AI dynamic model **cascading** and **routing** under an accuracy requirement | The **green cascade** application framing. Heuristic; no guarantee. |

**Runners-up (checked, not in the top five):** Conformal Cascade (arXiv:2607.25018) gives distribution-free **accuracy** guarantees for multi-tier LLM inference — the same conformal machinery, different target; CarbonRouter (arXiv:2610.35230) does cache-aware multi-turn carbon routing across regions; CEDAR (Systron Lab, 2026) does queue-level carbon routing for agentic inference; "Dissecting the Carbon-SLO Tension" (arXiv:2610.14571) characterises the carbon–SLO trade-off with a calibrated simulator. The classical **Bandits with Knapsacks** (Badanidiyuru, Kleinberg, Slivkins, 2013) supplies the \(O(\sqrt T)\) resource-constrained regret shape; it is the theory ancestor, outside the 2024–2026 window.

**Method donors the repair must cite and differentiate from:** "Uncertainty-Aware Decarbonization for Datacenters" (arXiv:2407.02390) already applies conformal prediction to carbon-intensity forecast uncertainty for temporal and spatial load shifting; "Contextual Robust Optimization for AI Data Center Scheduling with Statistical Guarantees" (arXiv:2606.17466) already gives finite-sample joint chance-constraint guarantees for AI workload scheduling. Neither does per-query model cascades or a quality–carbon frontier; both operate at cluster/job level. The repair's novelty is the object (settled carbon per query in a cascade), not the conformal-CI machinery.

### 2.1 Exact overlap assessment

- **GAR vs G1.** Same decision variable (which model per query), same constraint family (quality floor, tail latency, carbon), same objective (carbon down). GAR is one-shot routing; G1 is a cascade. The cascade adds a stopping problem, but the feasible frontier is the same object. **Same estimand.**
- **C3PO vs G1.** Same constraint semantics (per-query chance constraint), same action space (cascade with escalation), same guarantee type (conformal, finite-sample). Only the resource differs: dollars/tokens vs grams CO\(_2\). With a constant \(\gamma\), \(C=\gamma\cdot\text{cost}\), and C3PO applies after rescaling. **Same guarantee.**
- **CTD vs G1.** Same guarantee shape, different target (safety risk vs budget). Confirms the guarantee is a standard tool, not a differentiator.
- **Wiesner et al. vs G1.** Same carbon-budget + quality objective, but annual and service-level. Their budget is soft over a year; G1's per-query cap is a different (harder) object.
- **Cruciani & Verdecchia vs G1.** Same application (green cascade), heuristic, no guarantee. Occupies the "green cascade" framing but not the theory.

### 2.2 Kill decision

No single paper has *both* the carbon estimand and the budgeted-cascade guarantee. But the pair (GAR, C3PO) spans the functional, and composing them is a unit change. Under the brief's 8+ rule — a **new functional**, not a new application — G1 fails: carbon-aware routing is a functional with at least five papers in 20 months, and the budgeted-cascade guarantee is a functional with at least two. The residual delta (forecast error) is a statistical wrinkle, not a new decision problem.

**KILL as briefed.**

### 2.3 Nearest defensible repair: settle what you schedule

The one thing neither GAR nor C3PO does is guarantee the budget **against settled carbon** when decisions use a forecast. GAR's own limitations section concedes that its carbon-intensity data "may lag real-time conditions by several hours." That is the gap.

**Repaired estimand.**

\[
Q^\star_{\text{settled}}(b,\alpha)=\sup_\pi \mathbb{E}[Y_\tau]
\quad\text{s.t.}\quad
\Pr\Big(\sum_{k\le\tau}\gamma(t_k)E_k/3.6\times10^{6}>b\Big)\le\alpha,
\]

where the policy sees only \(\hat\gamma(t)\), and the probability is over the joint law of \((\hat\gamma,\gamma,x)\). The decision changes from "which model" to "how much of the forecast budget may I spend," i.e., a **carbon-forecast safety factor**.

**Repair contributions (the salvage paper).**

1. **Frontier lower bound.** At the budget frontier, a plug-in that spends the forecast budget \(S=b\) to the cap violates a per-query cap with probability \(\Pr(\gamma>\hat\gamma)\) — 43.9% in UK 2025 data — for any calibration-set size.
2. **Settlement-conformal calibration.** Split-conformal quantile of the ratio \(R=\gamma/\hat\gamma\) gives a finite-sample safety factor \(r_{1-\alpha}\); spending \(\hat\gamma E\le b/r_{1-\alpha}\) restores \(\Pr(C>b)\le\alpha+1/(n+1)\). Mondrian by region × hour handles the policy's endogenous selection of high-error strata.
3. **Price of robustness.** The quality cost is approximately \(\kappa\,b\,(1-1/r_{1-\alpha})\), where \(\kappa\) is the local slope of the quality–carbon frontier; measurable, and the pilot estimates it.
4. **Carbon–cost wedge.** A variance decomposition (Section 1.5) that says exactly when the carbon problem is C3PO in disguise.

**Why this is only a ~5.8 paper.** The fix is a one-line safety factor once the problem is posed; the conformal step is standard; the empirical fact is striking but not a method. It is a good applied safety paper for a sustainability venue, not an 8+ methods contribution.

---

## 3. Theorem / bound shape and proof sketches

### T1 — Budget-index structure (the policy is not a static threshold)

**Assumptions.** (A1) \(p_k(x)\) nondecreasing in \(k\). (A2) MLRP: the likelihood ratio of \(s_k\) under \(Y_k=1\) vs \(Y_k=0\) is increasing in \(s_k\). (A3) \(E_k\perp Y_k\mid x\). (A4) spent carbon is metered after each stage, so the remaining budget \(r=b-\sum_{j\le k}C_j\) is observed.

**Statement.** The optimal policy for the per-query chance problem is a threshold policy: escalate at stage \(k\) iff
\[
s_k\le\theta_k(r),
\]
with \(\theta_k\) nonincreasing in \(r\). As \(r\to\infty\), \(\theta_k(r)\to\theta_k^\infty\), the unconstrained deferral threshold; as \(r\to 0\), \(\theta_k(r)\to 0\). A static "quality-per-joule" threshold is the \(r=\infty\) special case and is **not** optimal when the budget binds.

**Proof sketch.** Backward induction on \(k\). Define \(V_k(r,s)=\max\{p_k(s),\ \mathbb{E}[V_{k+1}(r-C_{k+1},s_{k+1})]\}\). A1–A2 make the stop value increasing in \(s\) and the continuation value independent of \(s\) given the state; single crossing gives a threshold. A3 and a coupling argument give \(\partial V_k/\partial r\ge0\) (a larger reserve weakly expands the feasible continuation set), hence \(\partial\theta_k/\partial r\le0\). Standard optimal-stopping arguments (Topkis monotone comparative statics) supply the lattice conditions.

**What would make it false.** If confidence is uninformative (A2 fails), the optimal rule depends on \(x\) beyond \(s_k\). If energy correlates with correctness (A3 fails — e.g., a model that is right on long outputs), the continuation value is no longer separable. If the budget is not metered (A4 fails), the policy is open-loop and the threshold is static — which is exactly the deployed case for many routers.

### T2 — Price of a chance constraint (hard budget ≠ mean budget)

**Statement.** Let \(\Delta C\) be the carbon of escalation, and let \(E\) be the escalation event with \(\Pr(E)=p_E\). A per-query chance constraint at level \(\alpha\) requires, on \(E\),
\[
\Pr(C_{\text{pre}}+\Delta C\le b\mid E)\ge 1-\alpha/p_E .
\]
Hence the policy must reserve the \((1-\alpha/p_E)\)-quantile of \(\Delta C\):
\[
C_{\text{pre}}\le b-q_{1-\alpha/p_E}(\Delta C).
\]
The mean-budget policy reserves \(\mathbb{E}[\Delta C]\). The **price of the chance constraint** is
\[
\Delta(b,\alpha)=q_{1-\alpha/p_E}(\Delta C)-\mathbb{E}[\Delta C]\ge 0,
\]
and the attainable quality satisfies
\[
Q^\star_{\text{chance}}(b,\alpha)\le Q^\star_{\text{mean}}\big(b-\Delta(b,\alpha)\big),
\]
where \(Q^\star_{\text{mean}}(\cdot)\) is indexed by the pre-escalation budget; equivalently, the chance constraint shifts the effective pre-escalation budget down by \(\Delta\). The effective price is zero if \(\Delta C\) is deterministic, or if the cap is slack (\(\alpha\ge p_E\), in which case no reserve is required). For heavy-tailed \(\Delta C\), \(\Delta\) can exceed \(\mathbb{E}[\Delta C]\).

**Proof sketch.** Feasible-set inclusion. The chance constraint is a quantile constraint on the total; conditional on escalating, the pre-escalation spend must leave room for the \((1-\alpha/p_E)\)-quantile. The mean constraint only requires room for the mean. Since \(q\ge\mathbb{E}\), the chance-feasible pre-escalation set is a subset of the mean-feasible set with the shifted budget. The quality inequality follows by monotonicity of \(Q^\star_{\text{mean}}\) in its budget argument. A first-order Taylor expansion of the frontier gives the interpretable form \(\Delta Q\approx\kappa\,\Delta\).

**What would make it false.** If escalation carbon is deterministic, or if the cap never binds, the two budgets coincide. If the service can **pool risk across queries** (annual budget), the per-query quantile is the wrong object and the bound is vacuous — Wiesner et al.'s annual formulation is then the right one. If the quality–carbon frontier is flat (\(\kappa\approx0\)), the price is zero and the distinction does not matter.

### T3 — Settlement-conformal guarantee (the repair's core)

**Assumptions.** (A5) calibration pairs \((\hat\gamma_i,\gamma_i)\) are exchangeable with the deployment pair; (A6) the ratio \(R=\gamma/\hat\gamma\) is independent of the query and the policy's stage choice, or calibration is Mondrian by stratum (region × hour).

**Statement.** Let \(R_{(1)}\le\dots\le R_{(n)}\) be the order statistics of \(n\) calibration ratios, and \(k=\lceil(n+1)(1-\alpha)\rceil\). If the policy spends forecast carbon \(S=\hat\gamma E\le b/R_{(k)}\), then
\[
\Pr\big(\gamma E>b\big)\le\alpha+\frac{1}{n+1}.
\]
With \(S\) strata, the union bound replaces \(1/(n+1)\) by \(S/(n+1)\), and the feasible budget uses the worst stratum's quantile.

In deployment \(E\) is itself random and unknown at decision time. Replace it by a split-conformal upper bound \(\hat E_{1-\beta}\) calibrated on the same energy log, and combine the two bounds by a union bound: the guarantee becomes \(\alpha+\beta+2/(n+1)\). This is the honest form; a guarantee that treats \(E\) as known is not deployable.

**Proof sketch.** Split conformal for a monotone transform. The event \(\gamma E>b\) is \(R>b/S\); choosing \(b/S=R_{(k)}\) gives coverage \(k/(n+1)\ge1-\alpha\). Exchangeability gives the finite-sample correction. Mondrian calibration restores validity when the policy selects strata endogenously.

**What would make it false.** Nonstationary forecast error (new grid topology, weather regime, demand shock) breaks exchangeability; the guarantee degrades silently. If \(\hat\gamma\approx0\), the ratio is undefined; use an absolute-error formulation instead. If the policy selects high-error strata and calibration is marginal, the guarantee is invalid — this is the most likely practical failure. If the deployment cannot meter per-query energy, the bound is unenforceable.

### T4 — Forecast-plug-in lower bound (the decision-changing fact)

**Statement.** Under A5–A6 and a continuous error distribution, any policy that spends the forecast budget to the cap (\(S=b\)) violates the realized per-query cap with probability
\[
\Pr(R>1)=1-F_R(1).
\]
A policy that adds a naive margin \(m\) (\(S=b/(1+m)\)) violates with probability \(1-F_R(1+m)\).

**Empirical instantiation (UK 2025, 17,476 half-hourly forecast/actual pairs).**

| Policy | Forecast spend | Realized violation |
|---|---:|---:|
| No margin | \(b\) | **43.9%** |
| Naive 5% margin | \(b/1.05\) | **27.0%** |
| Correct 5% cap (\(r_{0.95}=1.241\)) | \(b/1.241\) | 5% (by construction) |
| Correct 10% cap (\(r_{0.90}=1.163\)) | \(b/1.163\) | 10% |
| Correct 1% cap (\(r_{0.99}=1.449\)) | \(b/1.449\) | 1% |

The 5% cap requires a **19.4% cut** to the forecast budget, not a 5% margin. This is the number that changes a deployment decision: GAR-style forecast budgeting at the cap is infeasible for a chance constraint, and the error is more than fivefold at \(\alpha=0.05\).

**Proof.** Immediate from the definition of \(R\); the contribution is the measurement, not the algebra. The empirical distribution is right-skewed (mean ratio 1.011, median 0.992, q95 1.241, q99 1.449, Pearson \(r=0.978\), MAPE 8.7%, P95 absolute percentage error 26.8%), so the mean is a bad safety factor and the quantile is the right one.

**What would make it false.** If realized intensity is observable at decision time (no lag), \(R\equiv1\) and the bound is vacuous. If the deployment is in a low-error grid (q95 < 1.08), the correction is small. If the budget is annual, errors pool and the per-query quantile overstates the problem.

### Corollary — when the whole thing collapses

If the objective wedge is zero (constant grid factor, price proportional to energy) **and** the constraint wedge is zero (\(R_\gamma\equiv1\), no forecast error), then \(C\) is a deterministic rescaling of C3PO's cost, T1–T3 reduce to existing results, and the carbon paper has no independent content. **This is the primary kill switch.**

---

## 4. Dataset and measurement protocol

### 4.1 Public QA sets

| Set | Size (test) | Use in pilot | Metric |
|---|---:|---|---|
| MMLU (Hendrycks et al., 2021) | 14,042, 57 subjects | 80 items, 10 subjects | letter accuracy |
| ARC-Challenge (Clark et al., 2018) | 1,172 | 40 items | letter accuracy |
| GSM8K (Cobbe et al., 2021) | 1,319 | 40 items, 8-shot CoT | exact match on final number |

Calibration/test split 40/60, random within dataset. For conformal validity the split must be exchangeable; do **not** stratify by difficulty. For conditional-coverage analysis, report by prompt-length quartile as a post-hoc diagnostic.

### 4.2 Models

Pilot (fits 6 GB): Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, all Q4_K_M via the same llama.cpp runtime, batch 1, context 2048, greedy, `max_new_tokens=160`. Same runtime is essential: mixing HF and llama.cpp confounds energy with the serving stack. None of the three is saturated on these sets (expected MMLU 45–65%, ARC 45–60%, GSM8K 55–80%), so cascade headroom exists; if the 3B model saturates a set, K6 fires.

Full study: add a 7B/8B 4-bit model on a 24 GB rental as the remote tier; add a second region only if the multi-region claim is retained (otherwise it duplicates CarbonRouter).

### 4.3 Energy measurement

- `nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 100` for the duration of each generation.
- Integrate power over wall-clock with the trapezoid rule; subtract the idle baseline measured for 60 s before and after the run.
- Report J/query, J/output token, J/1k prompt tokens; 5 repeats on 30 fixed queries; median and IQR; measurement CV.
- **Honesty about the laptop 3060.** `power.draw` may be coarse or unavailable. Check the min/max granularity first. If coarse, cross-check with `codecarbon` (if installed) and calibrate a proxy \(P\approx P_{\text{idle}}+u\,(P_{\text{TDP}}-P_{\text{idle}})\) against a wall meter on 10 queries. State the proxy and its error; never silently substitute.
- Boundary: operational carbon only (scope 2). Embodied/host carbon excluded; EcoServe (arXiv:2502.05043) finds host processing dominates embodied carbon, so the exclusion must be stated because it can materially change the attributable share.

### 4.4 Carbon intensity

- **Primary:** UK National Grid ESO Carbon Intensity API, `https://api.carbonintensity.org.uk/intensity/{from}/{to}`. Free, no key, 30-minute resolution, returns both `forecast` and `actual`. This is exactly the forecast/settled pair the repair needs. The full-year 2025 pull (17,476 valid points) gives a mean actual of 129.2 gCO\(_2\)/kWh.
- **Fixed regional average (brief's requirement):** use 129.2 gCO\(_2\)/kWh for the UK 2025 mean, and report the time-varying result separately. Do not mix the two.
- **Alternatives:** ENTSO-E Transparency Platform (free key), US EIA-930 (free), CAISO OASIS (free). Avoid paid APIs; Electricity Maps' free tier is limited and should not be the sole source.
- Conversion: \(C[\text{g}]=E[\text{J}]/3.6\times10^{6}\times\gamma[\text{g/kWh}]\).

### 4.5 Logging schema

`query_id, dataset, split, model, region, prompt_tokens, output_tokens, latency_ms, energy_J, power_trace_path, confidence, correct, ci_forecast, ci_actual, carbon_g, timestamp`.

### 4.6 Statistical plan

- Primary: violation rate of the per-query cap at \(\alpha=0.05\), Wilson 95% CI. A two-proportion test for 0.27 vs 0.05 needs ~57 queries per arm at 90% power; the 160-query pilot is adequately powered even before exploiting the paired design, and the 2,000-query study gives ±1%.
- Secondary: quality at matched mean carbon, paired bootstrap over queries (10,000 resamples).
- Conditional coverage: binomial test per prompt-length quartile, Bonferroni across four strata.
- Wedge: variance decomposition of \(\log R\); report each component with bootstrap CIs.

---

## 5. Pilot: ≤2 h on the 3060, with kill criteria

**Hypothesis (falsifier).** Settled-carbon budgeting is a different problem from forecast/mean budgeting at a decision-relevant scale: forecast-plug-in routing violates a 5% per-query cap far above 5%, and the correction costs less than 3 accuracy points.

**Design.**

| Phase | Work | Time |
|---|---|---:|
| Setup | load 3 models, warm up, verify telemetry | 10 min |
| Baseline | 60 s idle power before/after | 2 min |
| Main run | 160 queries × 3 models, greedy, 160 new tokens | ~45 min |
| Repeats | 15 queries × 2 extra reps × 3 models | ~10 min |
| Analysis | energy/carbon distributions, wedge, policy simulation | 25 min |
| Contingency | model reload / failed generations | 20 min |
| **Total** | | **~112 min** |

**Sample:** MMLU 80 (8 subjects × 10), ARC 40, GSM8K 40. Full factorial (every model on every query) so the oracle frontier and counterfactual policies are computable.

**Analyses.**
1. Energy distribution: CV, q95/mean, q99/mean, by model and dataset.
2. Wedge: variance of \(\log R\) across model, time, and query; Jaccard overlap of the carbon-optimal and price-optimal escalation sets.
3. Forecast error: recompute the UK ratio quantiles on the current 12-month window; confirm 27.0% at 1.05 and 43.9% at 1.0.
4. Policy simulation: oracle frontier; mean-budget; C3PO-style marginal conformal; settlement-calibrated. Report violation rate at \(\alpha=0.05\) and quality at matched mean carbon.
5. Confidence AUC for correctness, per model.

**Kill criteria (pre-registered).**

| # | Test | Kill if |
|---|---|---|
| K1 | Forecast error | \(q_{0.95}(R)<1.08\) **or** \(\Pr(R>1.05)<0.10\) — settlement framing has no teeth |
| K2 | Price of robustness | settlement-calibrated policy loses \(>5\) accuracy points vs forecast-plug-in at matched mean carbon (3–5 points = revise, not kill) |
| K3 | Carbon ≠ cost | Jaccard(carbon-optimal, price-optimal escalation sets) \(>0.95\) — C3PO applies after rescaling |
| K4 | Measurement | energy CV across repeats \(>25\%\) — cannot resolve policies |
| K5 | Deferral signal | confidence AUC \(<0.65\) for at least two models |
| K6 | Cascade headroom | oracle cascade gains \(<3\) points over the best single model at matched carbon |

**Pass condition:** all six pass, and the settlement policy achieves \(\le5\%\) violation at \(\le3\) accuracy points lost. Only then is the salvage worth a full study.

**Decision-changing outcome.** If the pilot reproduces ~27% violation for the forecast-plug-in and \(\le5\%\) for the settlement policy at \(\le3\) points, a deployment decision changes: promise the cap only with settlement calibration. If the violation is \(<10\%\), nothing changes and the paper is not worth writing.

---

## 6. Scores (0–10; rubric weights 25/20/15/10/10/10/5/5)

### 6.1 G1 as briefed

| Factor | Score | Justification |
|---|---:|---|
| Novelty | 3 | Estimand owned by GAR; guarantee owned by C3PO/CTD; composition is a unit change. |
| Method | 5 | Cascade + conformal + measurement; all components standard. |
| Math | 4 | No non-identity result beyond known conformal/PAC bounds; the carbon multiplier is a rescaling. |
| Experiments | 4 | 3060, three small models; weak external validity; the CI trace adds realism. |
| Dataset | 5 | Public QA + public CI; standard, reusable log. |
| Journal fit | 5 | Sustainability venues, not top ML. |
| Feasibility | 9 | Doable in 6 months with the stated hardware. |
| Budget | 9 | Fits the 3060 + small rental budget. |

Core (1–6, renormalised over 90):
\[
(3\cdot25+5\cdot20+4\cdot15+4\cdot10+5\cdot10+5\cdot10)/90=375/90=4.2.
\]

Weighted (all eight):
\[
(375+9\cdot5+9\cdot5)/100=465/100=4.7.
\]

**Independence:** prior-art 3/10 (derivative of GAR + C3PO); portfolio 7/10 provisional (separate green track; shares measurement plumbing with G2/G5 but not the estimand). FYDP/HaluRISC/IntentGate overlap not assessable from this workspace.

### 6.2 Repaired version (settled-carbon budgets)

| Factor | Score | Justification |
|---|---:|---|
| Novelty | 5 | Forecast-vs-settled carbon is not occupied for LLM cascades; the fix is a composition of known tools. |
| Method | 5 | Conformal ratio calibration + budget-index cascade + measurement; solid, not new. |
| Math | 5 | Frontier lower bound, price of robustness, conditional coverage; non-identity but shallow. |
| Experiments | 6 | The UK 27% vs 5% result is striking, cheap, and replicable; the 3060 cascade is small. |
| Dataset | 6 | Forecast/actual CI pairs + per-query energy log is a reusable artifact. |
| Journal fit | 7 | Strong fit for sustainable-computing venues. |
| Feasibility | 8 | Pilot is 2 h; full study is 6 months. |
| Budget | 9 | Well within the budget. |

Core:
\[
(5\cdot25+5\cdot20+5\cdot15+6\cdot10+6\cdot10+7\cdot10)/90=490/90=5.4.
\]

Weighted:
\[
(490+8\cdot5+9\cdot5)/100=575/100=5.8.
\]

**Independence:** prior-art 5/10; portfolio 7/10 provisional.

---

## 7. Verdict and what would change it

**KILL for the 8+ portfolio.** The functional is occupied; the residual is a statistical correction, not a new decision problem. The repaired version is a defensible ~5.8/10 applied paper — worth doing only if the team's objective is a safe Q1-fit sustainability paper rather than an 8+ methods contribution.

**What would upgrade the verdict to CONDITIONAL (6.5–7):**
- the 5% cap violation for the forecast plug-in is \(\ge20\%\) (measured 27.0%) **and** the settlement-calibration quality loss is \(\le1.5\) points, **and**
- the carbon–cost wedge is large (Jaccard \(<0.7\)), **and**
- a conditional-coverage theorem with a non-trivial sample-complexity separation between marginal and Mondrian calibration is proved.

The measured \(q_{0.95}(R)=1.241\) already supports the first clause; the missing pieces are the quality-loss measurement and the theory. On current evidence even CONDITIONAL is not met.

**What would change it to GO (unlikely):**
- A new action space that is not routing or cascading — e.g., carbon-budgeted **batch composition** with non-separable per-query energy, or carbon-budgeted **verifier design** where the verifier itself consumes energy — with an impossibility or approximation bound that does not reduce to a known resource-allocation result.

**What would kill even the salvage:**
- Annual/cumulative budgeting with risk pooling (Wiesner et al. already covers it).
- Realized CI observable at decision time (no forecast error).
- Embodied carbon dominating the total footprint (EcoServe shows host processing dominates embodied carbon), making per-query operational routing second-order.
- A hidden paper doing settlement-calibrated carbon budgets (search again before committing; the lane is moving monthly).

---

## 8. Six-month path if the salvage is pursued

| Month | Milestone | Gate |
|---|---|---|
| 1 | Reproduce the UK forecast-error analysis; lock the settled estimand; re-check GAR v2 and new 2026 papers | Stop if \(q_{0.95}(R)<1.08\) |
| 2 | 3-model cascade harness; 2,000-query full-factorial energy/carbon log | Stop if measurement CV \(>25\%\) |
| 3 | Budget-index policy + settlement-conformal calibration; pilot comparison | Stop if quality loss \(>5\) points |
| 4 | Full evaluation on MMLU/ARC/GSM8K; ablations (constant CI, no forecast error, no energy tail); 8B tier on rental | Stop if Jaccard \(>0.95\) |
| 5 | Robustness: nonstationary CI, region shift, conditional coverage; write | — |
| 6 | Internal review, submission | — |

**Venues (verify current quartile at submission; do not assume Q1):** IEEE Transactions on Sustainable Computing; Sustainable Computing: Informatics and Systems; ACM e-Energy; IEEE International Conference on Sustainable Computing and Communications. A methods venue (NeurIPS/ICML) is not realistic for this contribution.

---

## 9. Evidence log

**Search coverage and limits.** arXiv API metadata queries: `carbon-aware AND LLM`; `energy-aware AND routing AND language model`; `model cascade AND energy`; `carbon AND budget AND inference`; `carbon budget AND language model`; `energy budget AND language model`; `chance-constrained AND language model`; `bandits with knapsacks`; `cascade AND deferral`; `carbon intensity AND forecast AND uncertainty`; plus keyless web search for carbon-aware routing. Semantic Scholar and Google Scholar were not exhaustively crawled; the lane moves monthly, so re-run the scan before committing. All five closest papers were read at the primary source (abstract and, for GAR and C3PO, full HTML).

- **GAR** — arXiv:2605.11603 (v1, 12 May 2026); HTML read 8 Nov 2026. OpenReview `wVd99lgt4j`, ICLR 2026 withdrawn. Key quote: "A full regret analysis for discrete LLM routing with rolling-window constraints is left for future work."
- **C3PO** — arXiv:2511.07396 (v1, 10 Nov 2025); HTML read. Theorem 1: conformal cost guarantee; costs fixed and query-independent; experiments on GSM8K/MATH-500/BigBench-Hard/AIME.
- **Calibrate-Then-Delegate** — arXiv:2604.14251 (v2); abstract read. Finite-sample risk/budget guarantees; streaming.
- **Wiesner et al.** — arXiv:2411.19058 (v4, 4 Mar 2026); e-Energy'25; DOI 10.1145/3679240.3734614 (ACM landing page bot-blocked; venue verified via arXiv comment + DOI).
- **Cruciani & Verdecchia** — arXiv:2509.19996 (v1, 24 Sep 2025); Green-Aware 2025 workshop.
- **Conformal Cascade** — arXiv:2607.25018 (v3); distribution-free accuracy guarantee, no carbon.
- **CarbonRouter** — arXiv:2610.35230 (v1, 26 Oct 2026); multi-turn, cache-aware, multi-region.
- **CEDAR** — Systron Lab, 2026 (PDF, `systronlab.github.io/files/publications/2026-cedar.pdf`); queue-level carbon routing, simulation.
- **Dissecting the Carbon-SLO Tension** — arXiv:2610.14571 (v1, 10 Oct 2026); calibrated simulator, MILP schedulers.
- **Conformal / robust carbon-forecast donors** — "Uncertainty-Aware Decarbonization for Datacenters" (arXiv:2407.02390) uses conformal prediction for CI forecast uncertainty in temporal/spatial load shifting; "Contextual Robust Optimization for AI Data Center Scheduling with Statistical Guarantees" (arXiv:2606.17466) gives finite-sample joint chance-constraint guarantees for AI workload scheduling. Both are cluster/job level; neither is a per-query cascade.
- **Carbon intensity data** — UK National Grid ESO API, `https://api.carbonintensity.org.uk/intensity/2025-01-01T00:00Z/2026-01-01T00:00Z` (pulled in 15-day chunks; 31-day ranges are rejected), pulled 8 Nov 2026: 17,476 valid half-hour points; mean forecast 130.12, mean actual 129.15 gCO\(_2\)/kWh; ratio q50 0.992, q90 1.163, q95 1.241, q99 1.449; \(\Pr(\gamma>\hat\gamma)=0.439\); \(\Pr(\gamma>1.05\hat\gamma)=0.270\); \(\Pr(\gamma>1.25\hat\gamma)=0.046\); Pearson \(r=0.978\); MAPE 8.7%; P95 absolute percentage error 26.8%.
- **Energy context** — "The SIMPLE Law for Energy Cost of LLM Inference" (arXiv:2611.03330) and "Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms" (arXiv:2608.28044) support the claim that energy is not linear in tokens; "Understanding Efficiency: Quantization, Batching, and Serving Strategies" (arXiv:2601.22362) shows serving-stack choices can change per-request energy by orders of magnitude. These are why the energy model must be query- and configuration-conditional, not a fixed per-model constant.
- **EcoServe** — arXiv:2502.05043; host processing dominates embodied carbon, so an operational-only boundary can overstate the share that GPU routing can move. A material falsifier for the whole operational-routing framing.

---

## 10. One-paragraph adversarial summary

G1 is the right question asked two years too late. GAR already formalises carbon-aware routing with accuracy and p95-latency constraints and reports 74% carbon reduction; C3PO already gives a conformal, finite-sample guarantee that a cascade exceeds a cost budget with probability at most \(\alpha\); CTD already does the same for safety risk; Wiesner et al. already adapt quality tiers to an annual carbon budget; Cruciani and Verdecchia already frame the green cascade. The carbon instantiation is a unit change unless the grid factor is stochastic and mis-forecast. It is — 27% of UK 2025 half-hours exceed a 5% forecast margin — and that gives a real but narrow salvage: settle what you schedule, calibrate the ratio \(R=\gamma/\hat\gamma\) conformally, and charge the price of robustness. That is a 5.8/10 applied safety paper, not an 8+ contribution. Kill G1 and put the team on G2 (energy-optimal retrieval depth) or G3 (carbon-aware quantization allocation), where the functional is less crowded and the theorem can be about a decision that does not already have a paper.
