# Master Prompt: Adversarial Research-Novelty Investigation and Project Selection

**Target model:** DeepSeek V4.1 Flash (1M context) or another capable research agent  
**Research cutoff:** 5 November 2026  
**Objective:** Identify whether any candidate can support a genuinely novel, mathematically defensible, experimentally strong international journal paper within approximately 5–6 months.

---

## 0. Your role and operating principle

Act as a coordinated research group consisting of:

- a senior research scientist and journal editor;
- a domain-specific literature-review team;
- a mathematical statistician / machine-learning theorist;
- an experimental ML researcher;
- a dataset and benchmark auditor;
- an adversarial Reviewer 2 whose job is to find reasons the project should be rejected.

You are the **main research orchestrator**. You should use subagents for independent candidate investigations, literature searches, mathematical review, feasibility review, and adversarial comparison whenever subagent functionality is available. Do not merely describe that you would use subagents—delegate the work and collect their findings.

If subagents are unavailable, perform the same work as separate, explicitly independent investigation passes. Do not silently skip a workstream.

### Primary principle

**Do not promote an idea because it sounds novel, combines fashionable terms, has an appealing application, or is easy to implement. Try to disprove its novelty first.**

The goal is not to guarantee a Q1 publication. The goal is to find the strongest defensible research contribution, or conclude that none of the candidates is ready.

A different application domain, new dataset, additional feature, new name, or combination of established methods is not automatically methodological novelty. Compare the underlying problem formulation, estimand, assumptions, objective, algorithm, and guarantees.

---

## 1. Researcher and project context

The researcher is a computer-science student/software engineer working with a team of **3–4 people**. They are interested in NLP, LLMs, ML, statistics, causal inference, agentic AI, software engineering, and other areas where a meaningful research contribution is possible. The project does **not** need to be restricted to Data Mining or NLP.

### Existing projects — avoid duplicating their central contributions

1. **FYDP:** https://github.com/Atik203/FYDP  
   Trust-calibrated multi-agent scientific deliberation; evidence-grounded trust weighting to reduce sycophantic consensus and minority collapse.

2. **HaluRISC:** https://github.com/Atik203/HaluRISC  
   Black-box LLM hallucination-risk prediction, evidence-aware ML features, XGBoost, calibration, SHAP, and cross-domain evaluation.

3. **IntentGate:** https://github.com/Atik203/IntentGate  
   Request-derived intent consistency and runtime action authorization for tool-using LLM agents; injection and tool-use security.

Use these repositories as context for portfolio independence, available engineering experience, and possible reusable infrastructure. Do not recommend a superficial extension of their main research questions as a new independent project. It is acceptable to reuse engineering practices, evaluation harnesses, statistical tools, or general-purpose code where appropriate.

### Constraints and preferences

- **Timeline:** approximately 5–6 months for research execution and manuscript preparation.
- **Team:** 3–4 people.
- **Compute:** local NVIDIA RTX 3060 with 6 GB VRAM, Google Colab GPUs when available, CPU resources, and limited paid GPU/API spending.
- **Budget:** constrained; prefer open-source models, public datasets, efficient experiments, and selective use of paid APIs. Give explicit cost assumptions rather than pretending all APIs or GPUs are free.
- **LLM agents:** available to accelerate literature mapping, coding, dataset discovery, test creation, experiment orchestration, analysis support, and documentation. Do not assume agents eliminate human annotation, ground-truth verification, mathematical proof checking, or independent reproduction.
- **Language:** English-only is entirely acceptable and may be preferable for broad international relevance. Bangla or multilingual data should be optional, not a requirement or a claimed source of novelty by itself.
- **Publication ambition:** a strong international journal, ideally a currently verified Q1 venue in the relevant category. Do not guarantee acceptance or assume a journal is Q1 without checking its current category-specific status.
- **Research preference:** prioritize substantive novelty, methodological contribution, mathematical/theoretical depth, strong empirical evidence, and a reusable dataset/benchmark when it genuinely helps.
- **Do not restrict the search to Data Mining.**

---

## 2. Candidate research directions to investigate

Investigate all four candidates independently. You may reframe, merge, split, or reject them—but justify the decision using literature and mathematical comparison.

### Candidate 1 — Counterfactual evaluation of tool-using LLM agents

**Working title:** *Counterfactual Evaluation of Tool-Using LLM Agents under Sequential Policy Shift and Selective Feedback*

**Core question:** Can we estimate the value, reliability, or task performance of a new stateful LLM-agent policy from historical trajectories generated by different policies, supplemented by a limited randomized evaluation/audit budget?

Potential issues include:
- sequential rather than one-step contextual-bandit decisions;
- stateful tool interactions and action-dependent observations;
- limited overlap/positivity between behavior and target policies;
- partial or delayed feedback;
- unobserved counterfactual outcomes;
- trajectory distribution shift;
- variance and effective sample size of off-policy estimators;
- safe or budget-constrained randomized exploration.

Potential method families to audit include sequential off-policy evaluation, importance sampling, per-decision importance sampling, doubly robust estimators, marginalized importance sampling, model-based OPE, causal inference, POMDPs, partial identification, sensitivity analysis, and active experimental design.

**Do not claim novelty merely because the trajectories come from LLM agents or use tools.** Determine whether a genuinely new estimator, estimand, assumption set, guarantee, or evaluation protocol is required.

### Candidate 2 — MNAR-aware experimental/observational data fusion for software dependency breakage

**Working title:** *MNAR-Aware Experimental–Observational Data Fusion for Downstream Software Dependency Breakage Risk*

**Core question:** Can a small randomized CI replay/audit sample be combined with large historical, selection-biased upgrade outcomes to estimate downstream dependency breakage risk more accurately and with valid uncertainty quantification?

Potential data sources include dependency graphs, package releases, upgrade histories, repository metadata, CI pipelines, issue/incident reports, and controlled replay outcomes. Candidate ecosystems include npm, PyPI, Maven/Java, and other ecosystems where reproducible builds and metadata are accessible.

The formulation must distinguish:
- historical outcomes observed only for projects that chose to upgrade;
- audit/replay outcomes sampled with known inclusion probabilities;
- projects for which replay is infeasible or non-reproducible;
- the target population and exact estimand;
- real downstream behavioral failure versus build/test failure;
- selection bias from outcome-dependent missingness;
- measurement error and imperfect labels in mined incidents.

Potential method families to audit include prediction-powered inference, PPI with inverse-probability weighting, active statistical inference, active multiple-prediction-powered inference, experimental/observational data fusion, MNAR identification, sensitivity analysis, partial identification, survey sampling, two-phase sampling, transportability, and software dependency risk prediction.

**Critical question:** If randomized audit/replay outcomes are observed with known sampling probabilities, do design-based or existing prediction-powered estimators already solve the estimation problem? What additional role, if any, does the MNAR historical dataset play? Partial identification should not be added unless the target parameter is genuinely not point-identifiable under the stated observation process.

**Critical distinction:** CI replay is not automatically equivalent to real-world behavioral breakage. Define what replay can and cannot establish and validate the relationship where possible.

### Candidate 3 — Audit-budgeted selective learning under policy shift and delayed feedback

**Working title:** *Adaptive Auditing under Policy Shift and Delayed Selective Feedback: Risk Estimation and Learning with a Fixed Audit Budget*

**Core question:** Under which conditions can a learner allocate a limited audit budget across selectively observed, delayed, and policy-dependent labels while controlling population risk or estimation error?

Potential components:
- policy-dependent selective labels;
- delayed outcomes;
- changing selection policies;
- non-random missingness;
- audit costs that vary by example;
- adaptive sampling;
- uncertainty or risk bounds;
- learning and evaluation under feedback loops.

Potential method families to audit include active statistical inference, selective-label learning, active learning with nonuniform query costs, prediction-powered inference, PPI-IPW, data fusion under MNAR, partial identification, fairness auditing under partial feedback, online learning, and adaptive experimental design.

**Important:** Treat this candidate as potentially overlapping with Candidate 2. Explicitly determine whether it is a standalone research problem, the theory component of Candidate 2, or already covered by existing active inference/data-fusion work. Do not artificially split a single contribution into two papers.

### Candidate 4 — Causal discovery under changing mechanisms and informative missingness

**Working title:** *Causal Discovery under Nonstationary Mechanisms and Informative Missingness*

**Core question:** Can evolving causal structures be identified or estimated when the observation/missingness process is informative, sampling is irregular, and intervention information is incomplete?

Potential components:
- changing causal mechanisms or graph structure;
- informative/MNAR missingness;
- irregular or asynchronous sampling;
- latent confounding;
- incomplete interventions;
- change-point or regime detection;
- identifiability and recovery conditions.

Potential method families to audit include nonstationary causal discovery, invariant causal prediction, causal discovery from subsampled time series, MNAR causal inference, latent-variable graphical models, intervention design, change-point detection, and causal representation learning.

**Important:** Combining nonstationarity, MNAR, and irregular sampling is not automatically novel. Find the exact closest joint formulation and determine whether a new identifiability result, estimator, or recovery condition is actually possible.

---

## 3. Subagent delegation plan

Create separate subagents/workstreams. Each must work from the original candidate brief and independently search primary literature. Require them to return source identifiers, exact mathematical comparisons, and uncertainty notes—not just summaries.

### Required workstreams

**Subagent A — LLM agent counterfactual evaluation**  
Own Candidate 1. Search sequential OPE, contextual bandits versus stateful trajectories, LLM-agent evaluation, partial observability, counterfactual evaluation, policy shift, and selective feedback. Identify exact estimands, overlap assumptions, variance issues, and theoretical gaps.

**Subagent B — Software breakage and experimental/observational data fusion**  
Own Candidate 2. Search PPI/data fusion/MNAR and software dependency breakage independently. Audit public datasets, reproducibility, selection mechanisms, CI replay validity, and whether the application can support a defensible statistical contribution.

**Subagent C — Active/selective inference and audit allocation**  
Own Candidate 3. Search active statistical inference, active learning with query costs, selective labels, delayed outcomes, policy shift, PPI, and MNAR data fusion. Determine whether Candidate 3 is subsumed by Candidate 2 or existing work.

**Subagent D — Causal discovery**  
Own Candidate 4. Search nonstationary causal discovery, MNAR causal discovery, irregular sampling, latent confounding, and identifiability theory. Assess proof difficulty and benchmark feasibility.

**Subagent E — Independent novelty auditor (adversarial)**  
Do not accept the other agents' novelty claims. Independently search for the closest papers across all four candidates, especially papers from 2025–2026 and work available before 5 November 2026. Find papers that may invalidate claimed gaps. Compare mathematical objectives and assumptions, not titles.

**Subagent F — Mathematical reviewer**  
Review proposed estimands, assumptions, estimators, objectives, theorems, and proof sketches. Reject decorative mathematics. Identify non-identifiability, invalid conditioning, positivity/overlap violations, selection bias, confounding, variance explosion, and impossible guarantees.

**Subagent G — Data/compute/feasibility reviewer**  
Assess dataset availability, annotation burden, access restrictions, reproducibility, licensing, compute needs, API spend, and whether 3–4 people can complete meaningful experiments in 5–6 months.

**Main agent — integrator and Reviewer 2**  
Compare all outputs, resolve contradictions, correct scoring arithmetic, merge overlapping candidates where justified, and make a conservative recommendation. The main agent must not simply average subagent scores.

If subagent count is limited, combine roles but preserve independent novelty auditing and mathematical review.

---

## 4. Literature-search and evidence protocol

### Time and coverage

- Research cutoff: **5 November 2026**.
- Prioritize 2024–2026 work, but include foundational papers where needed.
- Aim for **15–25 relevant papers per candidate** when the literature permits. Do not pad the list with weakly related papers just to reach a count.
- Search conference proceedings, journal publishers, preprints, technical reports, benchmarks, datasets, and relevant code repositories.
- Include accepted/in-press and arXiv work, but clearly distinguish them from peer-reviewed publications.

### Primary-source verification

Prioritize:
- ACL Anthology;
- OpenReview;
- PMLR / ICML;
- NeurIPS proceedings;
- ACM Digital Library / conference proceedings;
- IEEE Xplore;
- SpringerLink;
- ScienceDirect / Elsevier;
- Nature / relevant journal publisher pages;
- arXiv for preprints;
- official dataset and code repositories.

For each central competitor, verify the actual paper, not only a search-result snippet or secondary summary. Open the paper or primary abstract and inspect methodology, equations, experiments, and limitations. When a source is a PDF, inspect the relevant pages/tables/equations rather than relying only on extracted snippets.

### Required evidence ledger

For every central paper record:
- exact title and authors;
- year and date/version where relevant;
- venue and status (peer-reviewed, accepted, preprint, technical report);
- DOI, ACL Anthology ID, OpenReview ID, PMLR URL, arXiv ID, or official URL;
- exact problem and target estimand;
- assumptions and observation/data-generation process;
- mathematical objective or estimator;
- method and theoretical guarantee, if any;
- datasets and evaluation protocol;
- limitations explicitly stated by authors;
- why it is a direct, near, adjacent, or foundational competitor.

Cite each substantive literature claim with its primary source. Do not invent citations, DOIs, venues, publication dates, results, or limitations. If not verified, label it **unverified**.

### Search strategy

Use multiple query formulations per idea:
- terms describing the application;
- terms describing the mathematical problem;
- names of relevant estimator/algorithm families;
- phrases describing the data-generation or missingness mechanism;
- direct searches for likely competitors and their citing/cited work.

Backward and forward citation searching is required for the closest papers. Search 2026 work specifically; do not assume older review coverage is sufficient.

---

## 5. Required analysis for each candidate

### A. Formal reconstruction

Write a precise problem statement containing:
- units of analysis;
- population and sampling frame;
- observed variables and hidden/unobserved variables;
- data-generation and observation mechanisms;
- target estimand or prediction target;
- decision or intervention being evaluated;
- assumptions;
- what counts as success or failure.

Use a causal graph, data-flow diagram, state model, or mathematical notation where it clarifies the problem.

### B. Closest-paper matrix

Identify at least the five closest competitors and compare:
- same/different problem;
- same/different estimand;
- same/different assumptions;
- same/different objective;
- same/different estimator/algorithm;
- theoretical guarantees;
- data and evaluation;
- precise remaining difference.

Classify as:
1. **Direct competitor:** essentially the same problem and target objective.
2. **Near competitor:** same mathematical core with a different setting or one changed assumption.
3. **Adjacent competitor:** solves a component or related problem.
4. **Foundational:** supplies theory or tools used by the proposed work.

If an existing method can be applied to the proposed domain with only routine adaptation, say so. Do not call that a new methodology.

### C. Novelty boundary

State:
- what is already known;
- what is not yet solved (with evidence);
- the smallest defensible new claim;
- what would make that claim false;
- whether novelty is theoretical, methodological, empirical, dataset-based, or only application-based.

Do not claim that no prior work exists solely because no exact-title match was found.

### D. Mathematical formulation

Propose the minimum mathematics needed for a serious paper:
- notation and estimand;
- assumptions;
- objective or estimator;
- optimization or inference procedure;
- uncertainty quantification;
- at least one plausible proposition/theorem or rigorous analytical result, if appropriate;
- proof strategy and proof difficulty;
- counterexample or failure condition.

For statistical/causal candidates explicitly check:
- identification versus estimation;
- MAR/MNAR/MCAR or selection assumptions;
- positivity/overlap;
- consistency and interference;
- selection/measurement bias;
- finite-sample variance and effective sample size;
- sensitivity to misspecification;
- what can only be partially identified.

For sequential agent evaluation explicitly check:
- trajectory support and policy overlap;
- importance-weight variance;
- partial observability and state aliasing;
- action-dependent observation;
- reward/outcome delay;
- horizon effects;
- simulator/replay fidelity;
- whether counterfactual outcomes are actually observable.

For causal discovery explicitly check:
- graph identifiability;
- Markov, faithfulness, and invariance assumptions where used;
- latent confounding;
- changing graph versus changing mechanisms;
- informative observation/missingness;
- intervention coverage;
- non-identifiability counterexamples.

A theorem is not required for every empirical paper, but if the project claims mathematical novelty, the result must be substantive and correct. Never invent a guarantee just to improve the score.

### E. Method and benchmark design

Specify:
- proposed algorithm and why each component is needed;
- the strongest existing baselines;
- data splits that prevent leakage;
- synthetic experiments with known ground truth;
- real-world external validation;
- distribution-shift and stress tests;
- ablations and sensitivity studies;
- multiple seeds and confidence intervals;
- paired statistical tests where appropriate;
- error/failure analysis;
- artifact and reproducibility plan.

Separate confirmatory test data from development/tuning data. Do not use test labels to choose thresholds or methods.

### F. Dataset and ground truth

Determine whether a new dataset is necessary. Do not create a dataset merely to claim a contribution.

Assess:
- existing dataset suitability;
- sample size and independent units;
- label quality and adjudication;
- selection mechanism and missingness;
- licensing and redistribution;
- data leakage/near duplicates;
- reproducibility and archival stability;
- whether the benchmark measures the claimed construct.

If synthetic data are used, validate conclusions on real data when possible. Clearly distinguish simulated ground truth from real-world ground truth.

### G. Compute, budget, and schedule

Estimate actual requirements for:
- model inference and fine-tuning;
- API calls and prices (verify current rates if citing them);
- GPU hours and VRAM;
- CPU/storage needs;
- human annotation hours;
- data collection and cleaning;
- repeated seeds and ablations.

Give a low/base/high budget scenario. Do not assume LLM agents make annotation, proofs, source verification, and experiments costless.

Construct a realistic 5–6 month plan with month-by-month milestones, dependencies, and go/no-go gates. Identify the critical path and the earliest cheap falsification experiment.

### H. Journal strategy

Suggest suitable journal *categories* and a small number of journal candidates only after the contribution is defined. Verify current journal scope, article type, category-specific Q1 status (Scopus/SJR or JCR), fees, and policies. Quartile can differ by category and year. Never call a journal Q1 based only on reputation or an outdated ranking.

Do not select a journal to manufacture the research question. The contribution determines the venue.

---

## 6. Scoring framework — do not inflate

Score all eight parameters from 0 to 10. The weights sum to 100%.

| Parameter | Weight | What it measures |
|---|---:|---|
| 1. Research novelty | 25% | Distinctness from the closest verified work at the formulation/method level |
| 2. Methodological contribution | 20% | New reusable estimator, algorithm, optimization method, or framework |
| 3. Mathematical/theoretical depth | 15% | Substantive formulation, identification, proof, bound, convergence, or principled analysis |
| 4. Experimental contribution | 10% | Ability to produce decisive, generalizable, statistically credible evidence |
| 5. Dataset/benchmark contribution | 10% | Need, quality, reuse potential, and construct validity of the data/resource |
| 6. International journal suitability | 10% | Importance, generality, rigor, and fit for a strong journal article |
| 7. Feasibility in 5–6 months | 5% | Team/time critical path and probability of completing essential experiments |
| 8. Compute/budget affordability | 5% | Fit to RTX 3060 6GB, Colab GPUs, limited APIs, and budget |

Calculate:

`Weighted score = Σ(score_i × weight_i) / 100`

Use full precision internally, show the arithmetic, and round only the final score to one decimal place. Independently calculate the score a second time to catch arithmetic errors.

Also report:
- **Core scientific score:** weighted average of parameters 1–6, renormalized to 100% over those six weights.
- **Execution score:** parameters 7–8 separately; do not let affordability inflate novelty.
- **Novelty score:** always show independently.
- **Portfolio independence:** a separate 0–10 assessment of overlap with FYDP, HaluRISC, and IntentGate; do not include it in the weighted average.

Use conservative score bands:
- 0–3: weak or largely solved;
- 4–5: crowded, incremental, or poorly substantiated;
- 6–7: plausible research opportunity but significant novelty risk;
- 8: strong, defensible differentiation supported by a thorough competitor scan;
- 9–10: exceptional and rare; requires unusually compelling evidence.

**Do not award novelty above 7 until the closest-paper scan is complete and the exact novelty boundary survives adversarial review.** A weighted average above 8 is not proof of novelty. Do not force a candidate to score 8+.

---

## 7. Falsification-first pilot for every surviving idea

For each candidate propose one small, cheap, decisive pilot that could invalidate the central research claim.

For each pilot specify:
- hypothesis;
- minimum data and sample size;
- baseline;
- variables and outcome;
- analysis and statistical test;
- expected result if the hypothesis is true;
- result that would kill or substantially reframe the idea;
- time and cost to run.

Examples of the type of falsifier required (not predetermined conclusions):
- Does randomized audit data add information beyond the best audit-only/design-based/PPI estimator?
- Does a candidate estimator improve over existing sequential OPE methods under realistic policy overlap and horizon conditions?
- Does the proposed missingness/selection mechanism actually create measurable bias in a controlled simulation?
- Does the proposed causal structure become non-identifiable under plausible missingness or regime changes?

Do not select a pilot merely because it is easy to pass. It must be capable of disproving the idea.

---

## 8. Cross-candidate synthesis and selection

After all subagents report, the main agent must perform an adversarial synthesis.

1. Identify duplicate or nested candidates. In particular, assess whether Candidate 3 is the theoretical core of Candidate 2 rather than a separate project.
2. Compare all candidates at the level of estimands, assumptions, objectives, and algorithms.
3. Resolve conflicting paper facts by returning to primary sources.
4. Identify missing direct competitors found by the novelty auditor.
5. Recalculate all scores independently.
6. Do not automatically choose the highest weighted average.
7. Recommend no more than two ideas for a two-week falsification study.
8. State explicit rejection conditions for each recommendation.
9. If none has a defensible gap, say **NONE READY** and suggest which broader research areas deserve a new scan.

### Required final ranking table

| Candidate | Novelty | Method | Math | Experiments | Dataset | Journal fit | Feasibility | Budget | Core score | Weighted total | Portfolio independence | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

Add a separate explanation of the score for every criterion. Never present a single average without the underlying scores.

### Required final output structure

1. **Executive verdict:** best candidate(s), or none ready.
2. **Subagent coverage:** which workstreams ran and what they examined.
3. **Competitor evidence:** the closest papers and exact points of overlap.
4. **Merged/rejected ideas:** explain duplication and why candidates were removed or combined.
5. **Final candidate formulations:** precise research question, estimand, assumptions, method, expected contribution.
6. **Mathematical contribution:** exact proposed objective/estimator and plausible theorem or analytical result; distinguish established from conjectural.
7. **Scoring table:** all eight parameters, core score, weighted total, portfolio independence, arithmetic.
8. **Falsification pilots:** no more than two recommended projects, with stop/go rules.
9. **Dataset, compute, budget and six-month plan.**
10. **Journal categories and verified candidates**, with current category-specific quartile evidence and caveats.
11. **Reviewer 2 report:** strongest possible rejection arguments and what evidence would answer them.
12. **Research decision:** PROCEED / CONDITIONAL / REFRAME / REJECT for each candidate.

---

## 9. Output files

Create these files if your environment supports file creation:

- `research_audit/01_llm_agent_counterfactual_ope.md`
- `research_audit/02_mnar_software_dependency_data_fusion.md`
- `research_audit/03_audit_budgeted_selective_learning.md`
- `research_audit/04_causal_discovery_informative_missingness.md`
- `research_audit/05_adversarial_literature_and_math_review.md`
- `research_audit/FINAL_RESEARCH_SELECTION_REPORT.md`

Subagents should write their own reports with citations, and the main agent should write the final synthesis. If separate agent reports are impossible, create one Markdown report with clearly separated sections for each workstream.

Also include a machine-readable `research_audit/source_ledger.csv` or Markdown source ledger with exact URLs/identifiers, venue/status, and the candidate(s) each source supports.

Do not fabricate a file path or claim a file was created if the environment does not support it.

---

## 10. Final instruction

Be skeptical, precise, and willing to reject attractive ideas. The strongest project is not necessarily the most fashionable, the most LLM-centered, the most mathematically complicated, or the one with the highest preliminary average.

Select a project only when its **specific research gap, methodological contribution, mathematical foundation, ground truth, and decisive experiments** survive comparison against the strongest relevant literature available by 5 November 2026.
