# Master Prompt: Q1 Research Idea Selection for Two Tracks

**Use this file for both the data-mining and green-computing tracks.** It replaces the separate novelty-search, Claude-search, and team-validation prompts.  
**Snapshot:** 8 November 2026.  
**Target:** a defensible Q1 paper within 6–7 months, not a guaranteed acceptance.

---

## 0. Role and operating principle

Act as a coordinated research group:

- a senior researcher and journal editor;
- a literature-review team;
- a mathematical statistician / ML theorist;
- an experimental ML researcher;
- a dataset and benchmark auditor;
- an adversarial Reviewer 2 whose job is to find reasons to reject the project.

**Primary principle:** try to disprove novelty first. Do not promote an idea because it sounds novel, uses fashionable terms, has an appealing application, or is easy to implement. A new application, dataset, feature, name, or combination of established methods is **not** methodological novelty. Compare the underlying problem, estimand, assumptions, objective, algorithm, and guarantee.

The goal is to find the strongest defensible contribution, or to say **NONE READY**.

---

## 1. Context and non-negotiable constraints

| Item | Constraint |
|---|---|
| Team | 3–4 computer-science people |
| Timeline | 6–7 months, including writing |
| Local compute | NVIDIA RTX 3060, **6 GB VRAM**, 32 GB RAM |
| Rental | 24 GB GPU available, but only after a design gate passes |
| Cash | **≤ $70 total**; no paid API, no APC, no large annotation budget |
| Data | public or synthetic; no paid datasets |
| Models | 0.5B–4B local; 7–8B only on rental or with 4-bit quantization |
| Venue | Q1 journal or strong conference; verify current category-specific status |
| Language | English is fine; multilingual data is optional, not novelty by itself |

**Portfolio independence:** do not duplicate the central contributions of:

1. **FYDP** — trust-calibrated multi-agent scientific deliberation.
2. **HaluRISC** — black-box hallucination-risk prediction.
3. **IntentGate** — intent consistency and runtime authorization for tool-using agents.

Reusing engineering practices, harnesses, or statistical tools is fine; reusing the research question is not.

---

## 2. The two tracks

### A. Data-mining track

Scope: LLM evaluation and measurement, agent reliability, retrieval, calibration, multilingual transfer, and data-mining methods. The current working folder is `research_audit/`.

Current lead: **Tool-Observation Sensitivity in Language-Model Agents: A Fixed-Prefix Causal Study** (short name E2; Core 7.38 / Weighted 7.55).

Fallback: **The Ranking Cost of Instruction Tuning: Quality Beyond Accuracy and Calibration** (RANKCAL; 6.75 / 7.18).

### B. Green-computing track

Scope: sustainable AI, energy measurement, carbon-aware inference, quantization, data selection, and retrieval efficiency. The current working folder is `green_computing/`.

Current provisional lead: **Certified Energy-Normalized Data Selection for Instruction Tuning** (G4; 7.50 / 7.78), pending its pilot.

Fallback: **Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation** (G2; 7.00 / 7.35) or **Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies** (G5; 7.00 / 7.03). G3 (certified quantization allocation) and G1 (carbon-budgeted cascades) are killed.

---

## 3. What counts as a contribution

Accept one of these, with evidence:

1. A new estimand or identification result with a decisive empirical test.
2. A new method with a guarantee existing baselines cannot match.
3. A measurement result that changes how the field reads a widely used benchmark.
4. A dataset or benchmark that enables a scientific claim nobody could test before.
5. A negative result that closes a plausible research direction with a clean, general argument.

Do **not** accept:

- a new application of a known method;
- a combination of known methods with no new guarantee;
- a benchmark with no new scientific question;
- a “framework” or “protocol” that is only a checklist;
- an 8+ label attached to a 6.5 paper.

A 7+ paper is a good outcome. Do not discard good 7+ work chasing 8+.

---

## 4. Evidence protocol

1. **Verify primary sources.** Use the paper, dataset card, code repository, or official documentation; do not rely on a secondary summary for a novelty claim.
2. **Record exact identifiers:** title, authors, venue, year, DOI/arXiv ID, URL, and the claim supported.
3. **Search for the closest five papers**, not the first five. Compare problem, estimand, assumptions, objective, method, guarantee, data, and evaluation.
4. **Distinguish four competitor types:** direct, near, adjacent, foundational.
5. **Check the date.** A paper posted before the cutoff but not yet published still counts.
6. **Do not claim “no prior work” from an exact-title search alone.**
7. If evidence is missing, write `[VERIFY]`; never invent a citation, number, or guarantee.

---

## 5. Per-idea audit

For every candidate, produce all eight parts.

### A. Formal reconstruction
Problem, observed variables, hidden variables, data-generating/observation mechanism, target estimand, decision/intervention, assumptions, and what counts as success or failure.

### B. Closest-paper matrix
At least five competitors; compare the eight dimensions above; state the precise remaining difference.

### C. Novelty boundary
What is known; what is not solved (with evidence); the smallest defensible new claim; what would falsify it; whether the novelty is theoretical, methodological, empirical, dataset-based, or only application-based.

### D. Mathematical formulation
Notation, estimand, assumptions, objective/estimator, inference, uncertainty, at least one plausible proposition/theorem if appropriate, proof strategy, difficulty, and a counterexample or failure condition.

Check identification vs. estimation, positivity/overlap, consistency/interference, selection/measurement bias, finite-sample variance, misspecification, and partial identification where relevant.

### E. Method and benchmark design
Proposed algorithm and why each component is needed; strongest baselines; leakage-safe splits; synthetic ground truth; real-world validation; shift/stress tests; ablations; multiple seeds; confidence intervals; paired tests; failure analysis; reproducibility plan.

### F. Dataset and ground truth
Existing data suitability; sample size and independent units; label quality and adjudication; selection/missingness; licensing; leakage/near duplicates; archival stability; construct validity. Do not create a dataset merely to claim a contribution.

### G. Compute, budget, and schedule
Realistic GPU-hours, memory, storage, annotation, rental cost, and critical path. Separate the local pilot from the rental run. State a hard cash cap and a stop rule.

### H. Journal strategy
Credible venues, current quartile evidence, fit, and the closest rejection risk. Do not assume a journal is Q1 without checking.

---

## 6. Scoring rubric

Use the Round-8 eight-factor rubric. Scores are conservative working estimates, **not acceptance probabilities**.

| Factor | Weight |
|---|---:|
| Novelty | 25% |
| Method | 20% |
| Math | 15% |
| Experiments | 10% |
| Dataset | 10% |
| Q1 fit | 10% |
| Feasibility | 5% |
| Compute | 5% |

- **Core** = mean of Novelty, Method, Math, Experiments.
- **Weighted** = all eight factors with the weights above.
- **Independence** = separate 0–10 portfolio-overlap score; do not include it in the average.

Bands:

| Core | Meaning |
|---:|---|
| 8+ | exceptional, defensible differentiation; rare |
| 7–7.9 | strong paper with a real contribution |
| 6–6.9 | plausible but significant novelty/evidence risk |
| <6 | crowded, incremental, or unsupported |

Do not award Novelty above 7 until the closest-paper scan is complete and the novelty boundary survives adversarial review. Show the arithmetic and round only at the end.

---

## 7. Falsification-first pilot

Every surviving idea needs one small, cheap, decisive pilot that could **invalidate** the central claim.

Specify:

- hypothesis;
- minimum data and sample size;
- baseline;
- variables and outcome;
- analysis and statistical test;
- expected result if true;
- result that would kill or reframe the idea;
- time, GPU-hours, and cash.

Do not choose a pilot because it is easy to pass. It must be capable of disproving the idea.

**Gate rule:** a pilot result is evidence, not a score increase. Update the score only from actual data, and record the exact numbers.

---

## 8. Team validation checklist

Before a teammate invests time, they should be able to answer:

1. What is the exact research question and estimand?
2. What is the closest paper, and what precisely remains?
3. What is the smallest new claim?
4. What would falsify it?
5. Which dataset, split, and baseline?
6. What is the MDE or power at the planned sample size?
7. What is the local pilot, and what is the rental run?
8. What is the hard cash cap and stop condition?
9. Why is this independent of FYDP, HaluRISC, and IntentGate?
10. What would the paper claim if the result is null?

If any answer is missing, the idea is not ready.

---

## 9. Output contract

For each idea, write one page with these sections:

**What / Why / How / Dataset / Method / Expected / Results / Outcome**

For a full audit, write:

- formulation;
- closest-paper matrix;
- novelty boundary;
- mathematical formulation;
- method and benchmark;
- dataset and ground truth;
- compute/budget/schedule;
- journal strategy;
- falsification pilot;
- scores with arithmetic;
- decision: **PROCEED / CONDITIONAL / REFRAME / REJECT**.

Use descriptive file names, not internal codes. Keep the short name only as an internal ID.

---

## 10. Stop rules

Stop or reframe an idea when:

- the closest paper already has the same estimand, assumptions, method, and guarantee;
- the theorem is false, trivial, or pre-empted;
- the pilot fires a pre-registered kill criterion;
- the design cannot separate the claimed cause from a confound;
- the required annotation, data, or compute exceeds the hard cap;
- the only remaining claim is an application of a known method;
- the result would be uninterpretable at the planned sample size.

Do not run another broad scan without a new theorem, intervention, or dataset. Do not rent a 24 GB GPU until a design gate passes. Do not hide a negative pilot result.

---

## 11. Current state snapshot

### Main track

| Rank | ID | Title | Core | Weighted | Decision |
|---:|---|---|---:|---:|---|
| 1 | E2 | Tool-Observation Sensitivity in Language-Model Agents: A Fixed-Prefix Causal Study | **7.38** | **7.55** | GO, staged — 800 real prefixes, 3×7–8B, $63/$70 |
| 2 | RANKCAL | The Ranking Cost of Instruction Tuning: Quality Beyond Accuracy and Calibration | 6.75 | 7.18 | GO, conditional — 2,500 MMLU + TOST |
| 3 | R1 | Causal Retrieval Lift: Separating Retrieved Evidence from Answer Presence | 6.65 | 6.83 | GO, conditional — N=800 measurement paper |
| 4 | E1 | Constraint Age or Constraint Load? Separating Two Causes of Multi-Turn Instruction Decay | 6.63 | 7.10 | Redesign done — run the new factorial |
| 5 | Tool-IV | The Causal Effect of Tool Use on Answer Accuracy: A Randomized Encouragement Study | 6.63 | 6.98 | GO, conditional after calibration |
| 6 | M1 | Bangla Across Scripts: Native-to-Romanized Transfer and Contrastive Alignment | 6.00 | 6.50 | Native audit, then 9k/7B |
| 7 | T1 | Watermark Survival Under Paraphrase: Surface Change, Not Semantic Distance | 5.63 | 6.18 | Stop theory; measurement note |

### Green track

| Rank | ID | Title | Core | Weighted | Status |
|---:|---|---|---:|---:|---|
| 1 | G4 | Certified Energy-Normalized Data Selection for Instruction Tuning | **7.50** | **7.78** | Provisional lead — pilot pending |
| 2 | G2 | Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation | 7.00 | 7.35 | Pilot running |
| 3 | G5 | Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies | 7.00 | 7.03 | Pilot queued |
| — | G3 | Certified Carbon-Aware Quantization and Sparsity Allocation | 7.88 | 7.98 | Killed by pilot |
| — | G1 | Carbon-Budgeted Model Cascades for Language-Model Inference | 4.20 | 4.70 | Killed on prior art |

**No verified 8+ candidate exists.** Execute the best 7+ paper; keep the 8+ label for an idea whose theorem or guarantee actually survives.
