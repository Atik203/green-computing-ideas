# Team Idea Validation Prompt

**How to use:** paste everything below into Claude/GPT with web access. Replace the idea with your candidate. Ask for one markdown report + one CSV ledger, not a chat-only answer.

---

You are a senior ML/NLP researcher and adversarial reviewer. Your job is to **kill or keep** a research idea under a hard budget. Do not inflate scores. A false 8+ is worse than an honest 6.

## 1. Constraints (non-negotiable)

- Cash: ≤ **$50 total**. No paid API, human annotation, cloud A100, or APC.
- Compute: local **RTX 3060 6 GB**, free Colab/CPU.
- Models: GPT-2, Pythia-70M–1B, Qwen2.5-0.5B/1.5B, Llama-3.2-1B, Qwen3.5-4B (Ollama, 3.3 GB).
- Data: public or synthetic only.
- Team: 3–4 people; **6–7 months**.
- Target: Q1 journal (TACL, CL, JMLR, Machine Learning, Neural Networks, Biometrics, JRSS-B, IEEE TKDE/TIFS). Verify quartile and APC.
- Do not duplicate **HaluRISC** (hallucination risk), **IntentGate** (intent/agent authorization), or **FYDP** (trust-calibrated multi-agent deliberation).

## 2. What counts as a contribution

One of:

- a new **estimand** (target not previously defined or identified);
- a new **identification result**;
- a new **theorem, minimax bound, or lower bound**;
- a new **algorithm with a guarantee**;
- a **robust empirical law** that predicts behavior across models/tasks.

Not enough: new application, new dataset with no question, combination of known methods, new benchmark, framework/protocol.

## 3. The four survivor properties

| Property | Test | Kill signal |
|---|---|---|
| **New functional** | Can you write the target as a minimax/identification quantity? | It is an ATE or a standard sample-size formula |
| **Non-saturated model** | Does the model impose a restriction that yields a prediction? | It can fit any distribution |
| **Non-identity theorem** | A sign, bound, impossibility, or rate | It restates definitions |
| **Decision-changing test** | The experiment flips a published conclusion | It only reports a number |

## 4. Scoring (0–10)

| # | Parameter | Weight |
|---:|---|---:|
| 1 | Novelty | 20% |
| 2 | Method | 15% |
| 3 | Math | 15% |
| 4 | Experiments | 10% |
| 5 | Dataset | 5% |
| 6 | Q1 fit | 10% |
| 7 | Feasibility | 10% |
| 8 | Compute fit | 5% |
| 9 | API/cash | 5% |
| 10 | Independence | 5% |

Core = mean 1–6. Weighted = sum. Report both the **scan score** and the **audit-adjusted score**. **8+ requires novelty ≥8, method ≥8, math ≥8, and a verified theorem/dataset/algorithm.**

## 5. Mandatory adversarial audits

1. **Closest-paper check** at the **functional** level, not the topic. Exact arXiv IDs/DOIs. Label `verified`, `abstract-only`, or `unverified`.
2. **Theorem audit.** Prove or disprove; find a counterexample; check quantifiers, support, determinism, containment.
3. **Estimand audit.** Define the distribution, equivalence relation, label, and monitor/estimator. Is it identified? Is it circular?
4. **Experiment audit.** Power/MDE, seeds, accuracy gate, confounds.
5. **Dataset audit.** Public labels? reusable? or a generator?
6. **Collinearity/leakage audit.** If the new metric is >0.8 correlated with the old one, or the feature leaks the answer, kill it.

## 6. Failure modes that kill

- **New object + known estimator.** New dataset + paired t-test; new task + standard identification.
- **Theorem false as stated.** One counterexample is enough.
- **Model containment/saturation.** If the new model contains the old one, no strict tightening; if it can fit anything, no prediction.
- **Estimand undefined.** No pair distribution, equivalence, label, or monitor class.
- **Circularity.** The label is “whatever the model does.”
- **Underpowered experiments.** Compute the MDE before promising a result.
- **Synthetic-only dataset.** A generator is not a dataset unless it enables a claim no natural benchmark can test.
- **Collinearity.** 0.88 Spearman with the old metric is not a new construct.
- **Crowded application.** “Another benchmark” or “another LLM application” is not a Q1 method.

## 7. Test three 8+ contribution types

- **T1 — Identification.** A target not identified under the standard design becomes identified under a new, credible, testable assumption. Needs a theorem, a counterexample, and a decision change.
- **T2 — Impossibility/lower bound + matching algorithm.** A non-vacuous bound and an algorithm that attains it. Check whether it is a corollary of adaptive data analysis, Le Cam, or VC theory.
- **T3 — Controlled dataset/intervention.** A factor randomized while all else is fixed, enabling a claim no existing dataset can. Check pre-emption; a generator is not enough.

Most ideas fail all three. If none lifts the idea to 8+, say so.

## 8. Output

One markdown report:

1. **Verdict** and score (core/weighted).
2. **Exact idea**: estimand, assumptions, theorem/algorithm/dataset.
3. **Closest 5 papers** with IDs and verification status.
4. **Theorem or counterexample**, with proof sketch.
5. **Experiment plan**: data, models, GPU-hours, cash, power/MDE.
6. **2–3 day kill check**: one cheap decisive test, with explicit kill criteria.
7. **T1/T2/T3 verdict**: does any lift it to 8+? If not, why.
8. **Final recommendation**: KILL / CONDITIONAL / GO.

Also write a CSV ledger:
`idea,title,authors_or_source,year,venue_status,identifier,url,problem,estimand,method,guarantee,verified,notes`

**Final chat ≤300 words:** verdict, closest paper, the exact 8+ path if one exists, score, report path.

## 9. Do not re-propose without a new functional

These areas are exhausted from a previous search:

- hallucination detection / calibration tax / conformal abstention;
- tokenization / merge-level Shapley / witness coverage;
- chain-of-thought trace identifiability / communication-complexity bounds;
- unlearning audit / network sensitivity / multilingual safety;
- ICL example influence / PT→IT model-merging frontier.

A new dataset or application in these areas is not enough. Bring a new functional, a new intervention, or a new theorem.
