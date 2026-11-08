# Green Computing — Shortlist and Recommendation

**Date:** 8 November 2026. **Theme:** sustainable AI / green computing in LLM and NLP.  
**Constraints:** 3–4 people, 4-5 months, RTX 3060 6 GB, 24 GB rental possible, ≤ $50 cash, public data, Q1 target.

## Bottom line

The green track has **two surviving 7.0 papers**, both **MARGINAL** after their pilots:

1. **Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation (G2; 7.00/7.35)** — the stronger quantitative result: a 5pp accuracy margin gives \(k=4\) with **69.3% energy savings**, and the optimal-depth identity holds at \(k=2\).
2. **Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies (G5; 7.00/7.03)** — a budget-indexed reversal and an identified first break-even, but the fixed-n power gate fails.

**Recommendation:** take **G2** as the green paper. It has the higher weighted score and a larger, decision-relevant effect. Use **G5** only if the team wants a more independent measurement paper or if G2’s 5pp-margin framing is rejected.

**No 7.5+ green idea remains.** The track is two honest 7.0 papers, not a 7.9 lead.

## Ranking

| Rank | ID | Title | Core | Weighted | Indep. | Status |
|---:|---|---|---:|---:|---:|---|
| 1 | G2 | **Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation** | 7.00 | 7.35 | 3.0 | **MARGINAL** — static 2pp claim killed; 5pp margin PASS at \(k=4\), 69.3% savings |
| 2 | G5 | **Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies** | 7.00 | 7.03 | 7.5 | **MARGINAL** — budget reversal present; break-even 256.7 J/correct; MDE 33% > 25% |

Scores are evidence-adjusted working estimates, **not acceptance probabilities**.

## Why G2 is the lead

- **Large effect:** \(E(32)/E(1)=3.45\); a 5pp accuracy margin at \(k=4\) saves **69.3%** of energy.
- **Theory check:** the optimal-depth identity \(kJ'/J=kA'/A\) holds at \(k=2\) (gap +0.044, CI includes zero) and fails at \(k=32\) (+0.700).
- **Decision-relevant:** the green lever is **prefill of retrieved passages** (85.1% of energy at \(k=32\)), not BM25 retrieval (3.5%).
- **Feasible:** 200 SQuAD questions; local `qwen3.5:4b`; no paid API.

**Caveat:** the static 2pp claim is dead. The paper must pre-specify a 5pp margin (or a larger \(n\)) and report the budget-indexed result, not a universal \(k^\star\).

## G5 fallback

- Pooled CPCA: Direct **21.6**, CoT-short **119.9**, CoT-long **554.8**, SC3 **1614.4** J/correct.
- Direct is budget-optimal only below **100.4 J/query**; the first break-even is **256.7 J/correct [192.8, 387.7]**.
- The fixed-n power gate fires (MDE 33% > 25%); certifying 15% differences needs ~387 queries/arm.
- No dominated CPCA runner-up was found.

## What to do next

**If green is the chosen course track:** write **G2** as a budget-constrained retrieval-depth paper. Pre-specify the accuracy margin, report the 5pp-margin result and the failed 2pp claim, and include the energy-split analysis. Do not claim a universal \(k^\star\).

**If G2 is rejected:** write **G5** as a carbon-effectiveness frontier paper with the budget-indexed reversal and break-even price. Report the power limitation openly.

## Files

- Research reports: `02_g2_green_rag.md`, `05_g5_cpca.md`
- Pilot reports: `pilots/02_g2_pilot.md`, `pilots/05_g5_pilot.md`
- Ledgers: `ledger_g2.csv`, `ledger_g5.csv`
- Plain-language pages:
  - `ideas/energy-optimal-retrieval-depth.md` — G2
  - `ideas/carbon-per-correct-answer.md` — G5
