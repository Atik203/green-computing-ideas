# Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies

**Internal short name:** G5

**One-line:** Measure joules per correct answer for reasoning strategies — direct answer, CoT, self-consistency, debate — and report the carbon-effectiveness frontier.  
**Score:** Core **7.00** / Weighted **7.03** (repaired G5′) — **MARGINAL**; original **4.00 / 4.63 — KILLED**.  
**Independence:** 7.5/10.

## What

Reasoning strategies improve accuracy but use more tokens and energy. A strategy that is 2 points more accurate but 3× more energy-intensive may be worse per correct answer. The question is which strategy is greenest per correct answer, and when the ranking changes with the carbon budget.

## Why

Most green-AI papers measure training energy or single-call inference. They do not measure the carbon cost of reasoning strategies per correct answer, including retries and failures. A deployment team needs to know whether “think longer” is worth the carbon.

## How

1. Run direct, CoT-short, CoT-long, and self-consistency (k=3) on the same questions.
2. Measure joules per query and accuracy.
3. Compute \(\text{CPCA}=J/\text{correct}\), with failure-inclusive estimation.
4. Report the carbon-effectiveness frontier and the break-even price \(\lambda^\star=\Delta\mu/\Delta A\).

## Dataset

50 GSM8K + 30 ARC-Challenge questions; Ollama `qwen3.5:4b` Q4; `nvidia-smi` at 100 ms plus idle subtraction; fixed regional carbon intensity.

## Method

CPCA estimator; failure-inclusion bias \(R=E^+ + ((1-A)/A)E^-\); paired bootstrap/influence-function CIs; frontier and break-even price; rank-reversal test.

## Expected

A positive result: a frontier or budget reversal — the strategy with the best CPCA is not the strategy that is optimal at a tighter carbon budget. If all strategies rank the same, the benchmark is incremental.

## Results

**Pilot: MARGINAL.** 80 questions × 4 arms, 320 metered queries, 0 parse failures. Pooled CPCA:

| Strategy | Accuracy | CPCA (J/correct) |
|---|---:|---:|
| Direct | 0.488 | **21.6** [17.3, 28.2] |
| CoT-short | 0.838 | **119.9** [105.0, 138.8] |
| CoT-long | 0.888 | **554.8** [494.4, 630.0] |
| SC3 | 0.913 | **1614.4** [1463.6, 1798.9] |

Direct is budget-optimal only below **100.4 J/query**. At that budget, CoT-short buys **+35.0 accuracy points**; the break-even price is **256.7 J/correct [192.8, 387.7]**. The decision reversal is real, but only in the weak, budget-indexed form: all four pooled points lie on the concave frontier, and no dominated CPCA runner-up was found.

The pre-registered fixed-n power gate fires: **MDE 33% > 25%**. Certifying 15% CPCA differences needs ~387 queries per arm. Failure-inclusion is large where it matters: Direct’s success-conditioned E+ (9.9 J) understates its CPCA (21.6 J) by 54.1%. Calibration CV is 31.2% full / 1.0% steady; replication CV is 7.6% Direct / 14.7% CoT-long.

**Original killed.** **Token Arena (`2605.00300`)** already defines \(J_{CA}=j_eT_e/A_e\); **Yang & Armour (`2610.14389`)** measure it across 86 checkpoints; Mamun et al., Jin et al., and Kashaniyan & Jannesari already compare strategies. A fifth leaderboard is incremental.

**Repair G5′:** CPCA-CEA — carbon-effectiveness frontier, break-even price, failure-inclusive estimation, paired CIs. **7.00 / 7.03.**

## Outcome

**MARGINAL — a 7.0 paper, not a standalone benchmark.** Report the budget-indexed reversal and the first break-even price. Do not claim a dominated CPCA runner-up. To certify 15% CPCA differences, run ~387 queries per arm; otherwise keep the estimator as a section in G2/G3.
