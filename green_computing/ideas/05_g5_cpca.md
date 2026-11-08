# G5 — Carbon per Correct Answer (CPCA) Benchmark

**One-line:** Measure joules per correct answer for reasoning strategies — direct answer, CoT, self-consistency, debate — and report the carbon-effectiveness frontier.  
**Score:** Core **7.00** / Weighted **7.03** (repaired G5′); original **4.00 / 4.63 — KILLED**.  
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

GSM8K, ARC, MMLU; Qwen2.5-1.5B; `nvidia-smi` at 100 ms plus idle subtraction; fixed regional carbon intensity.

## Method

CPCA estimator; failure-inclusion bias \(R=E^+ + ((1-A)/A)E^-\); paired bootstrap/influence-function CIs; frontier and break-even price; rank-reversal test.

## Expected

A positive result: a frontier or budget reversal — the strategy with the best CPCA is not the strategy that is optimal at a tighter carbon budget. If all strategies rank the same, the benchmark is incremental.

## Results

**Original killed.** **Token Arena (`2605.00300`)** already defines the exact estimand \(J_{CA}=j_eT_e/A_e\); **Yang & Armour (`2610.14389`)** measure it across 86 checkpoints, 7 benchmarks, and reasoning modes; Mamun et al., Jin et al., and Kashaniyan & Jannesari already compare strategies. A fifth leaderboard is incremental; in a fixed region carbon is energy × a constant.

**Repair G5′:** CPCA-CEA — carbon-effectiveness frontier, break-even price, failure-inclusive estimation, paired CIs. Non-identity result: CPCA rank ≠ carbon-effectiveness rank; the #2 CPCA strategy can be dominated by fleet routing, while the #3 is optimal above 750 J/correct. **7.00 / 7.03.**

## Outcome

**Low priority.** Run only if the pilot shows a frontier or budget reversal. Otherwise keep the estimator as a section in G3/G2, not a standalone paper.
