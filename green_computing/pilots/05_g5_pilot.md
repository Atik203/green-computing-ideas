# G5 pilot - CPCA and the carbon-effectiveness frontier

**Date:** 8 November 2026  
**Status:** **MARGINAL** - the budget reversal is present and the first break-even is identified, but the pre-registered fixed-n power gate fires (K3, MDE 33% > 25%). The strong reversal (a dominated CPCA runner-up) was **not** found.  
**Model / hardware:** Ollama `qwen3.5:4b` (Q4_K_M, 4.2B, 100% GPU, batch 1, `num_ctx` 2048, `think=false`), RTX 3060 Laptop 6 GB, 32 GB RAM.  
**Files:** this memo | `ledger_g5_pilot.csv` | `queue/DONE_G5`. Raw per-query data and sampler trace: `C:\Users\atiku\AppData\Local\Temp\opencode\g5\run\`.

> **One line.** Pooled CPCA ranks Direct (21.6 J/correct) < CoT-short (119.9) < CoT-long (554.8) < SC3 (1614.4), but Direct is budget-optimal only below 100.4 J/query. At that budget CoT-short buys +35 accuracy points, and the break-even price is 256.7 J/correct [192.8, 387.7]. The decision reversal is real; the pilot cannot certify 15-20% CPCA differences (MDE 33%), and all four pooled points sit on the concave frontier, so the stronger "dominated runner-up" claim fails.

---

## 1. Design and measurement

- **Data:** 50 GSM8K test items + 30 ARC-Challenge test items, fixed seed 20261108; ARC stratified by answer letter (A/B/C/D = 8/8/7/7). Same items for every arm, so contrasts are paired. 80 items, 320 metered queries, **0 parse failures**.
- **Arms:** Direct (40 new tokens, greedy); CoT-short (max 128, prompt asks for <=3 short sentences and a final `The answer is ...` line); CoT-long (max 512); SC k=3 (three CoT-long samples, temperature 0.7, majority vote). Failed attempts are included in the numerator; parse failures count as incorrect.
- **Energy:** persistent `nvidia-smi -lms 100 --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu`; 34,302 samples, median spacing 109 ms. Each query window is integrated with the trapezoid rule, then idle-subtracted. Idle baseline = 14.643 W from the 58.4 s quiescent plateau inside the main trace (sample CV 5.7%). Boundary: GPU board power only; CPU, RAM, display, PSU and cooling excluded.
- **Calibration:** five identical CoT generations gave 38.4, 99.8, 99.0, 97.7, 99.8 J. The first repeat after idle is a cold outlier; full CV = **31.2%**, steady-state CV (last four) = **1.0%**.
- **Replication:** 20 GSM8K items, second seed, Direct and CoT-long. Gross within-item CV = 7.6% (Direct) / 14.7% (CoT-long); accuracy 20% / 90% vs 22% / 86% in the main run. Ordering stable (K7 pass).
- **Run totals:** 60.4 min, gross 219.2 kJ (60.9 Wh), net 166.1 kJ (46.1 Wh); gross carbon 24.4 gCO2e at 400 g/kWh (18.5 g net); harness (load + warmup + calibration) 0.43 kJ. SC3 accounts for 71% of arm energy.

## 2. Pooled results

| Arm | Accuracy [95% CI] | Net J/query | CPCA J/correct [95% CI] | E+ J | E- J | Failure gap J/correct | Tokens/q | Query CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 0.4875 [0.375, 0.600] | 10.54 | **21.6** [17.3, 28.2] | 9.9 | 11.1 | 11.7 | 3.0 | 0.159 |
| CoT-short (128) | 0.8375 [0.750, 0.912] | 100.39 | **119.9** [105.0, 138.8] | 95.5 | 125.7 | 24.4 | 71.5 | 0.330 |
| CoT-long (512) | 0.8875 [0.812, 0.950] | 492.39 | **554.8** [494.4, 630.0] | 473.1 | 644.7 | 81.7 | 336.8 | 0.313 |
| SC k=3 | 0.9125 [0.850, 0.963] | 1473.14 | **1614.4** [1463.6, 1798.9] | 1429.4 | 1928.9 | 185.0 | 1026.9 | 0.260 |

All paired CPCA contrasts are negative with 95% bootstrap CIs excluding zero; bootstrap P(first arm lower CPCA) = 1.00 for every pair. The CPCA order and the success-conditioned E+ order are identical, so **no success-conditioned rank reversal** at this n.

**Failure-inclusion bias is large where it matters.** Direct's success-conditioned E+ = 9.9 J understates its CPCA of 21.6 J by **54.1%**; CoT-short 20.4%, CoT-long 14.7%, SC3 11.5%. Direct's failures cost as much as its successes (11.1 vs 9.9 J) and it fails half the time, so dropping failures would make the cheap strategy look roughly twice as green as it is. The direction confirms Proposition 3; a rank reversal would need a larger accuracy gap or cheaper failures.

## 3. Frontier and the decision reversal

Pooled coordinates (mu, A): Direct (10.54, 0.4875), CoT-short (100.39, 0.8375), CoT-long (492.39, 0.8875), SC3 (1473.14, 0.9125). Frontier slopes fall monotonically (0.00390 > 0.000128 > 0.0000255), so **all four points lie on the concave upper envelope** at the point estimates. Bootstrap frontier probabilities are 1.00, 1.00, 0.77, 0.69; no arm is dominated at the point estimate and **no dominated CPCA runner-up** was found (the strong reversal is absent).

**Budget staircase (D2):**

| Per-query budget B (J) | Best strategy | Accuracy |
|---|---|---:|
| < 10.54 | none feasible | - |
| 10.54 <= B < 100.39 | Direct | 0.4875 |
| 100.39 <= B < 492.39 | CoT-short | 0.8375 |
| 492.39 <= B < 1473.14 | CoT-long | 0.8875 |
| >= 1473.14 | SC3 | 0.9125 |

**Reversal.** The CPCA-min (Direct) is budget-optimal only for B < 100.4 J/query. At B = 100.4, CoT-short delivers 83.8% versus 48.8% accuracy (+35.0 pp) for +89.8 J/query. Under an accuracy floor (D1), Direct is infeasible at any A_min >= 0.5 (CoT-short wins), and SC3 wins at A_min >= 0.9. This satisfies the pass condition "best-CPCA strategy is not budget-optimal", but only in the weak, budget-indexed form.

**Break-even carbon prices (D3):**

| Transition | lambda* (J/correct) | 95% CI | Verdict |
|---|---:|---|---|
| Direct -> CoT-short | **256.7** | [192.8, 387.7] | identified; CI width 76% of point |
| CoT-short -> CoT-long | 7840 | [-31,361, 31,958] | **not identified** (5.0 pp for 392 J) |
| CoT-long -> SC3 | 39,230 | [-80,343, 80,930] | **not identified** (2.5 pp for 981 J) |

CEAC confirms only the first step: at lambda = 192.8, Direct is optimal in 97.5% of bootstrap replicates; at 256.7 it is 49.7% Direct / 50.3% CoT-short; at 387.7, CoT-short is 97.7%; at 7840, CoT-short 50.5% / CoT-long 49.5%; at 39,230, CoT-long 47% / SC3 45%. Above lambda ~ 390 J/correct the frontier is not decision-certified.

**Benchmark heterogeneity (diagnostic, not pre-registered).**

| Benchmark | Direct | CoT-short | CoT-long | SC3 |
|---|---:|---:|---:|---:|
| GSM8K CPCA (J/correct) | 50.5 (A=0.22) | 141.5 (0.76) | 560.2 (0.86) | 1546.2 (0.92) |
| ARC CPCA (J/correct) | 10.3 (A=0.933) | 91.5 (0.967) | 546.5 (0.933) | 1730.6 (0.90) |

On GSM8K the pooled reversal is strong. On ARC, Direct is already at 93.3%, CoT-short adds only 3.3 pp, and CoT-long and SC3 are strictly dominated (lower accuracy, much higher energy). A post-hoc router - Direct on ARC, CoT-short on GSM8K - achieves 70.8 J/query and 82.5% accuracy (CPCA 85.9), beating the pooled Direct-CoT-short chord at that energy (72.3%). The pooled frontier therefore understates what query-type routing can do; a single pooled strategy is not the deployable object.

## 4. Power, kill criteria, verdict

**Power.** Primary contrast Direct vs CoT-long: sigma_Delta = 303.5 J/correct, MDE at n = 80 = 95.1 J/correct = **33.0%** of the pair mean; K3 fires (>25%). Direct vs CoT-short MDE 31.3%; CoT-short vs CoT-long 26.2%; CoT-long vs SC3 18.0%. Required paired n per arm: **387** for a 15% MDE and **871** for 10% (Direct vs CoT-long); 348/782 for Direct vs CoT-short. The observed effects are 98-185% of pair means, so the ordering is certified even though small differences are not.

**Kill/pass scorecard.**

| Gate | Result |
|---|---|
| K1 energy CV <= 0.35 | **Pass with caveat**: full calibration 31.2%, steady 1.0%, replication 7.6/14.7%, per-query 15.9-33.0%. Direct is 0.2 s/query, near sampler resolution; doubling its energy would not change the ranking. |
| K2 accuracy non-degenerate | Pass: pooled 48.8-91.3% |
| K3 MDE <= 25% | **Fail: 33.0%** (fixed-n pilot underpowered) |
| K4 decision change | **Pass (b)**: budget reversal at B >= 100.4 J/query; no dominated runner-up (a) |
| K5 trade-off present | Pass: SC3 uses 140x Direct energy for +42.5 pp |
| K6 instrumentation overhead | Pass: sampler adds negligible wall time; harness 0.43 kJ |
| K7 seed stability | Pass: 20-item replication keeps the Direct/CoT-long ordering |

**Verdict: MARGINAL.** Under the letter of the pre-registered gate, K3 kills the fixed-n pilot. The repair hypothesis nevertheless survives conditionally: the large-effect reversal is certified, the Direct->CoT-short break-even is positive and finite, and the failure-inclusion bias is large for the CPCA-min. But the strong reversal (dominated CPCA runner-up) is absent, the upper frontier steps are not identified, and the break-even CI width (76%) misses the 30% GO target. This is not enough for a six-month study on its own.

**Path.** (1) Full study with 300-500 paired items/arm and a second model (Qwen2.5-1.5B/3B) plus a second benchmark; target MDE <= 15% and a Direct->CoT-short lambda* CI width < 30%. (2) Pre-register the benchmark router as a deployment arm; report frontier probabilities, not just point ranks. (3) If the reversal survives 15% MDE with a tight lambda*, proceed to the 2 models x 3 benchmarks x 6 arms design (6-8 weeks), then analysis (4 weeks) and writing (6 weeks); target ACM JCSS or IEEE T-SUSC, NeurIPS D&B as fallback. If the reversal disappears under a second model or on ARC-only, downgrade R1 to a measurement note.

**Measurement caveats.** One seed per item for the main grid (20-item replication only); Direct queries are short relative to the 100 ms sampler; single 3060 laptop and one 4B Q4 model; prompts are a design choice and CoT-short includes an explicit brevity instruction; carbon is a constant rescaling (400 g/kWh mid; 100/700 sensitivity), so no carbon-aware scheduling claim is made.
