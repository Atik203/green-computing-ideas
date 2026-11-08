# Carbon-Budgeted Model Cascades for Language-Model Inference

**Internal short name:** G1

**One-line:** Route each query to a small, medium, or large model under a hard carbon budget and a latency SLA.  
**Score:** Core **4.20** / Weighted **4.70** — **KILLED**.  
**Independence:** 3/10 (prior art overlap).

## What

A serving system can answer with a local 0.5B model, a 1.5B model, or a large/remote model. The large model is more accurate but uses more energy. The question is which model to use for each query when the total carbon per request must stay under a budget.

## Why

Most routing work optimizes money or latency. Carbon is usually treated as a price, not a hard constraint. A carbon budget changes the decision: a query may be routed to the small model even when the large one is more accurate, because the budget is binding.

## How

1. Measure energy per model on a fixed workload.
2. Estimate each model’s quality on the query distribution.
3. Learn a threshold policy in a quality-per-joule score.
4. Calibrate the threshold on held-out queries so the budget is respected with high probability.

## Dataset

MMLU, ARC, GSM8K; measured energy on the local 3060 and a 24GB rental. Public carbon-intensity data for the region.

## Method

Carbon-aware cascade; threshold policy; conformal or empirical-Bernstein budget guarantee.

## Expected

A positive result would show the threshold policy beats a static “always small” or “always large” baseline at the same carbon budget, with a valid finite-sample guarantee.

## Results

**Killed.** The exact estimand already exists in **GAR, `2605.11603`** (minimize CO₂ subject to accuracy, p95 latency, and a rolling carbon budget). The finite-sample guarantee is owned by **C3PO, `2511.07396`** (conformal bound on cascade cost exceeding budget). Carbon is a multiplier on cost; composing the two is a unit change, not a new theorem. The repair “settle what you schedule” is a competent applied paper at **5.4 / 5.8**, independence 3/10.

## Outcome

**Stop.** Reallocate to G3′ or G2. Do not write this as a carbon-budget paper.
