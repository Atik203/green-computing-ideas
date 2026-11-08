# G2 — Energy-Optimal Retrieval Depth (Green RAG)

**One-line:** Find the retrieval depth \(k^\star\) that minimizes joules per correct answer, not just latency or tokens.  
**Score:** Core **7.00** / Weighted **7.35** — **CONDITIONAL**.  
**Independence:** 3/10 (overlaps G5 and prior RAG-efficiency work).

## What

Retrieval costs energy, and generation costs energy. Retrieving more passages can reduce generation length and errors, but it also adds context energy. The question is: what is the energy-optimal number of passages \(k^\star\) per query?

## Why

RAG systems are usually tuned for accuracy or latency. Energy per correct answer is a different objective. If the optimal \(k\) is interior, then both “retrieve more” and “retrieve less” are wrong; if the curve is flat, the whole tuning problem is less important.

## How

1. Run the same reader with \(k\in\{0,1,2,4,8,16,32\}\) on a fixed question set.
2. Measure joules per query and accuracy; compute \(E(k)=\text{joules per correct answer}\).
3. Fit the energy and accuracy curves; test for an interior \(k^\star\).
4. Derive the elasticity condition \(kJ'/J=kA'/A\) at the optimum.
5. Add a certification step: how many queries are needed to certify \(k^\star\)?

## Dataset

SQuAD and NQ; BM25 retriever; Qwen2.5-1.5B Q4 reader; `nvidia-smi` at 50 ms with idle subtraction.

## Method

Energy-accuracy curve; unimodality test; elasticity condition; static-vs-oracle price bound; certification cost.

## Expected

A positive result: an interior \(k^\star\) with ≥20% energy savings at matched accuracy, and a certificate that is cheaper than the savings.

## Results

**Conditional.** **Pan et al., NeurIPS 2025 MLForSys (`an17DUVvuS`)** already shows RAG configurations within 3% accuracy differ by up to **20.2×** in energy. That paper has no \(E(k)=J/\text{correct}\), no \(k^\star\), and no guarantee, so the estimand is not literally pre-empted. But a static-vs-oracle bound is unbounded under query heterogeneity, and certification itself costs energy. Distractors can create two local minima, breaking unimodality.

## Outcome

Only as **certification theory**, not another energy curve. If the pilot shows a flat curve or no interior \(k^\star\), merge into G5 or stop.
