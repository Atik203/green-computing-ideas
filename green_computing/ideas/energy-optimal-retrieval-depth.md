# Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation

**Internal short name:** G2

**One-line:** Find the retrieval depth \(k^\star\) that minimizes joules per correct answer, not just latency or tokens.  
**Score:** Core **7.00** / Weighted **7.35** — **MARGINAL**.  
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

200 SQuAD v1.1 dev questions; 20,958 SQuAD contexts chunked to 45,651 passages; BM25 top-32; Ollama `qwen3.5:4b` Q4 reader; `nvidia-smi` power at 100 ms with idle subtraction and block calibration.

## Method

Energy-accuracy curve; unimodality test; elasticity condition; static-vs-oracle price bound; certification cost.

## Expected

A positive result: an interior \(k^\star\) with ≥20% energy savings at matched accuracy, and a certificate that is cheaper than the savings.

## Results

**Pilot: MARGINAL.** 200 SQuAD questions × \(k\in\{0,1,2,4,8,16,32\}\), 1,400 generations. The energy curve is steep: \(E(32)=157.1\) vs \(E(1)=45.5\) J/correct — **3.45×** — and the affine energy model fits with \(R^2=.999\). Accuracy rises 0.395 → 0.570, but is flat at 0.520–0.535 for \(k=4\)–16.

The pre-registered **static 2pp claim is killed**: \(k^\star=32\) lands on the boundary and bootstrap \(P(k^\star\text{ optimal})=0.61<0.8\). The optimal-depth identity \(kJ'/J=kA'/A\) holds at \(k=2\) (gap +0.044, CI [−0.050, +0.128]) and fails at \(k=32\) (+0.700), so 32 passages is over-retrieval by the theory’s own diagnostic. The context-side energy share rises from 36.3% to **85.1%**; BM25 retrieval is only 3.5–17.3%, so the lever is prefill, not the retriever.

Under a **5pp EM margin**, the pilot is a **PASS**: \(k=4\), **69.3% savings**, \(N^\star=876\) queries. Adaptive matched-oracle gain is 78.8%.

**Prior art:** Pan et al., NeurIPS 2025 MLForSys (`an17DUVvuS`) already shows RAG configurations within 3% accuracy differ by up to **20.2×** in energy. It has no \(E(k)=J/\text{correct}\), no \(k^\star\), and no guarantee, so the estimand is not literally pre-empted.

## Outcome

**MARGINAL.** Do not claim a static 2pp-optimal depth. The defensible next step is a powered rerun with \(n\ge500\)–1000 and a **pre-specified accuracy margin** (5pp gives \(k=4\), 69.3% savings), or a pivot to query-conditional depth. If neither is done, merge the energy-split result into G5 rather than keeping a separate paper.
