# Green Computing — Idea Search Brief

**Date:** 8 November 2026.  
**Theme:** sustainable AI / green computing in LLM, ML, and NLP.  
**Team:** 3–4 people · 6–7 months · RTX 3060 6 GB local, 24 GB rental possible · ≤ $70 cash · public data · Q1 journal target.

**Rules:** This is a separate track. Do not touch `research_audit/round9`, `round10`, or the six running gates. No GPU, no model loads, no HF downloads, no paid API. Web search and CPU-only calculations are allowed. Write only your assigned file under `green_computing/`.

**Scoring:** Round-8 eight-factor rubric: Novelty 25%, Method 20%, Math 15%, Experiments 10%, Dataset 10%, Q1 fit 10%, Feasibility 5%, Compute 5%. Core = mean of Novelty, Method, Math, Experiments. Weighted = all eight. Independence is separate.

**8+ bar:** a new functional (not just a new application), a non-saturated model, a non-identity theorem (sign/bound/impossibility/rate), and a decision-changing test. If the closest paper already has the same estimand and guarantee, the idea is at most a 6.

**Energy measurement:** prefer `nvidia-smi --query-gpu=power.draw` plus wall-clock, or `codecarbon` if installed. A laptop 3060 may expose coarse power readings; state this honestly and use a proxy (GPU utilization × TDP, or measured joules from a wall meter) only if documented. Carbon intensity must come from a public dataset/API with a fixed regional average; no paid API.

---

## Five candidate directions

### G1 — Carbon-budgeted model cascade

**Question:** Given a per-query carbon budget and a latency SLA, which model should answer: local small, local medium, or large/remote?  
**Functional:** policy \(\pi(x)\) maximizing expected quality subject to \(\mathbb{E}[\text{carbon}(\pi)] \le B\) and latency \(\le L\).  
**Theorem shape:** threshold structure in a quality-per-joule score; a distribution-free controller with a finite-sample budget guarantee; a lower bound showing no static threshold is minimax-optimal under heterogeneous queries.  
**Dataset:** public QA/reasoning sets (MMLU, ARC, GSM8K) + measured energy per model.  
**Why it might survive:** carbon budget is a hard constraint, not a cost penalty; most routing work optimizes cost/latency, not carbon with a guarantee.

### G2 — Energy-optimal retrieval depth (Green RAG)

**Question:** Retrieval costs energy; generation costs energy; more retrieved passages reduce generation length/errors but add context energy. What is the energy-optimal \(k\)?  
**Functional:** \(E(k)=\text{energy per correct answer}\) or joules per correct answer; \(k^\star=\arg\min E(k)\).  
**Theorem shape:** a U-shaped risk/carbon curve under a noisy retriever; a minimax lower bound on the energy needed to certify \(k^\star\); a matching adaptive policy.  
**Dataset:** NQ/SQuAD + BM25; local reader; measured energy.  
**Why it might survive:** RAG efficiency work measures latency/tokens, rarely joules per correct answer with a guarantee.

### G3 — Carbon-aware quantization / sparsity allocation

**Question:** Given a total energy budget at inference, how should bit-width/sparsity be allocated across layers to maximize accuracy?  
**Functional:** \(\max_{\{b_l,s_l\}} \text{accuracy}\) subject to \(\sum_l E_l(b_l,s_l) \le B\).  
**Theorem shape:** a knapsack/LP relaxation with an integrality gap; a greedy allocation with a \((1-1/e)\)-type guarantee under submodular energy-accuracy curves; a lower bound.  
**Dataset:** small models (Qwen2.5-0.5B/1.5B, Pythia) on MMLU/ARC; measured energy per configuration.  
**Why it might survive:** most quantization work reports accuracy/size, not a carbon-constrained allocation with a guarantee.

### G4 — Energy-normalized data selection for instruction tuning

**Question:** Which instruction examples give the most quality per joule of fine-tuning?  
**Functional:** select a subset \(S\) maximizing \(\Delta\text{quality}(S)\) subject to training energy \(E(S)\le B\); or maximize quality per joule.  
**Theorem shape:** submodularity of the quality-energy utility; greedy \((1-1/e)\); a lower bound; an energy-aware influence estimator.  
**Dataset:** Alpaca/Dolly/FLAN subsets; Pythia-410M/Qwen2.5-0.5B; LoRA; measured energy.  
**Why it might survive:** data selection is crowded, but energy-normalized selection with a guarantee is not the standard objective.

### G5 — Carbon per correct answer: a reasoning-strategy benchmark

**Question:** Which reasoning strategy is greenest per correct answer: direct answer, CoT, self-consistency, or debate?  
**Functional:** \(\text{CPCA}(s)=\text{joules per correct answer}\) for strategy \(s\), with a decomposition into model, tokens, retries, and tool calls.  
**Theorem shape:** an estimand/measurement paper; a variance-aware estimator and a decision rule for when a strategy is green; no theorem required, but a reusable benchmark is needed.  
**Dataset:** GSM8K/ARC/MMLU; small models; measured energy.  
**Why it might survive:** most green-AI papers measure training or single-call inference, not the carbon cost of reasoning strategies per correct answer.

---

## Required output per idea

1. **Plain-language problem** (2–3 sentences).
2. **Exact estimand/functional** and decision it changes.
3. **Closest 5 papers 2024–2026** at the functional level, with IDs and verified/unverified labels. If the same estimand+guarantee exists, say KILL and propose the nearest defensible repair.
4. **Theorem/bound shape** with assumptions and a proof sketch, or a measurement estimand with a variance/power plan.
5. **Dataset and measurement protocol** (public data, energy meter, carbon intensity source).
6. **Pilot:** ≤2 hours on the 3060, with explicit kill criteria.
7. **Scores** (all eight factors, Core/Weighted, Independence).
8. **Verdict:** GO / CONDITIONAL / KILL, and what would change it.

**File names:** `01_g1_cascade.md`, `02_g2_green_rag.md`, `03_g3_quantization.md`, `04_g4_data_selection.md`, `05_g5_cpca.md`, each with `ledger_gX.csv`.
