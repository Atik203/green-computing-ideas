# Certified Carbon-Aware Quantization and Sparsity Allocation

**Internal short name:** G3

**One-line:** Allocate bits and structured sparsity across layers to maximize accuracy under a measured energy budget — with a certificate that accounts for noisy energy measurements.  
**Score:** Pre-pilot Core **7.88** / Weighted **7.98** — **PILOT KILL**.  
**Independence:** 7.5/10.

## What

A small model can be compressed unevenly. Some layers tolerate 2-bit weights or 2:4 sparsity; others break. The practical question is: given a fixed energy budget per query, which layers get the bits and which get the sparsity? Today’s answer is a grid search over a few uniform configurations, with no guarantee that the chosen model actually fits the budget when the energy measurement itself is noisy.

## Why

Uniform quantization wastes energy on layers that are not sensitive. Per-layer allocation could preserve accuracy at lower joules. But the allocation is only useful if the measured energy is trustworthy; otherwise the “optimal” model may exceed the budget in deployment.

## How

1. Measure accuracy and energy for each layer–configuration pair (bit-width × sparsity).
2. Solve the multiple-choice knapsack exactly by dynamic programming (\(L\le32\), 10 choices/layer — milliseconds).
3. Build a **certified feasible set** using one-sided confidence radii on the noisy energy estimates.
4. Maximize a lower confidence bound on accuracy over that set; report the certified frontier \(A^*_\delta(B)\).
5. Prove the price of noise and test whether the allocation beats uniform compression at the same measured joules.

## Dataset

Qwen2.5-0.5B and Pythia-410M; ARC-Easy and MMLU, 200 items each; FP16 / INT8 / NF4; `nvidia-smi` at 100 ms, 5 repetitions.

## Method

Multiple-choice knapsack + DP; certified frontier; price-of-noise theorem \(O(\lambda^*\sigma\sqrt{\log(1/\delta)/n})\) with matching \(\Omega(\lambda^*\sigma/\sqrt{n})\); shadow price of a joule.

## Expected

A positive result: a per-layer allocation gives **≥3 accuracy points** over uniform compression at \(B=0.8\times\)FP16 energy, with additive energy \(R^2\ge0.8\) and CV ≤20%. If energy is essentially latency, the paper collapses to a measurement.

## Results

**Pilot (8 Nov, 0.5B/3060): KILL.** The pilot ran Qwen2.5-0.5B and Pythia-410M with FP16/INT8/NF4 plus layer mixes, 400 queries per rep, idle-subtracted energy. The two decisive failures:

- **K1 — 4-bit is an energy loss.** NF4 uses **15.0% more energy** than FP16 on Qwen (95% CI [−19.4, −10.4]%) and 11.8% more on Pythia (CI includes 0). NF4 runs 1.56× slower while drawing only ~10 W less. INT8 is the only energy win (−11.9% Qwen, −37.7% Pythia), but it is a uniform config — there is nothing to allocate.
- **K4 — held-out additivity fails.** The additive energy model has in-sample \(R^2=.88\)–.90, but leave-one-out \(R^2\) is **−0.99** (Qwen) and **0.506** (Pythia), below the 0.8 bar. Allocation differences (0.05–0.2 J/query) are inside 3–5-rep noise (SD 0.03–0.40 J/query).

The pass criterion is unreachable: at \(B=0.8\times\)FP16, Qwen has **no feasible config** (lowest is INT8 at 2.38 J vs budget 2.16 J); Pythia has only uniform INT8, no mix. Accuracy is at the chance floor (Qwen 15.8% FP16; Pythia 27.3%), so the nominal 3–4 point “gain” is letter bias, not knowledge. K3 (CV <20%) and K5 (energy ≠ latency; Spearman .07/.39) pass, but they cannot rescue the idea.

**Original framing killed:** the knapsack/greedy version is **6.75 / 7.05**. Exact DP solves it in milliseconds, so an integrality-gap or greedy theorem is not decision-changing; SigmaQuant (`2602.22136`), LieQ (`2508.03332`), and PolyQ (`2607.14618`) already allocate bits under budgets.

**Repaired idea:** make **measurement noise and non-separability** the core. Certified frontier + price-of-noise theorem + a counterexample where layer-separable accuracy is off by \(\Omega(1)\). No 2024–2026 paper has the same estimand and finite-sample guarantee. Pre-pilot score **7.88 / 7.98** — but the pilot shows the premise does not hold at 0.5B/3060.

## Outcome

**KILL the static per-layer allocation at this scale.** Do not proceed to the full study on 0.5B/3060. The measurement apparatus is sound, so a **measurement note** is possible: “At 0.5B on an RTX 3060, NF4 is an energy and latency loss; uniform INT8 is the only energy win; allocation-level differences are below 5-rep noise.” A minimum viable G3 pivot needs ≥1.5B models, 20–30 reps, and above-chance few-shot accuracy — a separate, larger study. Do not revive G3′ as the green lead.
