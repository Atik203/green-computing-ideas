# G4 — Energy-Normalized Data Selection for Instruction Tuning

**One-line:** Select the instruction examples that give the most quality per joule of fine-tuning, with a certificate rather than an optimistic point estimate.  
**Score:** Core **7.50** / Weighted **7.78** (repaired G4′); original **5.88 / 6.25 — KILLED**.  
**Independence:** 7/10 (overlaps G3′’s certified-budget device).

## What

Instruction-tuning datasets contain thousands of examples. Some teach more per unit of training energy; some are redundant. The question is which subset to train on when the energy budget is fixed, and whether the selected subset’s advantage is real or just selection optimism.

## Why

Data selection is usually framed as “maximize quality” or “minimize cost.” Energy-normalized selection is different: a small, high-value subset may beat a larger, cheaper one. But the selected subset’s estimated gain is optimistic; a certificate is needed before deployment.

## How

1. Score candidate examples with an influence or submodular utility.
2. Measure the energy of selection, training, and evidence.
3. Maximize a uniform lower confidence bound over candidate subsets, or abstain if the budget is too low.
4. Compare against random, BM25, and diversity baselines.

## Dataset

Alpaca, Dolly, FLAN subsets; Pythia-410M and Qwen2.5-0.5B with LoRA; measured training energy.

## Method

Submodular selection + certification; certification tax \(\tau=\sigma\sqrt{2/n_v}(\sqrt{\log(2M_2/\delta)}-\sqrt{\log(2M_1/\delta)})\); abstention floor \(B_{\min}\); total energy = selector + training + evidence.

## Expected

A positive result: the certified subset beats uniform/random selection by ≥0.5pp at the same energy, with a certificate that is not vacuous. If evidence energy dominates training energy, the problem is mostly measurement.

## Results

**Original killed.** The functional is already published: `2610.14028` formalizes max utility under per-sample prices with an approximation guarantee; `2410.16208` budgets selection + training cost; **SPICE `2601.23155` (ICLR 2026)** owns \((1-1/e)\) submodular selection for instruction tuning; Haddaji et al. already report an interior energy-optimal subset size. The proposed theorem is classical plumbing.

**Repair G4′:** certify the selected subset; report the certificate tax and abstention floor. **7.50 / 7.78.** But it uses the same certified-budget device as G3′, and G3′ was killed by its pilot. G4′ therefore cannot be folded into G3′; it must stand alone and clear its own pilot.

## Outcome

**Standalone only if** the pilot shows optimism ≥0.5pp, \(n_v\le5k\), and \(E_{val}/E_{train}\ge2\). Otherwise **kill G4′**; do not merge it into the killed G3′.
