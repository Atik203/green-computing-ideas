# Green Computing — Shortlist and Recommendation

**Date:** 8 November 2026. **Theme:** sustainable AI / green computing in LLM and NLP.  
**Constraints:** 3–4 people, 4-5 months, RTX 3060 6 GB, 24 GB rental possible, ≤ $50 cash, public data, Q1 target.

## Bottom line

**Provisional green lead: Certified Energy-Normalized Data Selection for Instruction Tuning (G4; Core 7.50 / Weighted 7.78).** Its pilot is running. The paper asks whether a selected instruction subset’s quality gain is real at the available energy and evidence budget, or whether the procedure should abstain.

The two fallbacks are **Energy-Optimal Retrieval Depth (G2; 7.00/7.35)** and **Carbon per Correct Answer (G5; 7.00/7.03)**. Both pilots are **MARGINAL**: each has a real, decision-relevant result but fails its strictest pre-registered claim.

## Ranking

| Rank | ID | Title | Core | Weighted | Indep. | Status |
|---:|---|---|---:|---:|---:|---|
| 1 | G4 | **Certified Energy-Normalized Data Selection for Instruction Tuning** | **7.50** | **7.78** | 7.0 | **Pilot running** — standalone only if it clears K2/K3 and \(E_{val}/E_{train}\ge2\) |
| 2 | G2 | **Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation** | 7.00 | 7.35 | 3.0 | **MARGINAL** — static 2pp claim killed; 5pp margin PASS at \(k=4\), 69.3% savings |
| 3 | G5 | **Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies** | 7.00 | 7.03 | 7.5 | **MARGINAL** — budget reversal present; break-even 256.7 J/correct; MDE 33% > 25% |

Scores are pre-pilot research estimates; G4 becomes final only after its pilot lands. They are conservative working estimates, **not acceptance probabilities**.

## Why G4 is the lead

- **New functional:** certify the selected subset under noisy energy measurements, rather than assert an optimistic point estimate.
- **Real guarantee shape:** a certification tax \(\tau=\sigma\sqrt{2/n_v}(\sqrt{\log(2M_2/\delta)}-\sqrt{\log(2M_1/\delta)})\) and an abstention floor \(B_{\min}\).
- **Decision-changing:** tells an engineer whether a selected subset’s advantage is real at the available evidence budget, and when to abstain.
- **Feasible:** 500-example LoRA arms on Pythia-410M; the pilot is ≤2h on the 3060.
- **Not pre-empted:** `2610.14028` gives a per-sample-price approximation guarantee; `2410.16208` budgets selection + training cost; SPICE `2601.23155` owns \((1-1/e)\) submodular selection. None gives the finite-sample **certificate** under noisy energy.

**Caveat:** if G4’s pilot fails K2/K3, kill it rather than merging it into another idea. It must stand alone.

## Survivor results

**G2 — MARGINAL.** 200 SQuAD questions × \(k\in\{0,\dots,32\}\). \(E(32)/E(1)=3.45\); accuracy is flat at 0.520–0.535 for \(k=4\)–16. The static 2pp claim is killed (\(k^\star=32\) boundary, bootstrap \(P=0.61\)), but a 5pp margin gives \(k=4\), **69.3% savings**, \(N^\star=876\). The optimal-depth identity holds at \(k=2\) (+0.044, CI includes 0). The lever is prefill, not BM25: context share reaches 85.1%.

**G5 — MARGINAL.** Pooled CPCA: Direct **21.6**, CoT-short **119.9**, CoT-long **554.8**, SC3 **1614.4** J/correct; accuracies .488/.838/.888/.913. Direct is budget-optimal only below **100.4 J/query**; the first break-even is **256.7 J/correct [192.8, 387.7]**. The fixed-n power gate fires (MDE 33% > 25%), and no dominated runner-up was found. Needs ~387 queries/arm for 15% MDE.

## What to run next

**G4 is running now.** Pythia-410M LoRA; 500-example arms; optimism, \(E_{val}/E_{train}\), certificate tax. Kill if energy \(R^2<0.90\), optimism <0.5pp, \(n_v>5k\), or gain <0.5pp.

**If G4 fails:** the green track is a **7.0 paper** from G2 or G5. Do not start a new green idea in the remaining time.

## Honest caveat

The green track has no 7.9 executable idea. The best case is G4 at 7.5–7.8 if its pilot clears the evidence gates; otherwise it is a solid 7.0 paper (G2 or G5). The 8+ path is gone at this scale.

## Files

- Research reports: `02_g2_green_rag.md`, `04_g4_data_selection.md`, `05_g5_cpca.md`
- Pilot reports: `pilots/02_g2_pilot.md`, `pilots/05_g5_pilot.md`; G4 pending
- Ledgers: `ledger_g2.csv`, `ledger_g4.csv`, `ledger_g5.csv`
- Plain-language pages:
  - `ideas/energy-optimal-retrieval-depth.md` — G2
  - `ideas/certified-data-selection.md` — G4
  - `ideas/carbon-per-correct-answer.md` — G5
