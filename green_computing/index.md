# Green Computing — Shortlist and Recommendation

**Date:** 8 November 2026. **Theme:** sustainable AI / green computing in LLM, ML, and NLP.  
**Constraints:** 3–4 people, 6–7 months, RTX 3060 6 GB, 24 GB rental possible, ≤ $70, public data, Q1 target.

## Bottom line

**The green lead changed: G3′ failed its pilot and is killed at this scale.** G3′ — certified carbon-aware quantization/sparsity allocation — was the highest-scoring green idea (7.88/7.98), but the 0.5B/3060 pilot fired two kill criteria: 4-bit NF4 uses **15% more energy** than FP16 on Qwen (and 11.8% more on Pythia), and held-out additive energy \(R^2\) is 0.506/−0.99, far below the 0.8 bar. At \(B=0.8\times\)FP16, Qwen has **no feasible config**; Pythia has only uniform INT8, so no per-layer mix can be compared. There is no ≥3-point allocation gain to certify.

**Provisional green lead: G4′ — certified data selection (Core 7.50 / Weighted 7.78).** It is now the highest-scoring survivor, but its pilot has not run. G2 (7.00/7.35) and G5′ (7.00/7.03) are solid 7.0 fallbacks. G1 is killed on prior art.

## Ranking

| Rank | Idea | Core | Weighted | Indep. | Verdict |
|---:|---|---:|---:|---:|---|
| 1 | **G4′ — certified data selection** | **7.50** | **7.78** | 7.0 | **Provisional lead** — pilot queued; standalone only if it clears K2/K3 and \(E_{val}/E_{train}\ge2\) |
| 2 | **G2 — energy-optimal retrieval depth** | 7.00 | 7.35 | 3.0 | **Pilot running** — only as certification theory; overlaps G5 |
| 3 | **G5′ — CPCA-CEA** | 7.00 | 7.03 | 7.5 | **Pilot queued** — 7.0 paper after repair |
| — | **G3′ — certified quantization/sparsity allocation** | 7.88 | 7.98 | 7.5 | **KILLED by pilot** — NF4 energy loss, no feasible mix at 0.8×FP16, additive \(R^2\) fails |
| — | **G1 — carbon-budgeted cascade** | 4.20 | 4.70 | 3.0 | **KILLED** — GAR + C3PO pre-empt |

Scores for G4′/G2/G5′ are pre-pilot research scores; they become final only after their pilots land. G3′ keeps its pre-pilot score in the table for the record, but it is **not executable** at 0.5B/3060.

## Why G4′ is now the lead

- **New functional:** certify the selected subset under noisy energy measurements, rather than assert an optimistic point estimate.
- **Real guarantee shape:** a certification tax \(\tau=\sigma\sqrt{2/n_v}(\sqrt{\log(2M_2/\delta)}-\sqrt{\log(2M_1/\delta)})\) and an abstention floor \(B_{\min}\).
- **Decision-changing:** tells an engineer whether a selected subset’s advantage is real at the available evidence budget, and when to abstain.
- **Feasible:** 500-example LoRA arms on Pythia-410M; the pilot is ≤2h on the 3060.
- **Not pre-empted:** `2610.14028` gives a per-sample-price approximation guarantee; `2410.16208` budgets selection + training cost; SPICE `2601.23155` owns \((1-1/e)\) submodular selection. None gives the finite-sample **certificate** under noisy energy.

**Caveat:** G4′ uses the same certified-budget device that G3′ proposed. With G3′ killed, G4′ cannot be “folded into G3”; it must stand alone. If its pilot fails K2/K3, kill it rather than merging.

## What the G3 pilot showed

| Check | Qwen2.5-0.5B | Pythia-410M | Verdict |
|---|---:|---:|---|
| NF4 vs FP16 energy | **+15.0%** [−19.4, −10.4] | **+11.8%** [−24.7, +5.8] | **K1 FAIL** — 4-bit is an energy loss |
| INT8 vs FP16 energy | −11.9% | −37.7% | INT8 is the only energy win |
| Accuracy spread (core) | 4.5 pt, near/below chance | 2.5 pt, at chance | **K2 FAIL in substance** |
| Max core CV | 10.3% | 17.8% | K3 PASS |
| Held-out additive \(R^2\) | −0.99 | 0.506 | **K4 FAIL** |
| Spearman(energy, latency) | 0.071 | 0.394 | K5 PASS |
| Mix feasible at \(0.8\times\)FP16 | none | only uniform INT8 | **Pass criterion unreachable** |

The measurement apparatus is sound: idle-subtracted joules, CV below 20%, and energy is not a restatement of latency. Two empirical facts are publishable as a measurement note: (i) at 0.5B on this GPU, NF4 is an energy and latency loss, while uniform INT8 is the only energy win; (ii) allocation-level differences are below 5-rep noise, so a certified per-layer allocation needs far more repetitions than a pilot can afford.

## What to run next

**Queue:** G2 (running) → G5 → G4. G3 is excluded; G1 was already excluded.

- **G2 — energy-optimal retrieval depth:** 150–200 SQuAD questions, \(k\in\{0,1,2,4,8,16,32\}\), \(E(k)=J/\text{correct}\), interior \(k^\star\). Kill if relative energy effect <10%, \(k^\star\) at boundary, or adaptive gain <15%.
- **G5 — CPCA-CEA:** 80 GSM8K + ARC queries; direct / CoT-short / CoT-long / SC3; CPCA, frontier, break-even price, rank reversal. Kill if CV >0.35, MDE >25%, or no reversal.
- **G4 — certified data selection:** Pythia-410M LoRA; 500-example arms; optimism, \(E_{val}/E_{train}\), certificate tax. Kill if energy \(R^2<0.90\), optimism <0.5pp, \(n_v>5k\), or gain <0.5pp.

**If G4 fails:** the green track has only G2 or G5 as a ~7.0 paper. Do not revive G3′ on 0.5B/3060. A minimum viable G3 pivot would need ≥1.5B models, 20–30 reps, and above-chance few-shot accuracy — that is a separate, larger study, not the current pilot.

## Honest caveat

The green track no longer has a 7.9 executable idea. The best case is G4′ at 7.5–7.8 if its pilot clears the evidence gates; otherwise the track is a solid 7.0 paper (G2 or G5). The 8+ path is gone at this scale. Do not spend the main-track GPU budget on reviving G3′.

## Files

- Research reports: `01_g1_cascade.md`, `02_g2_green_rag.md`, `03_g3_quantization.md`, `04_g4_data_selection.md`, `05_g5_cpca.md`
- Pilot reports: `pilots/03_g3_pilot.md`; G2/G5/G4 pending
- Ledgers: `ledger_g1.csv` … `ledger_g5.csv`, `pilots/ledger_g3_pilot.csv`
- Plain-language pages: `ideas/01_g1_cascade.md` … `ideas/05_g5_cpca.md`
