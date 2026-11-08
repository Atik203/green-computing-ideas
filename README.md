# Green-Computing Research Ideas — 2026

**Course track:** green computing / sustainable AI  
**Team:** 3–4 people · 4-5 months · local RTX 3060 6 GB · 24 GB rental available · public data · ≤ $50 cash  
**Target:** a good Q1 paper. A 7+ paper is a strong outcome; 8+ only if genuinely verified.

This repository is the **green-computing track** of a two-track selection package. The data-mining track is separate: **https://github.com/Atik203/data-mining-ideas**.

---

## Start here

| If you want… | Read |
|---|---|
| The green ranking and recommendation | [`green_computing/index.md`](green_computing/index.md) |
| One page per green idea | [`green_computing/ideas/`](green_computing/ideas/) |
| The five research reports | [`green_computing/`](green_computing/) |
| The pilot protocol and G2/G3 results | [`green_computing/pilots/`](green_computing/pilots/) |

The master prompt and task list are **workspace-level** files kept in the local `Research-Ideas` root, not in this repo.

---

## Current ranking (8 November 2026)

| Rank | ID | Title | Core | Weighted | Indep. | Status |
|---:|---|---|---:|---:|---:|---|
| 1 | G4 | **Certified Energy-Normalized Data Selection for Instruction Tuning** | **7.50** | **7.78** | 7.0 | **Provisional lead** — pilot queued; must stand alone |
| 2 | G2 | **Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation** | 7.00 | 7.35 | 3.0 | **MARGINAL** — static 2pp claim killed; 5pp margin PASS at \(k=4\), 69.3% savings |
| 3 | G5 | **Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies** | 7.00 | 7.03 | 7.5 | **MARGINAL** — budget reversal present; break-even 256.7 J/correct; MDE 33% > 25% |
| — | G3 | **Certified Carbon-Aware Quantization and Sparsity Allocation** | 7.88 | 7.98 | 7.5 | **KILLED by pilot** |
| — | G1 | **Carbon-Budgeted Model Cascades for Language-Model Inference** | 4.20 | 4.70 | 3.0 | **KILLED** on prior art |

**Certified Carbon-Aware Quantization and Sparsity Allocation (G3) was the pre-pilot lead, but its 0.5B/3060 pilot killed it.** NF4 uses **15% more energy** than FP16 on Qwen (11.8% more on Pythia), held-out additive \(R^2\) is 0.506/−0.99, and no layer mix is feasible at \(0.8\times\)FP16. The measurement is a useful negative result: on this GPU, uniform INT8 is the only energy win. See [`green_computing/pilots/03_g3_pilot.md`](green_computing/pilots/03_g3_pilot.md).

**Provisional recommendation:** **Certified Energy-Normalized Data Selection for Instruction Tuning (G4)**, pending its pilot. If G4 fails, the green track falls back to **Energy-Optimal Retrieval Depth (G2)** or **Carbon per Correct Answer (G5)** at ~7.0. G3 should not be revived at 0.5B/3060; a minimum viable pivot needs ≥1.5B models, 20–30 reps, and above-chance few-shot accuracy.

Scores use the Round-8 eight-factor rubric. They are conservative working estimates, **not acceptance probabilities**. Green scores become final only after the G2/G5/G4 pilots land.

---

## Idea pages

Each page has the same sections: **What / Why / How / Dataset / Method / Expected / Results / Outcome**.

| Idea | Page |
|---|---|
| Certified Energy-Normalized Data Selection for Instruction Tuning (G4) | [`certified-data-selection.md`](green_computing/ideas/certified-data-selection.md) |
| Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation (G2) | [`energy-optimal-retrieval-depth.md`](green_computing/ideas/energy-optimal-retrieval-depth.md) |
| Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies (G5) | [`carbon-per-correct-answer.md`](green_computing/ideas/carbon-per-correct-answer.md) |
| Certified Carbon-Aware Quantization and Sparsity Allocation (G3) | [`certified-quantization-allocation.md`](green_computing/ideas/certified-quantization-allocation.md) |
| Carbon-Budgeted Model Cascades for Language-Model Inference (G1) | [`carbon-budgeted-model-cascades.md`](green_computing/ideas/carbon-budgeted-model-cascades.md) |

---

## What is inside

```
green_computing/
├── index.md                     ← ranking and recommendation
├── 00_BRIEF.md
├── 01…05 research reports
├── ideas/                       ← five plain-language pages
└── pilots/
    ├── 00_PILOT_BRIEF.md
    └── 03_g3_pilot.md
```

## What is not here

Raw pilot JSON/CSV, model weights, ledgers, and credentials. G2/G5/G4 pilot reports will be added when they finish.

## Security

No API key, Hugging Face token, or `.env` file is included. The local workspace previously contained a hardcoded HF token in a pilot script; it has been redacted, and the token should be rotated separately.

## Status

Snapshot: **8 November 2026**. G3 is killed; G2 is running; G5 and G4 are queued. The green lead is provisional until G4’s pilot lands.
