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
| The two active research reports | [`green_computing/`](green_computing/) |
| The pilot reports | [`green_computing/pilots/`](green_computing/pilots/) |

The master prompt and task list are **workspace-level** files kept in the local `Research-Ideas` root, not in this repo.

---

## Current ranking (8 November 2026)

| Rank | ID | Title | Core | Weighted | Indep. | Status |
|---:|---|---|---:|---:|---:|---|
| 1 | G2 | **Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation** | 7.00 | 7.35 | 6/10 (3/5) | **MARGINAL** — static 2pp claim killed; 5pp margin PASS at \(k=4\), 69.3% savings |
| 2 | G5 | **Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies** | 7.00 | 7.03 | 7.5/10 | **MARGINAL** — budget reversal present; break-even 256.7 J/correct; MDE 33% > 25% |

**Independence scale:** G2 was originally scored **3/5**; it is shown as **6/10** for comparison with the 0–10 scores used elsewhere.

## Full eight-factor scores

| ID | Title | Novelty | Method | Math | Experiments | Dataset | Q1 fit | Feasibility | Compute | Core | Weighted | Independence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G2 | Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation | 7.0 | 7.5 | 7.5 | 6.0 | 7.0 | 8.0 | 8.5 | 9.0 | **7.00** | **7.35** | 6/10 (3/5) |
| G5 | Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies | 6.5 | 7.5 | 7.5 | 6.5 | 6.5 | 7.0 | 8.0 | 7.5 | **7.00** | **7.03** | 7.5/10 |

**Recommendation:** take **G2** as the green paper. It has the higher weighted score and the larger effect. **G5** is the independent fallback.

**No 7.5+ green idea remains.** The track is two honest 7.0 papers.

Scores use the Round-8 eight-factor rubric. They are conservative working estimates, **not acceptance probabilities**.

---

## Idea pages

Each page has the same sections: **What / Why / How / Dataset / Method / Expected / Results / Outcome**.

| Idea | Page |
|---|---|
| Energy-Optimal Retrieval Depth in Retrieval-Augmented Generation (G2) | [`energy-optimal-retrieval-depth.md`](green_computing/ideas/energy-optimal-retrieval-depth.md) |
| Carbon per Correct Answer: A Carbon-Effectiveness Frontier for Reasoning Strategies (G5) | [`carbon-per-correct-answer.md`](green_computing/ideas/carbon-per-correct-answer.md) |

---

## What is inside

```
green_computing/
├── index.md                     ← ranking and recommendation
├── 02_g2_green_rag.md
├── 05_g5_cpca.md
├── ideas/                       ← two plain-language pages
└── pilots/
    ├── 02_g2_pilot.md
    └── 05_g5_pilot.md
```

Killed ideas are not part of this repository. They are archived in the local workspace for the record.

## What is not here

Raw pilot JSON/CSV, model weights, ledgers, and credentials. The G2 and G5 pilot reports are included.

## Security

No API key, Hugging Face token, or `.env` file is included. The local workspace previously contained a hardcoded HF token in a pilot script; it has been redacted, and the token should be rotated separately.

## Status

Snapshot: **8 November 2026**. Both green pilots are complete; both are MARGINAL. G2 is the lead, G5 the fallback.
