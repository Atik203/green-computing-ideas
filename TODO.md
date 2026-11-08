# TODO — Green-Computing Track

Last updated: 8 November 2026.

## Complete

- [x] Five research reports: G1–G5 — `green_computing/01…05`
- [x] Five full-title idea pages — `green_computing/ideas/`
- [x] Green ranking and recommendation — `green_computing/index.md`
- [x] G3 pilot: **KILL** — NF4 uses 15% more energy than FP16; held-out additive \(R^2\) fails; no mix feasible at \(0.8\times\)FP16
- [x] G2 main run and analysis — **KILL signal** (k*=32 boundary, zero savings, uncertifiable); report pending

## Next

- [ ] G2: write report, ledger, and `DONE_G2`
- [ ] G5: run the CPCA / carbon-effectiveness pilot (80 GSM8K + ARC; direct / CoT-short / CoT-long / SC3)
- [ ] G4: run the certified data-selection pilot (Pythia-410M LoRA; 500-example arms)
- [ ] Update `green_computing/index.md` and the repo README with the G5/G4 outcomes

## Rules

- One GPU job at a time; use the queue flags under `green_computing/pilots/queue/`.
- Do not revive G3 at 0.5B/3060. A viable pivot needs ≥1.5B models, 20–30 reps, and above-chance few-shot accuracy.
- G3 and G1 are killed; keep them only as negative evidence.
- Do not claim an 8+ paper. The current best executable case is G4 at 7.5–7.8 if its pilot clears the evidence gates.
