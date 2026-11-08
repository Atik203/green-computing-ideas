# Green Computing — Pilot Brief (4 survivors)

**Date:** 8 November 2026.  
**Running:** G3, G2, G5, G4. **Excluded:** G1 (killed).  
**Hardware:** RTX 3060 6 GB laptop, 32 GB RAM, Windows, `C:\ProgramData\anaconda3\python.exe`.  
**Rules:** one GPU job at a time; queue via `green_computing/pilots/queue/DONE_GX` flags. No paid API. No model downloads larger than ~1 GB without checking the token/network. Do not touch `research_audit/round9`, `round10`, or other green files. Write only your `0X_gX_pilot.md`, ledger, and flag.

## Energy measurement (all pilots)

1. Check `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits` once. If it returns a number, sample every 100 ms during the run and integrate joules.
2. If power.draw is unavailable or constant, use a documented proxy: `GPU utilization × TDP` (3060 laptop TDP ~80–115 W, state the value used) and subtract idle. Report the measurement method in the pilot.
3. Record wall-clock, mean power, joules, and coefficient of variation. Energy claims need CV ≤20–25%; otherwise mark the pilot inconclusive.
4. Carbon: pilots report joules; carbon = joules × a fixed regional intensity. Do not pay for an API.

## Queue order and flags

`G3 → G2 → G5 → G4`

- G3 starts immediately.
- G2 waits for `queue/DONE_G3`.
- G5 waits for `queue/DONE_G2`.
- G4 waits for `queue/DONE_G5`.
- Each writes `queue/DONE_GX` with `PASS/MARGINAL/KILL` + one-line numbers when its report is complete.
- Poll every 3–5 minutes with `Start-Sleep 180`; do not load a model before the flag exists.

---

## G3 — certified quantization/sparsity allocation (first)

**Question:** does per-layer bit/sparsity allocation beat uniform compression at the same measured energy, with a certificate that survives measurement noise?

**Models:** Qwen2.5-0.5B and Pythia-410M. **Configs:** FP16, INT8, NF4 (bitsandbytes), plus one simple layer-wise mix (e.g., first/last blocks FP16, middle NF4) if time permits. **Data:** ARC-Easy and MMLU, 200 items each (subset if needed). **Energy:** 5 repetitions per config, 100 ms samples, idle subtraction.

**Metrics:** accuracy, joules/query, additive-energy \(R^2\), CV, Spearman(energy, latency), per-layer allocation gain at \(B=0.8\times\)FP16.

**Kill:** energy headroom <10%, accuracy spread <1pt, CV >20%, additive \(R^2<0.8\), or Spearman(energy, latency) >0.95. **Pass:** a per-layer mix gives ≥3 points over uniform at 0.8× FP16 energy with a tight certificate.

**Output:** `green_computing/pilots/03_g3_pilot.md`, `ledger_g3_pilot.csv`, `queue/DONE_G3`.

---

## G2 — energy-optimal retrieval depth

**Question:** is there an interior \(k^\star\) minimizing joules per correct answer?

**Model:** Ollama `qwen3.5:4b` (already local) or Qwen2.5-0.5B if faster; document the choice. **Data:** 200–300 SQuAD-style questions; reuse `research_audit/round9/prep_i08b.json` if available (20,958 passages) or download SQuAD. **Retriever:** BM25 (rank-bm25). **Grid:** \(k\in\{0,1,2,4,8,16,32\}\). **Energy:** 100 ms samples, idle subtraction.

**Metrics:** \(E(k)=J/\text{correct}\), accuracy, energy split retrieval/generation, interior \(k^\star\), elasticities \(kJ'/J\) and \(kA'/A\).

**Kill:** relative energy effect <10%, \(k^\star\) at boundary, uncertifiable CI, split <5%, or adaptive gain <15%. **Pass:** interior \(k^\star\) with ≥20% energy savings at matched accuracy and a certificate cheaper than the savings.

**Output:** `green_computing/pilots/02_g2_pilot.md`, `ledger_g2_pilot.csv`, `queue/DONE_G2`.

---

## G5 — CPCA / carbon-effectiveness frontier

**Question:** does a reasoning strategy with the best joules-per-correct-answer rank differ from the strategy that is optimal under a carbon budget?

**Model:** Ollama `qwen3.5:4b` (local). **Data:** 80 GSM8K + ARC queries. **Strategies:** direct, CoT-short, CoT-long, self-consistency \(k=3\). **Energy:** 100 ms samples, idle subtraction.

**Metrics:** accuracy, joules/query, CPCA, failure-inclusive \(R=E^+ + ((1-A)/A)E^-\), paired bootstrap CI, frontier, break-even price, rank reversal.

**Kill:** energy CV >0.35, MDE >25%, or no frontier/budget reversal. **Pass:** a reversal where the best-CPCA strategy is not budget-optimal.

**Output:** `green_computing/pilots/05_g5_pilot.md`, `ledger_g5_pilot.csv`, `queue/DONE_G5`.

---

## G4 — certified data selection

**Question:** does a certified selected subset beat random/uniform selection at the same training energy, and is the optimism real?

**Model:** Pythia-410M LoRA. **Data:** Alpaca/Dolly subset; 500-example arms. **Runs:** 5 calibration + 8 at \(k=500\) across four selectors (random, BM25/diversity, influence/submodular, certified). **Energy:** 100 ms samples during training; include selector + training + evidence energy.

**Metrics:** validation accuracy, optimism (selected vs held-out), \(E_{val}/E_{train}\), certificate tax, \(n_v\) needed, selection gain at equal energy.

**Kill:** energy \(R^2<0.90\), optimism <0.5pp, \(n_v>5k\), or selection gain <0.5pp. **Pass:** certified subset beats random by ≥0.5pp at equal energy with \(E_{val}/E_{train}\ge2\) (standalone threshold).

**Output:** `green_computing/pilots/04_g4_pilot.md`, `ledger_g4_pilot.csv`, `queue/DONE_G4`.

---

## Final chat per pilot

≤300 words: verdict, actual numbers (accuracy/energy/CV or \(k^\star\)/CPCA), kill/pass, path.
