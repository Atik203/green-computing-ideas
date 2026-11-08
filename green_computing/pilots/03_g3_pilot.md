# G3 pilot — certified quantization/sparsity allocation

**Date:** 8 November 2026.
**Verdict: KILL** (static per-layer energy allocation at this scale/hardware).
**Hardware:** RTX 3060 Laptop 6 GB (power.limit reported `[N/A]`; observed ceiling 94.8 W), 32 GB RAM, Windows.
**Software:** Python 3.12 (`C:\ProgramData\anaconda3\python.exe`), torch 2.11.0+cu128, transformers 5.14.1, bitsandbytes 0.50.2, datasets 5.1.0.

---

## 1. What was run

| Model | Params | Cache | Configs |
|---|---:|---|---|
| Qwen2.5-0.5B | 0.49 B | fetched 988 MB (cache held only 0-byte placeholders) | FP16, INT8, NF4, 4 layer mixes |
| Pythia-410M | 0.41 B | fetched ~0.8 GB (not previously cached) | FP16, INT8, NF4, 7 layer mixes |

**Data.** ARC-Easy test (200 items, seed 42) + MMLU test (200 items, stratified over 57 subjects, seed 42). 0-shot multiple choice: prompt lists the question and choices, then `Answer:`; each choice is scored by the log-probability of its single letter token. Options are batched; only the last-token logits are computed (`logits_to_keep=1`), no generation. Accuracy is deterministic across repetitions (verified). Some long prompts hit the 384-token truncation cap (max observed sequence 385 tokens).

**Layer mixes.** A mixed model is built by loading an FP16 copy and an NF4 copy and swapping whole decoder blocks (first/middle/last 4-layer blocks) from the FP16 model into the NF4 model, then restoring. `mix8` = layers 0–3 and 20–23 FP16, middle NF4 (the "simple per-layer mix"). The planned first2/last2 `mix4` was dropped by a config-filter bug in the harness and is not in the data; the verdict does not depend on it.

**Energy protocol.** `nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu -lms 100` run as a timestamped subprocess; trapezoidal integration over the run window; idle baseline measured for 10 s before and after each session and subtracted (`E_net = E_total − P_idle·t`). The driver updates power at roughly 1 Hz on this laptop even though it is queried at 100 ms; a 15–106 s run yields 137–971 sample lines. 16-question warm-up per config; **5 reps** for FP16/INT8/NF4, **3 reps** for mixes; configuration order shuffled and round-robined within each rep to balance thermal drift. Workload: 400 queries/rep (200 ARC-Easy + 200 MMLU), all option forwards included, so J/query is the energy of answering one multiple-choice question.

**Caveats.** Desktop/browser used ~1 GB of the 6 GB throughout. The power signal is coarse (≈1 Hz update; 0.01–0.1 W display granularity), so CV, not the 100 ms label, is the honest precision statement. The pilot measures forward scoring, not generation; conclusions apply to short-answer inference at this scale.

---

## 2. Results

### Qwen2.5-0.5B (400 queries, 5 core reps / 3 mix reps)

| Config | FP16 layers | ARC % | MMLU % | All % | J/query (SD) | CV % | Latency ms | Mean W |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| INT8 | none | 15.0 | 23.0 | 19.0 | **2.380** (0.245) | 10.3 | 265.1 | 26.8 |
| FP16 | all | 13.0 | 18.5 | 15.8 | 2.702 (0.078) | 2.9 | 71.8 | 55.8 |
| mix8 | 0–3, 20–23 | 11.5 | 22.0 | 16.8 | 3.039 (0.132) | 4.4 | 96.8 | 49.4 |
| last4 | 20–23 | 11.5 | 21.0 | 16.2 | 3.103 (0.227) | 7.3 | 106.8 | 47.0 |
| NF4 | none | 17.5 | 21.5 | 19.5 | 3.107 (0.125) | 4.0 | 112.1 | 45.7 |
| first4 | 0–3 | 15.0 | 23.5 | 19.2 | 3.115 (0.071) | 2.3 | 103.9 | 48.0 |
| mid4 | 10–13 | 17.0 | 22.5 | 19.8 | 3.201 (0.047) | 1.5 | 111.8 | 46.7 |

**NF4 uses 15.0% more energy than FP16** (95% bootstrap CI [−19.4, −10.4]%), not less: it runs 1.56× slower while drawing only 10 W less. **INT8 is the only energy win** (11.9% less than FP16) but is 3.7× slower.

### Pythia-410M (400 queries, 5 core reps / 3 mix reps)

| Config | FP16 layers | ARC % | MMLU % | All % | J/query (SD) | CV % | Latency ms | Mean W |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| INT8 | none | 29.0 | 27.5 | 28.2 | **1.258** (0.098) | 7.8 | 145.2 | 27.9 |
| mix8 | 0–3, 20–23 | 26.5 | 26.5 | 26.5 | 1.982 (0.096) | 4.8 | 48.5 | 60.6 |
| all12 | 0–3, 10–13, 20–23 | 28.0 | 30.0 | 29.0 | 2.007 (0.177) | 8.8 | 46.1 | 63.4 |
| FP16 | all | 26.5 | 28.0 | 27.3 | 2.020 (0.034) | 1.7 | 37.6 | 73.6 |
| firstmid | 0–3, 10–13 | 27.0 | 29.5 | 28.2 | 2.035 (0.146) | 7.2 | 50.5 | 60.0 |
| midlast | 10–13, 20–23 | 28.0 | 27.0 | 27.5 | 2.116 (0.297) | 14.0 | 51.9 | 60.3 |
| first4 | 0–3 | 26.0 | 31.5 | 28.7 | 2.133 (0.103) | 4.8 | 51.7 | 60.8 |
| mid4 | 10–13 | 29.5 | 28.0 | 28.7 | 2.141 (0.203) | 9.5 | 55.4 | 58.1 |
| last4 | 20–23 | 23.0 | 28.5 | 25.8 | 2.223 (0.227) | 10.2 | 55.8 | 59.4 |
| NF4 | none | 27.5 | 30.0 | 28.7 | 2.258 (0.401) | 17.8 | 69.0 | 54.2 |

**NF4 uses 11.8% more energy than FP16** (CI [−24.7, +5.8]%, includes zero because NF4's CV is 17.8%). INT8 uses 37.7% less energy but is 3.9× slower. Mixes sit between NF4 and FP16 in energy (at best 1.9% below FP16, `mix8`), never below INT8.

### Measurement quality

- Core CV: Qwen 1.5–10.3%, Pythia 1.7–17.8%; every core config is **<20%** (K3 pass). Best precision is FP16 (1.7–2.9%); NF4 is the noisiest (4.0–17.8%).
- 137–971 power samples per repetition; idle 17.8 W (Qwen) / 19.2 W (Pythia), mean of pre/post baselines.
- **Spearman(energy, latency):** Qwen ρ = **0.071** (p = 0.88, n = 7); Pythia ρ = **0.394** (p = 0.26, n = 10). Energy is not a restatement of latency (K5 pass): INT8 is slow but efficient, NF4 is slow and inefficient.
- Temperatures: 69–86 °C; no throttling signature in the per-rep order. Configuration order was shuffled/round-robined.

### Accuracy is at the chance floor

Qwen 0-shot accuracy is **below** 25% chance (FP16 15.8% all; ARC 13.0%); Pythia is at chance (27.3% all; ARC 26.5%, MMLU 28.0%). Paired bootstrap over 400 questions gives differences vs FP16 whose CIs mostly include zero:

| Comparison (all, pt) | Qwen | Pythia |
|---|---:|---:|
| NF4 − FP16 | +3.75 [−0.50, +8.00] | +1.50 [−2.75, +5.75] |
| INT8 − FP16 | +3.25 [+0.75, +5.75] | +1.00 [−1.25, +3.25] |
| mix8 − FP16 | +1.00 [−2.25, +4.25] | −0.75 [−5.75, +4.00] |

The point spread across core configs is 4.5 pt (Qwen, ARC and MMLU) and 2.5 pt (Pythia), but the paired MDE at N = 400 is ≈4.5–6 pt. The nominal "gain" of quantized models over FP16 is therefore noise plus letter bias, not knowledge. **No accuracy effect of the size required by the pass criterion (≥3 pt) is certifiable in this pilot.** The Pythia 2³ factorial confirms this: all main effects (first +0.22, mid +0.47, last −0.72 pt) and all interactions (|interaction| ≤0.59 pt) have 95% CIs that include zero. Accuracy is additive within noise at the chance floor — but there is no signal to allocate.

### Additive energy model

Using the three 4-layer blocks as factors, `E = E_NF4 + d_first·n_first + d_mid·n_mid + d_last·n_last`:

| Model | Design | In-sample R² | Leave-one-out R² | Max residual (% of E_FP16) |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | 5 cells (NF4, first4, mid4, last4, mix8; 4 parameters) | 0.903 | **−0.994** | 2.7% |
| Pythia-410M | full 2³ (8 cells) | 0.877 | **0.506** | 5.8% |

The in-sample fit looks good, but **held-out additivity fails the 0.8 bar on both models** (K4 fail). The reason is visible in the noise: allocation differences are 0.05–0.2 J/query while the per-rep SD is 0.03–0.40 J/query. With 3–5 reps the additive model cannot predict a held-out mix better than measurement noise. This is exactly the "price of noise" problem G3 proposes to solve — and at 0.5 B on this GPU it is the binding constraint, not the knapsack.

---

## 3. Kill criteria against the brief

| Criterion | Qwen | Pythia | Result |
|---|---|---|---|
| **K1** energy headroom (FP16−NF4) | **−15.0%** [−19.4, −10.4] | **−11.8%** [−24.7, +5.8] | **FAIL** — 4-bit saves nothing; Qwen is significantly worse |
| **K2** accuracy spread | 4.5 pt (ARC/MMLU) but near/below chance, CIs include 0 | 2.5 pt, at chance, CIs include 0 | **FAIL in substance** — no certifiable spread; MDE ≈4.5–6 pt |
| **K3** CV of J/query | 10.3% max (core) | 17.8% max (core) | **PASS** (<20%) |
| **K4** held-out additive R² | −0.99 (5-cell design) | **0.506** (full 2³) | **FAIL** (<0.8) |
| **K5** Spearman(energy, latency) | 0.071 | 0.394 | **PASS** (<0.95) |
| **Pass** mix ≥3 pt over uniform at 0.8×FP16 | no config feasible at B | only uniform INT8 feasible | **FAIL** |

**Energy budget at B = 0.8 × E_FP16.** Qwen: B = 2.161 J/query; the lowest measured config is INT8 at 2.380 J/query, so **the feasible set is empty** — no uniform and no mix can meet the budget. Pythia: B = 1.616 J/query; the only feasible config is **uniform INT8** (1.258 J/query); no mix is feasible (cheapest mix 1.982 J/query). Consequently the required comparison — a per-layer mix at 0.8×FP16 energy — does not exist on either model. At matched measured energy, the best mix loses to uniform NF4 on Qwen (`mix8` 16.75% vs 19.5%, −2.75 pt) and is statistically indistinguishable on Pythia (`all12` 29.0% vs 28.75%, +0.25 pt, CI wide). There is no ≥3 pt allocation gain to certify.

---

## 4. Verdict and path

**KILL the static certified-allocation framing at this scale.** Two kill criteria fire unambiguously:

1. **K1:** NF4 — the configuration the layer-wise mix is built on — has *negative* energy headroom on both models (−15.0% and −11.8%). On this GPU, bitsandbytes 4-bit dequantization costs more time than it saves power. Only INT8 saves energy (11.9% / 37.7%), and INT8 is a *uniform* config: it already is the allocation, so there is nothing to allocate.
2. **K4:** held-out additive energy R² is 0.506 (Pythia, full factorial) and −0.99 (Qwen, incomplete design), both below 0.8. The per-layer knapsack cannot be certified at 3–5 repetitions; allocation differences are inside the measurement noise.

The pass condition is additionally unreachable: 0.8×FP16 is below the measured frontier on Qwen (no feasible config), and on Pythia only uniform INT8 is feasible (no mix). Accuracy is at the chance floor, so even a nominal mix gain cannot be separated from letter bias at N = 400 (MDE ≈4.5–6 pt).

**What survives.** The measurement apparatus is sound: core CV 1.5–10.3% (Qwen) and 1.7–17.8% (Pythia), idle-subtracted joules, and energy is *not* redundant with latency (ρ = 0.07/0.39, K5 pass). Two empirical facts are publishable as a measurement note: (i) at 0.5 B on an RTX 3060, 4-bit NF4 is an energy *loss* and a latency loss, while uniform INT8 is the only energy win (12–38%) at 3.7–3.9× latency; (ii) allocation-level energy differences (≤6% of FP16) are below 5-rep noise, so a certified per-layer allocation at this scale would need far more repetitions than a pilot can afford.

**Do not proceed** to the full allocation study on 0.5 B/3060. If the team wants to keep G3 alive, the minimum viable pivot is: (a) ≥1.5 B models and/or larger batch, where 4-bit kernels can actually reduce energy; (b) 20–30 reps per config to certify 10% differences; (c) a few-shot accuracy protocol with above-chance signal; or (d) the measurement/benchmark paper ("CarbonQuant-4") described in `03_g3_quantization.md` §6.4. The carbon-intensity-conditioned profile repair is not indicated by this pilot (K5 passes; the problem is headroom, not carbon timing).

---

## 5. Files

- `green_computing/pilots/03_g3_pilot.md` — this report.
- `green_computing/pilots/ledger_g3_pilot.csv` — 17 config rows (model, config, layers, accuracy, J/query, SD, CV, latency, power, feasibility, headroom).
- `green_computing/pilots/queue/DONE_G3` — flag.

Raw per-rep JSON and harness live in `C:\Users\atiku\AppData\Local\Temp\opencode\g3\` (`qwen_full.json`, `pythia_full.json`, `analysis.json`, `run_g3.py`, `analyze_g3.py`).
