# G2 pilot — energy-optimal retrieval depth

**Date:** 8 November 2026.
**Verdict: MARGINAL.** The pre-registered *static* \(k^\star\) claim is **KILLED** at the 2 pp EM margin: \(k^\star=32\) (boundary) and \(P(k^\star\text{ optimal})=0.61<0.8\). The energy effect itself is large and precise, the optimal-depth identity holds at \(k=2\), and the adaptive-oracle gain is 78.8%. A powered rerun (\(n\approx500\)–1000) with a pre-specified accuracy margin, or a pivot to query-conditional depth, is warranted. Under a 5 pp EM margin the pilot is a **PASS** at \(k=4\): 69.3% savings, \(N^\star=876\) queries.
**Hardware:** RTX 3060 Laptop 6 GB (driver 621.01, max power limit 130 W; observed 9.5–86.4 J/query GPU, idle 15.6 W), Ryzen 7 5800H, 32 GB RAM, Windows.
**Software:** Python 3.12 (`C:\ProgramData\anaconda3\python.exe`), Ollama `qwen3.5:4b` (3.3 GB Q4, 100% GPU, 4096 ctx), rank-bm25, pynvml.

---

## 1. What was run

### Data and retriever

| Item | Value |
|---|---|
| Questions | 200 SQuAD v1.1 dev items from `research_audit/round9/prep_i08b.json`, stratified over 7 wh-types (28–30 each) |
| Corpus | 20,958 unique SQuAD contexts (256-word truncation, as in round9), chunked to **45,651** 70-word passages (mean 53.4 words) |
| Retriever | BM25 Okapi (`rank_bm25`), lowercase alphanumeric tokens, index built once (1.6 s), top-32 |
| Depths | \(k\in\{0,1,2,4,8,16,32\}\) |
| Main queries | 200 × 7 = **1,400** generations; \(k\) order shuffled within each question |
| Repeats | 3 × 20 identical queries at \(k=8\) for noise/drift |

Retrieval recall rises monotonically: the gold context is in the top-\(k\) for 63.0% (\(k=1\)), 71.0% (2), 77.5% (4), 81.0% (8), 83.5% (16), 88.0% (32); the gold answer string is in the retrieved text for 58.0%, 67.5%, 75.5%, 80.5%, 84.0%, 88.5%. The reader, not the retriever, is the accuracy bottleneck.

### Reader choice

`qwen3.5:4b` via Ollama, 3.3 GB Q4, 100% GPU, `num_ctx=4096`, greedy (`temperature=0`, `top_k=1`, `seed=42`), `num_predict=24`, `think=false`. The cached Qwen2.5-1.5B-Instruct HF reader was the fallback but was not used: the 4B fits entirely on the 6 GB card, is already local (no download, no paid API), and the main sweep ran at 0.68 s/query. The pilot's binding constraint is accuracy power, so the stronger reader was preferred over the faster one.

Prompt is fixed across depths: a system message ("answer using only the provided passages … if the answer is not in the passages, reply exactly: unknown"), then numbered passages, the question, and `Answer:`. No per-\(k\) prompt tuning.

**Correctness.** Normalized EM is primary (lowercase, strip articles/punctuation, collapse whitespace); lenient span match (gold span appears in a short prediction) and token F1 ≥ 0.5 are secondary. Strict EM penalizes answers such as "Denver linebacker Von Miller" for gold "Von Miller"; the lenient/F1 curves are reported because generative readers fail this way.

### Energy protocol

- **Sensor:** NVML `power.draw` sampled every **100 ms** by `pynvml` (the same sensor `nvidia-smi --query-gpu=power.draw` exposes). 11,377 samples over the run, no sampler errors. `power.draw` was available, so no utilization × TDP proxy was needed.
- **Sensor resolution caveat:** the 100 ms samples are not 100 ms of independent information. On this laptop the driver updates `power.draw` coarsely (G3 measured ~1 Hz on the same host). The 100 ms cadence is therefore a sampling label, not a 100 ms precision claim; absolute joules rely on the 20-query block calibration and the repeat CV, and per-query values are used for pairing/bootstrap rather than absolute precision.
- **Per-query:** trapezoidal integration of GPU power over the HTTP call, minus the pre-run idle baseline (15.56 W, 20 s, CV 10.4%). Post-run idle was 17.63 W (CV 15.9%); using it instead lowers every \(J\) by ~4% nearly uniformly and changes E ratios by <1%.
- **Block calibration:** per-query windows miss the ~100 ms power ramp, most at short depths. After the sweep, 20 back-to-back queries at each depth (same first 20 items) were integrated as a block and idle-subtracted; the per-query GPU integrals were scaled to the block total. Scale factors: 1.041 (0), 1.100 (1), 1.188 (2), 1.128 (4), 1.118 (8), 1.081 (16), 1.073 (32). Both calibrated and raw-window E are saved; the minimum moves between \(k=1\) and \(k=2\) (<1.1%), and the separation from \(k\ge8\) is unchanged.
- **CPU retrieval:** `pyRAPL` is unavailable on this Windows/AMD host. BM25 time is measured (mean 0.149 s/query) and converted at a documented **20 W** single-thread active-above-idle proxy = **3.11 J/query**. This is an upper bound; at 10–15 W the retrieval term would be 1.6–2.3 J, lowering E by <1.5%. Retrieval is a small share; the GPU is measured directly.
- **Prefix-cache control:** Ollama reuses KV prefixes. A \(k=32\) request following \(k=16\) reused the first 16 passages and cut prefill from 710 ms to 247 ms. Every request therefore carries a unique 10-hex nonce at the start of the user message, forcing a full prefill. Without this, the depth-energy curve is confounded by request order.
- **Order/thermal:** \(k\) randomized within each question; no concurrent GPU jobs (G3 had written `DONE_G3` before this run started). Temperature rose 71.9 → 88.4 °C, SM clock 1657 → 1003 MHz, mean power 82.5 → 54.2 W. Energy (power × time) was stable: repeat CV median 3.80%, max 9.24%.

---

## 2. Results

### 2.1 The curve

| \(k\) | \(A_{\text{EM}}\) | \(A_{\text{len}}\) | F1 | \(J_{\text{ret}}\) | \(J_{\text{gpu}}\) | \(J_{\text{tot}}\) | \(E_{\text{EM}}\) | \(E_{\text{len}}\) | ret. share | context share | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 0.00 | 9.50 | 9.50 | ∞ | ∞ | 0.0% | 0.0% | 0.197 |
| 1 | 0.395 | 0.540 | 0.514 | 3.11 | 14.87 | 17.98 | **45.53** | **33.30** | 17.3% | 30.0% | 0.295 |
| 2 | 0.460 | 0.615 | 0.593 | 3.11 | 18.05 | 21.16 | **46.01** | 34.41 | 14.7% | 36.3% | 0.332 |
| 4 | 0.520 | 0.675 | 0.652 | 3.11 | 21.99 | 25.10 | 48.27 | 37.18 | 12.4% | 48.8% | 0.432 |
| 8 | 0.530 | 0.670 | 0.658 | 3.11 | 30.60 | 33.71 | 63.61 | 50.32 | 9.2% | 63.4% | 0.608 |
| 16 | 0.535 | 0.695 | 0.682 | 3.11 | 47.66 | 50.77 | 94.90 | 73.05 | 6.1% | 78.1% | 0.976 |
| 32 | 0.570 | 0.705 | 0.698 | 3.11 | 86.44 | 89.55 | **157.10** | 127.02 | 3.5% | 85.1% | 1.732 |

\(E(k)=J(k)/A(k)\) in joules per correct answer. 95% bootstrap CIs over questions: \(E(1)=45.5\,[38.6,55.2]\), \(E(2)=46.0\,[39.9,53.9]\), \(E(4)=48.3\,[42.2,56.4]\), \(E(8)=63.6\,[55.8,73.5]\), \(E(16)=94.9\,[83.2,109.4]\), \(E(32)=157.1\,[139.6,179.6]\).

**The energy effect is large:** \(E(32)/E(1)=3.45\) and \(E(32)/E(2)=3.41\) — a **245% / 241%** relative effect, not a flat curve. The unconstrained minimum is \(k=1\)–2; \(E(1)\) and \(E(2)\) differ by 1.1% and their CIs overlap. The energy model is essentially affine: \(J(k)=15.74+2.283k\) J, \(R^2=0.9990\); the per-passage GPU cost is 2.283 J, the fixed cost 15.74 J.

**The accuracy side is flat and noisy.** \(A_{\text{EM}}\) rises from 0.395 at \(k=1\) to 0.570 at \(k=32\): +17.5 pp, but only +5.0 pp from \(k=4\) and +3.5 pp from \(k=16\). Paired differences vs \(k=32\): \(k=1\) −17.5 pp [−24.5,−11.0], \(k=2\) −11.0 [−16.5,−5.5], \(k=4\) −5.0 [−10.5,+0.5], \(k=8\) −4.0 [−9.0,+1.0], \(k=16\) −3.5 [−7.5,+0.5]. McNemar \(p\)-values vs \(k=32\): 0.174 (\(k=4\)), 0.337 (8), 0.287 (16), 0.0006 (2). **No depth below 16 is significantly less accurate than 32**, but none is certifiably within 2 pp either.

The stylized coverage model \(A(k)=1-(1-A_0)(1-p)^k\) fits poorly (\(A_0=0.324\), \(p=0.0218\), \(R^2=0.332\)); the empirical accuracy curve is a jump at \(k=1\), a plateau at 8–16, then a final rise at 32. The affine **energy** model is excellent, but the theorem's global unimodality assumption on \(A\) is rejected. The empirical \(E\) curve is still U-shaped on this grid.

### 2.2 The optimal-depth identity

At an interior optimum the depth-elasticity of energy equals that of accuracy, \(kJ'(k)/J(k)=kA'(k)/A(k)\). Using the fitted affine \(J\) and the empirical finite-difference \(A'\):

| \(k\) | \(\varepsilon_J\) | \(\varepsilon_A\) | gap | 95% bootstrap CI |
|---:|---:|---:|---:|---|
| 2 | 0.225 | 0.181 | **+0.044** | **[−0.050, +0.128]** |
| 4 | 0.367 | 0.090 | +0.278 | [+0.209, +0.348] |
| 8 | 0.537 | 0.019 | +0.518 | [+0.464, +0.572] |
| 16 | 0.699 | 0.050 | +0.649 | [+0.584, +0.711] |
| 32 | 0.823 | 0.123 | +0.700 | [+0.572, +0.840] |

**The identity holds at \(k=2\)** (gap CI includes zero) and fails monotonically beyond it. At \(k=32\) the energy elasticity is 6.7× the accuracy elasticity: by the theory's own diagnostic, 32 passages is gross over-retrieval. At \(k=1\) the gap is −0.456 because the derivative spans \(A(0)=0\), a boundary artifact; the discrete bracket \(E(0)=\infty>E(1)<E(2)\) still makes \(k=1\) the grid optimum.

This is the strongest theory-side result in the pilot: the identity is not merely U-shape talk; it picks out the interior optimum and flags the over-retrieval at high \(k\).

### 2.3 Where the energy goes

Retrieval-only (BM25 CPU proxy) is **3.5%** of total at \(k=32\) and 17.3% at \(k=1\). The context-side share, \(S(k)=(J_{\text{ret}}+J_{\text{pre}})/J\), estimated from the 2.283 J/passage GPU slope, is **85.1% at \(k=32\)**, 78.1% (16), 63.4% (8), 48.8% (4), 36.3% (2), 30.0% (1). The green lever is **prefill of retrieved passages**, not the BM25 fetch. Conflating the two would understate the lever by an order of magnitude. The retrieval-only share being <5% at high \(k\) is not a negligible split in the sense of K4; the context-side split is large and grows with \(k\).

### 2.4 Certification, adaptivity, and the accuracy margin

Bootstrap (3,000 paired resamples over questions):

- **Unconstrained:** \(k^\star=1\) with \(P=0.544\), \(k=2\) with 0.368, \(k=4\) with 0.088; \(P(\{1,2\})=0.912\), span 2.
- **Primary 2 pp EM gate** (\(A_{\min}=A(32)-0.02=0.550\)): \(k^\star=32\) with \(P=0.613\); 16 with 0.135, 4 with 0.127, 8 with 0.126. **Not certifiable** (K3).
- **Exact \(A(32)\) gate:** \(k^\star=32\) with \(P=0.901\).
- **0.98 max gate:** \(k^\star=32\) with \(P=0.783\).

The static \(k^\star\) output of the pre-registered procedure is therefore **32 (boundary), with only 61% bootstrap support**. This is the K2/K3 failure. It is a margin/power failure, not a flat-curve failure: the 2 pp floor sits inside the paired accuracy CI of \(k=16\) [−7.5,+0.5] pp.

**Margin sensitivity (EM).** The optimal depth is the accuracy floor, not an accuracy-free property of the energy curve:

| Accuracy margin below max | \(k^\star\) | \(A(k^\star)\) | savings vs \(k=32\) | bootstrap \(P(k^\star)\) |
|---:|---:|---:|---:|---:|
| 0 pp | 32 | 0.570 | 0% | 0.879 |
| 2 pp | 32 | 0.570 | 0% | 0.613 |
| 3 pp | 32 | 0.570 | 0% | 0.425 |
| 4 pp | 8 | 0.530 | 59.5% | 0.210 |
| 5 pp | 4 | 0.520 | **69.3%** | 0.537 |
| 6 pp | 4 | 0.520 | 69.3% | 0.650 |
| 8 pp | 4 | 0.520 | 69.3% | 0.737 |
| 10 pp | 4 | 0.520 | 69.3% | 0.598 |

**Lenient/F1 sensitivity.** Under the span-match rule at a 2 pp margin, \(k^\star=16\) with **42.5% savings** at a 1.0 pp accuracy loss (\(P=0.383\)). Under F1 ≥ 0.5 at 2–3 pp, \(k^\star=16\), 42.5% savings, \(P=0.625\)–0.634. At a 5 pp lenient margin, \(k^\star=4\) with 70.7% savings. The qualitative result (never 32) is robust; the certified location is not.

**Adaptive oracle.** Matching the static \(k=32\) accuracy set, the oracle that picks the cheapest correct depth per query costs **33.35 J/correct vs 157.10** — a **78.8%** gain. The union-accuracy oracle (66.5% vs 57.0%) costs 30.60 J/correct, an 80.5% gain. This is an upper bound; no learned policy was trained. It is well above the 15% K6 threshold and is the main reason the direction survives.

**Certificate cost.** The 1,400-query sweep cost **49.6 kJ**. Break-even \(N^\star=C_{\text{cert}}/(A(k^\star)[E(32)-E(k^\star)])\): 876 queries for \(k=4\), 1,000 for \(k=8\), 1,489 for \(k=16\). At 10k deployment queries the \(k=4\) saving is 566 kJ vs 49.6 kJ of calibration (8.8%); at 100k it is 0.9%. **The certificate is cheap; the accuracy margin, not the certificate, is the binding constraint.** This supports the decision-theoretic framing, but only after the margin is fixed.

### 2.5 Measurement quality

- Repeat CV at \(k=8\): **median 3.80%, max 9.24%** (3 × 20 identical queries; mean J by rep 24.39/24.42/24.59, drift <1%). K7 passes.
- 11,377 power samples; mean power 61.9 W; total sweep energy 49.6 kJ; mean 35.4 J/query over all 1,400.
- Block calibration factors 1.04–1.19; the largest correction is at \(k=2\) where the per-query window is shortest relative to the ramp.
- Thermal drift is real (clock −39%, temperature +16 °C) but \(k\) was randomized within each question and the repeat CV is low; the block calibration ran at the throttled steady state, which is the conservative condition for a deployment.
- The 20 W CPU-retrieval proxy is the only non-measured term; it is ≤17% of J at \(k=1\) and 3.5% at \(k=32\).

**Limitations.** One reader (4B), one retriever (BM25), one closed corpus (SQuAD), one GPU. Strict EM is the primary and it is the metric that makes the 2 pp gate fail; lenient/F1 give an interior \(k=16\) at 42.5% savings. The reader converts ~64% (EM) / ~80% (lenient) of the available evidence at \(k=32\); a stronger reader could shift the curve. The 12% of questions whose gold passage is outside the top-32 cap accuracy at 0.88 and add a final jump at \(k=32\) that a full-corpus retriever would not have.

---

## 3. Kill criteria against the brief

| Criterion | Result | Evidence |
|---|---|---|
| **K1** relative energy effect <10% | **PASS (not flat)** | \(E(32)/E(1)=3.45\) (**245%** relative); \(E(32)/E(2)=3.41\) (241%) |
| **K2** \(k^\star\) at boundary | **FAIL** | pre-registered 2 pp EM gate: \(k^\star=32\), savings 0% |
| **K3** uncertifiable CI | **FAIL** | \(P(k^\star=32)=0.61<0.8\); unconstrained \(P(k=1)=0.54\), \(P(\{1,2\})=0.91\) but no single \(k\ge0.8\) |
| **K4** split <5% and flat | **PASS** | context-side share 85.1% at 32, 36.3% at 2; retrieval-only 3.5–17.3%. The lever is prefill, not BM25 |
| **K5** \(k^\star\) spans >1 depth | **FAIL (pivot)** | gated span 3 (4,8,16,32); unconstrained span 2 (1,2,4) |
| **K6** adaptive gain <15% | **PASS (large)** | matched oracle gain **78.8%**; union 80.5% |
| **K7** CV >20% | **PASS** | repeat CV 3.80% median / 9.24% max |
| **Pass** interior \(k^\star\), ≥20% savings, cheap certificate | **Conditional** | 2 pp EM: no. 5 pp EM: \(k=4\), 69.3%, \(N^\star=876\). 2 pp lenient/F1: \(k=16\), 42.5%, \(P=0.38\)–0.63 |

The literal pre-registered static verdict is **KILL** (K2 + K3). The pilot verdict is **MARGINAL** because the boundary is induced by a 2 pp accuracy floor that sits inside the paired CI, the unconstrained energy optimum is interior, and the adaptive gain is 79%. The brief's K5 explicitly says to pivot, not to kill, when \(k^\star\) is unstable.

---

## 4. Verdict and path

**Do not write the paper as "we measured the curve and picked a static \(k^\star\)."** That claim fails at the pre-registered 2 pp EM margin: the energy-optimal depth is 32 only because the accuracy floor is at the maximum, and the bootstrap cannot certify it. The 3.41× energy effect is real, but the accuracy cost of using \(k\le16\) is 3.5–5 pp, not the 2 pp the pre-registration demanded.

**What survives, and is worth pursuing:**

1. **The affine energy model is excellent** (\(R^2=0.9990\)) and the **context-side share is large** (36–85%). The green lever is prefill of retrieved tokens, not BM25. That is a clean, publishable measurement.
2. **The optimal-depth identity holds at \(k=2\)** (elasticity gap +0.044, CI includes 0) and fails by +0.70 at \(k=32\). This is a genuine validation of the theory's diagnostic, even though the global coverage model is rejected.
3. **The adaptive-oracle gain is 78.8%.** Query-conditional depth is where the energy is; the static population \(k^\star\) is the wrong object.
4. **The certificate is cheap** (49.6 kJ; break-even 876–1,489 queries). The decision-theoretic framing \(N^\star\) is viable.

**Recommended next step (a powered rerun, not a full paper yet):**

- Pre-specify the accuracy margin with the deployment: 2 pp EM is unattainable at this reader; 5 pp gives \(k=4\) with 69.3% savings. Report both; let the margin be a policy input, not a post-hoc choice.
- Use **\(n\ge500\)–1000** questions to certify a 2–5 pp margin (the current MDE is ~4–5 pp). The 200-question pilot was sized for the energy curve, which is precise; the accuracy side is underpowered.
- Keep EM as primary but add a **human- or span-validated correctness** rule; the lenient/F1 curves already give an interior \(k=16\) at 42.5% savings and \(P=0.63\). Do not let a strict string match decide the green verdict.
- Train a simple **query-conditional policy** (BM25 top score, score gap, question type) and measure the fraction of the 79% oracle gain it captures. This is the pivot the data support.
- Add a second reader (Qwen2.5-1.5B) and a distractor/multi-hop set (HotpotQA) before claiming external validity; SQuAD's closed corpus and the 4B reader are the main threats.
- If the PI requires the static 2 pp claim as originally framed, the honest verdict is **KILL** and the measurement should fold into G5. If the 5 pp margin or the adaptive pivot is acceptable, the pilot is a **conditional PASS**.

**Bottom line.** The energy-depth tradeoff is real and steep (3.4×), the theory's identity works at the interior optimum, and adaptivity is worth 79% in oracle terms. The static certified \(k^\star\) at 2 pp is not supported at \(n=200\). Spend the next run on power and a policy, not on another curve.

---

## 5. Files

- `green_computing/pilots/02_g2_pilot.md` — this report.
- `green_computing/pilots/ledger_g2_pilot.csv` — 7 depth rows (accuracy, joules, E, split, elasticities, bootstrap, certificate, notes).
- `green_computing/pilots/queue/DONE_G2` — flag.
- Raw data and harness: `C:\Users\atiku\AppData\Local\Temp\opencode\g2work\` (`runs/raw_main_ollama.csv` 1,460 rows, `runs/blocks_main_ollama.json`, `runs/analysis_main_ollama.json`, `runs/numbers_main_ollama.json`, `g2_run.py`, `g2_analysis.py`, `g2_numbers.py`, `make_ledger_g2.py`).
