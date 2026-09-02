# Article Numbers — Exp 3 DSGE Modulation Classification

> Source: `exp3_classification_results.json` (git sha: d317ad8b8a0cb22081c0ce44fae9069bda694dfc)
> Verification report: `verification_report.txt` — **4/5 pass**
> Generated for: Tier 1 checkpoint (tasks 1.3.1–1.3.4)

---

## 1. Mean Accuracy by Feature Set (RF classifier, mean over 5 SNR)

| Feature Set | Basis | RF Accuracy | Note |
|-------------|-------|-------------|------|
| Spectral-only | — | **0.921** | Baseline |
| DSGE-only | poly_s4 (fixed) | **0.753** | Mean over 5 SNR |
| DSGE-only | per-SNR best | **0.757** | Best basis per SNR |
| Hybrid | poly_s4 (fixed) | **0.936** | Mean over 5 SNR |
| Hybrid | per-SNR best | **0.938** | Best basis per SNR |

> **Design.md §3 Table III** states: Spectral=0.921, DSGE=0.757, Hybrid=0.938 — these match the **per-SNR best** values.
> Note: poly_s4 fixed gives DSGE=0.753 and Hybrid=0.936.

---

## 2. Per-SNR DSGE-only RF Accuracy (best basis per SNR)

| SNR (dB) | Best Basis | RF Accuracy | ≥ 0.70? |
|----------|-----------|-------------|---------|
| −5 | poly_s4 | **0.7665** | ✓ |
| 0 | poly_s4 | **0.743** | ✓ |
| +5 | poly_s3 | **0.743** | ✓ |
| +10 | poly_s3 | **0.740** | ✓ |
| +15 | poly_s3 | **0.7945** | ✓ |
| **Mean** | | **0.7574** | ✓ |

> **Property 1 CONFIRMED**: All 5 SNR buckets have DSGE-only RF acc ≥ 0.70. Min = 0.740 > 0.70.

---

## 3. Per-SNR DSGE-only RF Accuracy (poly_s4 fixed)

| SNR (dB) | RF Accuracy |
|----------|-------------|
| −5 | 0.7665 |
| 0 | 0.743 |
| +5 | 0.7365 |
| +10 | 0.728 |
| +15 | 0.7925 |
| **Mean** | **0.7533** |

---

## 4. Full Basis × Classifier Table — DSGE-only (mean over 5 SNR)

| Basis | LR | RF | SVM-RBF |
|-------|-----|-----|---------|
| poly_s3 | 0.724 | **0.751** | 0.717 |
| poly_s4 | 0.722 | **0.753** | 0.717 |
| frac_s3 | 0.678 | **0.704** | 0.703 |
| trig_s4 | 0.618 | **0.654** | 0.661 |
| robust_s3 | 0.626 | **0.690** | 0.675 |

> Computed as mean over SNR={−5, 0, +5, +10, +15}.
> **poly_s4 RF = 0.753** is the best single-basis result.
> Per-SNR best gives 0.757 (poly_s3 wins at SNR=+5, +10, +15).

### LR per-basis detail (mean over 5 SNR):
- poly_s3: (0.723+0.716+0.7105+0.712+0.7565)/5 = 0.7236
- poly_s4: (0.7355+0.719+0.71+0.7065+0.7525)/5 = 0.7247
- frac_s3: (0.697+0.6735+0.693+0.6775+0.725)/5 = 0.6932
- trig_s4: (0.593+0.5975+0.6385+0.6045+0.6605)/5 = 0.6188
- robust_s3: (0.6655+0.5455+0.626+0.5685+0.725)/5 = 0.6261

### SVM-RBF per-basis detail (mean over 5 SNR):
- poly_s3: (0.7195+0.7165+0.718+0.712+0.765)/5 = 0.7262
- poly_s4: (0.719+0.717+0.7185+0.712+0.7595)/5 = 0.7252
- frac_s3: (0.695+0.705+0.6915+0.724+0.7665)/5 = 0.7164
- trig_s4: (0.665+0.663+0.666+0.643+0.6625)/5 = 0.6599
- robust_s3: (0.7005+0.6765+0.6275+0.6395+0.73)/5 = 0.6748

---

## 5. Full Basis × Classifier Table — Hybrid (mean over 5 SNR)

| Basis | LR | RF | SVM-RBF |
|-------|-----|-----|---------|
| poly_s3 | 0.887 | **0.934** | 0.914 |
| poly_s4 | 0.888 | **0.936** | 0.914 |
| frac_s3 | 0.892 | **0.930** | 0.918 |
| trig_s4 | 0.876 | **0.933** | 0.926 |
| robust_s3 | 0.879 | **0.935** | 0.934 |

### RF Hybrid per-basis detail (mean over 5 SNR):
- poly_s3: (0.9355+0.9335+0.9335+0.94+0.931)/5 = 0.9347
- poly_s4: (0.9365+0.9325+0.937+0.939+0.933)/5 = 0.9356
- frac_s3: (0.931+0.927+0.933+0.9345+0.9255)/5 = 0.9302
- trig_s4: (0.938+0.9255+0.93+0.942+0.934)/5 = 0.9339
- robust_s3: (0.9405+0.9345+0.938+0.939+0.9235)/5 = 0.9351

---

## 6. Spectral-only per-SNR (RF)

| SNR (dB) | RF Accuracy |
|----------|-------------|
| −5 | 0.9295 |
| 0 | 0.916 |
| +5 | 0.9245 |
| +10 | 0.9245 |
| +15 | 0.912 |
| **Mean** | **0.9213** |

---

## 7. PATP α-sweep at α*=0.6 per SNR

| SNR (dB) | α* | Accuracy | AUC |
|----------|-----|---------|-----|
| −5 | **0.60** | 0.7085 | 0.8018 |
| 0 | **0.60** | 0.6945 | 0.7853 |
| +5 | **0.60** | 0.702 | 0.7863 |
| +10 | **0.60** | 0.694 | 0.7707 |
| +15 | **0.60** | 0.7525 | 0.8592 |

> Design.md Table V rounds to: (0.709, 0.802), (0.695, 0.785), (0.702, 0.786), (0.694, 0.771), (0.752, 0.859) — all match ✓

---

## 8. PATP Boundary α Values (mean over 5 SNR)

| α | Mean Accuracy | Design.md value |
|---|--------------|-----------------|
| 0.0 (fractional) | **0.672** | 0.67 ✓ |
| 0.5 (degenerate) | **0.526** | 0.53 ✓ |
| 1.0 (polynomial) | **0.602** | 0.60 ✓ |

### Per-SNR detail for boundary α:

**α=0.0:**
- SNR=-5: 0.675, SNR=0: 0.645, SNR=+5: 0.660, SNR=+10: 0.6475, SNR=+15: 0.732
- Mean: 0.6719

**α=0.5:**
- SNR=-5: 0.500, SNR=0: 0.4965, SNR=+5: 0.4955, SNR=+10: 0.4965, SNR=+15: 0.642
- Mean: 0.5261

**α=1.0:**
- SNR=-5: 0.5865, SNR=0: 0.593, SNR=+5: 0.5995, SNR=+10: 0.6015, SNR=+15: 0.627
- Mean: 0.6015

---

## 9. Bootstrap Test

| Metric | Value | Design.md |
|--------|-------|-----------|
| Observed mean diff (hybrid − spectral) | **0.0165** | 0.0165 ✓ |
| Bootstrap p-value | **0.0** (< 0.001) | p < 0.001 ✓ |

---

## 10. Pass Criteria Summary (from verification_report.txt)

| ID | Metric | Value | Threshold | Pass |
|----|---------|-------|-----------|------|
| dsge_70pct | DSGE-only RF best-basis acc ≥ 0.70 (mean) | 0.7574 | 0.70 | ✓ |
| hybrid_beats_baselines | Hybrid ≥ Spectral AND ≥ DSGE on all SNR | 10/10 | 10/10 | ✓ |
| hybrid_vs_spectral_significant | Bootstrap p-value | 0.0 | 0.05 | ✓ |
| patp_alpha_interior | PATP α* strictly interior on ALL SNR | 5/5 | 5/5 | ✓ |
| patp_dominates_poly | PATP@α* ≥ poly-s3 (DSGE-LR), per SNR count | 0/5 | 2/5 | ✗ |
| **Pass rate** | | **4/5** | | |

---

## 11. Discrepancy Report (Task 1.3.1)

### Minor discrepancies between design.md §3 and real data:

| Location | Design.md value | Real data value | Status |
|----------|----------------|-----------------|--------|
| Table III header | "Best basis: poly s=4" | Per-SNR best varies (poly_s3 wins at SNR=+5,+10,+15) | ⚠ Clarification needed |
| Table III DSGE-only RF | 0.757 | 0.757 (per-SNR best) OR 0.753 (poly_s4 fixed) | ✓ (per-SNR best) |
| Table III Hybrid RF | 0.938 | 0.938 (per-SNR best) OR 0.936 (poly_s4 fixed) | ✓ (per-SNR best) |
| Table V SNR=-5 acc | 0.709 | 0.7085 | ✓ (rounded) |
| Table V SNR=0 acc | 0.695 | 0.6945 | ✓ (rounded) |
| Table V SNR=+15 acc | 0.752 | 0.7525 | ✓ (rounded) |
| Table V SNR=-5 AUC | 0.802 | 0.8018 | ✓ (rounded) |
| §3 note α=0 | acc≈0.67 | 0.672 | ✓ |
| §3 note α=0.5 | acc=0.53 | 0.526 | ✓ |
| §3 note α=1 | acc=0.60 | 0.602 | ✓ |
| Table IV dsge_70pct value | 0.7574 | 0.7574 | ✓ |
| Per-SNR DSGE best (task 1.2.2) | {-5:0.767, 0:0.743, +5:0.743, +10:0.740, +15:0.795} | {-5:0.7665, 0:0.743, +5:0.743, +10:0.740, +15:0.7945} | ✓ (rounded) |

### Key finding:
The design.md Table III header says "Best basis для DSGE: poly s=4" but the 0.757 mean accuracy
is achieved using **per-SNR best basis** (poly_s3 wins at SNR=+5, +10, +15).
With poly_s4 fixed, the mean is 0.753. This is consistent with the verification report
which reports "per-SNR best DSGE acc" = 0.7574.

**Recommendation**: Clarify in Table III that 0.757 is per-SNR best, and add a footnote
that poly_s4 gives 0.753 (single-basis best). Both values are defensible.

---

## 12. Property 1 Verification (Task 1.3.4)

**Claim**: DSGE-only RF acc ≥ 0.70 on ALL 5 SNR buckets.

| SNR | Best-basis acc | ≥ 0.70? |
|-----|---------------|---------|
| −5 dB | 0.7665 | ✓ |
| 0 dB | 0.743 | ✓ |
| +5 dB | 0.743 | ✓ |
| +10 dB | 0.740 | ✓ |
| +15 dB | 0.7945 | ✓ |

**RESULT: Property 1 PASSES. All 5/5 SNR buckets have acc ≥ 0.70. Min = 0.740.**
