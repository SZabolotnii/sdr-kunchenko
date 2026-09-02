# Tables Draft — Exp 3 DSGE Modulation Classification Article

> Source: `article_numbers.md` (verified against `verification_report.txt`)
> All numbers confirmed from `exp3_classification_results.json` and CSV files.
> Generated: Tier 2 tasks 2.1.1–2.1.5

---

## Table I: Notation and Symbols (§II)

| Symbol | Definition | Dimension |
|--------|-----------|-----------|
| x | Signal block (N samples) | ℝᴺ |
| F(x) | Basis function matrix from x | ℝᴺˣˢ |
| K_c | Coefficient matrix for class c | ℝˢˣᴺ |
| MSED_c(x) | Per-class reconstruction error | ℝ₊ |
| F_c | Basis function matrix from training blocks of class c | ℝᴹˣˢ |
| B_c | Target matrix (autoreconstruction) | ℝᴹˣᴺ |
| φᵢ(x) | i-th PATP basis function | ℝᴺ |
| α | PATP basis parameter | [0, 1] |
| pᵢ(α) | PATP exponent for i-th function | ℝ₊ |
| γ₃, γ₄ | Normalized skewness and excess kurtosis | ℝ |
| s | Basis order (number of functions) | ℕ |
| λ | Ridge regularization parameter | ℝ₊ |

### Symbol–Equation Consistency Check (Task 2.1.1)

| Symbol | Used in Equation | Status |
|--------|-----------------|--------|
| x | Eq.(1) MSED_c(x) = ‖F(x)·K_c − x‖² | ✓ |
| F(x) | Eq.(1) and Eq.(4) φᵢ(x) | ✓ |
| K_c | Eq.(1) and Eq.(2) K_c = (F_cᵀF_c + λI)⁻¹F_cᵀB_c | ✓ |
| MSED_c(x) | Eq.(1) and Eq.(3) f(x) = [log MSED_1(x), log MSED_2(x)] | ✓ |
| F_c | Eq.(2) ridge regression | ✓ |
| B_c | Eq.(2) target matrix | ✓ |
| φᵢ(x) | Eq.(4) φᵢ(x) = sign(x)·|x|^{pᵢ(α)} | ✓ |
| α | Eq.(4) and Eq.(5) pᵢ(α) | ✓ |
| pᵢ(α) | Eq.(5) pᵢ(α) = 1/i + (4−i−3/i)·α + (2i−4+2/i)·α² | ✓ |
| γ₃, γ₄ | Table II dataset characteristics | ✓ |
| s | Eq.(4) i = 2,...,s+1 | ✓ |
| λ | Eq.(2) ridge parameter | ✓ |

**Result**: All 12 symbols in Table I are used in the equations of §II. No missing or inconsistent symbols found.

---

## Table II: Dataset Characteristics (§IV)

| Parameter | Synthetic Polygauss | Real RadioML Residual |
|-----------|--------------------|-----------------------|
| Name | fpv_telemetry_polygauss_qpsk | fpv_realnoise_bpsk_qpsk |
| Class | synthetic_polygauss | real_radioml_residual |
| n_blocks | 100 000 | 100 000 |
| block_size N | 1024 | 1024 |
| SNR-buckets | {−5, 0, +5, +10, +15} dB | {−5, 0, +5, +10, +15} dB |
| γ̂₃ (μ±σ) | +0.08 ± 0.66 | +0.00 ± 0.09 |
| γ̂₄ (μ±σ) | **+0.50 ± 1.36** | **−0.46 ± 0.35** |
| Noise type | Polygauss (K=3 components) | Real RadioML2018 residual |
| Modulation | QPSK | BPSK + QPSK |
| Character | Leptokurtic (heavy tails) | Platykurtic (sub-Gaussian) |

### Verification (Task 2.1.2)

All values match design.md §3 Table II exactly:
- γ̂₃: +0.08±0.66 (polygauss) and +0.00±0.09 (realnoise) ✓
- γ̂₄: +0.50±1.36 (polygauss) and −0.46±0.35 (realnoise) ✓
- n_blocks: 100 000 for both ✓
- block_size: 1024 for both ✓
- SNR-buckets: {−5, 0, +5, +10, +15} dB for both ✓

---

## Table III: Accuracy by Feature Sets and Classifiers (§V)

Mean accuracy over 5 SNR-buckets {−5, 0, +5, +10, +15} dB.

### Part A: Summary (per-SNR best basis)

| Feature Set | Logistic Reg. | Random Forest | SVM-RBF | Note |
|-------------|--------------|--------------|---------|------|
| Spectral-only | — | **0.921** | — | Baseline |
| DSGE-only (best basis) | — | **0.757** | — | ≥70% target ✓ |
| Hybrid (DSGE+Spectral, best basis) | — | **0.938** | — | p < 0.001 vs spectral |

### Part B: Full DSGE-only — All Classifiers × All Bases (mean over 5 SNR)

| Basis | Logistic Reg. | Random Forest | SVM-RBF |
|-------|--------------|--------------|---------|
| poly_s3 | 0.724 | **0.751** | 0.726 |
| poly_s4 | 0.725 | **0.753** | 0.725 |
| frac_s3 | 0.693 | **0.704** | 0.716 |
| trig_s4 | 0.619 | **0.654** | 0.660 |
| robust_s3 | 0.626 | **0.690** | 0.675 |

> Best single-basis: poly_s4 RF = 0.753. Per-SNR best (poly_s3 wins at SNR=+5,+10,+15) = 0.757.

### Part C: Full Hybrid — All Classifiers × All Bases (mean over 5 SNR)

| Basis | Logistic Reg. | Random Forest | SVM-RBF |
|-------|--------------|--------------|---------|
| poly_s3 | 0.887 | **0.935** | 0.914 |
| poly_s4 | 0.888 | **0.936** | 0.914 |
| frac_s3 | 0.892 | **0.930** | 0.918 |
| trig_s4 | 0.876 | **0.933** | 0.926 |
| robust_s3 | 0.879 | **0.935** | 0.934 |

> Best single-basis: poly_s4 RF = 0.936. Per-SNR best = 0.938.

### LR per-basis detail (DSGE-only, mean over 5 SNR):
- poly_s3: (0.723+0.716+0.7105+0.712+0.7565)/5 = **0.7236**
- poly_s4: (0.7355+0.719+0.71+0.7065+0.7525)/5 = **0.7247**
- frac_s3: (0.697+0.6735+0.693+0.6775+0.725)/5 = **0.6932**
- trig_s4: (0.593+0.5975+0.6385+0.6045+0.6605)/5 = **0.6188**
- robust_s3: (0.6655+0.5455+0.626+0.5685+0.725)/5 = **0.6261**

### SVM-RBF per-basis detail (DSGE-only, mean over 5 SNR):
- poly_s3: (0.7195+0.7165+0.718+0.712+0.765)/5 = **0.7262**
- poly_s4: (0.719+0.717+0.7185+0.712+0.7595)/5 = **0.7252**
- frac_s3: (0.695+0.705+0.6915+0.724+0.7665)/5 = **0.7164**
- trig_s4: (0.665+0.663+0.666+0.643+0.6625)/5 = **0.6599**
- robust_s3: (0.7005+0.6765+0.6275+0.6395+0.73)/5 = **0.6748**

---

## Table IV: Pass Criteria — Spec §6.8 (§V)

| ID | Metric | Value | Threshold | Pass |
|----|--------|-------|-----------|------|
| dsge_70pct | DSGE-only RF best-basis acc ≥ 0.70 (mean over SNR) | **0.7574** | 0.70 | ✓ PASS |
| hybrid_beats_baselines | Hybrid ≥ Spectral AND ≥ DSGE on ALL SNR | **10/10** | 10/10 | ✓ PASS |
| hybrid_vs_spectral_significant | Paired bootstrap p-value (hybrid − spectral > 0) | **0.000** (< 0.001) | 0.05 | ✓ PASS |
| patp_alpha_interior | PATP α* strictly interior on ALL SNR | **5/5** | 5/5 | ✓ PASS |
| patp_dominates_poly | PATP@α* ≥ poly-s3 (DSGE-LR), per SNR count | **0/5** | 2/5 | ✗ FAIL |
| | **Pass rate** | **4/5 (80%)** | | |

### Notes (from verification_report.txt):
- `dsge_70pct`: per-SNR best DSGE acc: {−5: 0.7665, 0: 0.743, +5: 0.743, +10: 0.740, +15: 0.7945}
- `hybrid_beats_baselines`: hybrid mean=0.938, spectral mean=0.921, dsge mean=0.757
- `hybrid_vs_spectral_significant`: observed mean diff = 0.0165
- `patp_alpha_interior`: α* per SNR: {−5: 0.6, 0: 0.6, +5: 0.6, +10: 0.6, +15: 0.6}
- `patp_dominates_poly`: patp mean=0.710, poly mean=0.724 — PATP uses i∈{2,3,4}, poly-s3 uses i∈{1,2,3} (different functional support)

---

## Table V: PATP α-Sweep — α* per SNR (§V)

| SNR (dB) | α* | Accuracy | AUC | Interior? |
|----------|-----|---------|-----|-----------|
| −5 | **0.60** | 0.709 | 0.802 | ✓ |
| 0 | **0.60** | 0.695 | 0.785 | ✓ |
| +5 | **0.60** | 0.702 | 0.786 | ✓ |
| +10 | **0.60** | 0.694 | 0.771 | ✓ |
| +15 | **0.60** | 0.752 | 0.859 | ✓ |

> Raw values (before rounding): {−5: (0.7085, 0.8018), 0: (0.6945, 0.7853), +5: (0.702, 0.7863), +10: (0.694, 0.7707), +15: (0.7525, 0.8592)}

### Boundary α reference values (mean over 5 SNR):
- α = 0.0 (fractional basis): acc = **0.672**
- α = 0.5 (degenerate linear): acc = **0.526** ← catastrophic
- α = 1.0 (polynomial basis): acc = **0.602**
- α* = 0.6 (PATP optimum): acc = **0.710** ← best

> α*=0.6 is strictly interior: 0 < 0.6 < 1, and 0.6 ≠ 0.5 (degenerate). Stable across ALL 5 SNR buckets.

---

## Summary: Tier 2 Checkpoint

### Task 2.3.1 — All 5 figures exist:
| Figure | File | Size | Status |
|--------|------|------|--------|
| Fig. 1 | fig3_dsge_vs_spectral.pdf | 20,653 bytes | ✓ EXISTS |
| Fig. 2 | fig3_basis_comparison_bar.pdf | 16,577 bytes | ✓ EXISTS |
| Fig. 3 | fig3_patp_alpha_sweep.pdf | 21,824 bytes | ✓ EXISTS |
| Fig. 4 | fig3_alpha_optimal_per_snr.pdf | 18,809 bytes | ✓ EXISTS |
| Fig. 5 | fig3_accuracy_vs_snr.pdf | 19,067 bytes | ✓ EXISTS |

All 5 required figures exist. (6th figure `fig3_classifier_comparison.pdf` = 20,220 bytes also exists but is not used in the article per design.md §2.)

### Task 2.3.2 — All 5 tables filled with real numbers: ✓
- Table I: 12 symbols with definitions and dimensions ✓
- Table II: All dataset parameters filled ✓
- Table III: Full 5×3 matrix for DSGE-only and Hybrid, all classifiers ✓
- Table IV: All 5 pass criteria with exact numerical values ✓
- Table V: All 5 SNR rows with accuracy and AUC at α*=0.6 ✓

### Task 2.3.3 — Numbers match verification_report.txt: ✓
| Table | Key value | verification_report.txt | Match |
|-------|-----------|------------------------|-------|
| IV | dsge_70pct = 0.7574 | 0.7574 | ✓ |
| IV | hybrid_beats_baselines = 10/10 | 10.0000 | ✓ |
| IV | bootstrap p = 0.000 | 0.0000 | ✓ |
| IV | patp_alpha_interior = 5/5 | 5.0000 | ✓ |
| IV | patp_dominates_poly = 0/5 | 0.0000 | ✓ |
| V | α* = 0.6 on all SNR | {-5:0.6, 0:0.6, +5:0.6, +10:0.6, +15:0.6} | ✓ |
| III | DSGE RF mean = 0.757 | dsge mean=0.757 | ✓ |
| III | Hybrid RF mean = 0.938 | hybrid mean=0.938 | ✓ |
| III | Spectral RF mean = 0.921 | spectral mean=0.921 | ✓ |

**All numbers in tables are consistent with verification_report.txt. No discrepancies found.**
