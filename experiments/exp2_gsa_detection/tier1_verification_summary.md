# Tier 1 Verification Summary
**Spec:** GSA-CUSUM з PATP-базисом (IEEE TSP)  
**Date:** 2025-07-14  
**Status:** COMPLETE — всі артефакти знайдено, числові значення витягнуто

---

## 1.1 Наявність та цілісність артефактів

### 1.1.1 PDF-рисунки у `figures/` (polygauss)
**Статус: ✅ PASS — 6/6 файлів присутні**

| Файл | Присутній |
|------|-----------|
| `fig2_J_convergence.pdf` | ✅ |
| `fig2_add_vs_snr.pdf` | ✅ |
| `fig2_auc_vs_snr.pdf` | ✅ |
| `fig2_far_compliance.pdf` | ✅ |
| `fig2_alpha_optimal_per_snr.pdf` | ✅ |
| `fig2_patp_alpha_sweep.pdf` | ✅ |

### 1.1.2 PDF-рисунки у `figures_realnoise/` (realnoise)
**Статус: ✅ PASS — 6/6 файлів присутні**

| Файл | Присутній |
|------|-----------|
| `fig2_J_convergence.pdf` | ✅ |
| `fig2_add_vs_snr.pdf` | ✅ |
| `fig2_auc_vs_snr.pdf` | ✅ |
| `fig2_far_compliance.pdf` | ✅ |
| `fig2_alpha_optimal_per_snr.pdf` | ✅ |
| `fig2_patp_alpha_sweep.pdf` | ✅ |

### 1.1.3 `results/verification_report.txt` — polygauss
**Статус: ✅ PASS — 3/6 pass (підтверджено)**

```
Pass rate: 3/6
✗ far_ok          | max FAR = 0.060 on energy @ SNR=5.0
✓ add_monotone_s  | s=1 mean ADD=990, s=3 mean ADD=573
✓ J_monotone      | diffs min = 0.0004
✗ patp_dominates  | PATP wins 0/5 SNRs
✓ alpha_interior  | interior α: 4/5, values: [1.0, 0.5, 0.8, 0.7, 0.9]
✗ gsa_beats_energy| max gain = 0.000
```

### 1.1.4 `results_realnoise/verification_report.txt` — realnoise
**Статус: ✅ PASS — 3/6 pass (підтверджено)**

```
Pass rate: 3/6
✗ far_ok          | max FAR = 0.043 on gsa_poly_s1 @ SNR=3.0
✓ add_monotone_s  | s=1 mean ADD=inf, s=3 mean ADD=554
✓ J_monotone      | diffs min = 0.0000
✗ patp_dominates  | PATP wins 1/5 SNRs
✓ alpha_interior  | interior α: 3/5, values: [0.85, 0.85, 0.9, 1.0, 1.0]
✗ gsa_beats_energy| max gain = 0.001
```

### 1.1.5 CSV-файли
**Статус: ⚠️ PARTIAL — realnoise CSV доступні напряму; polygauss CSV зберігаються у git-lfs (pointer files)**

| Файл | Статус |
|------|--------|
| `results/exp2_per_detector.csv` | ⚠️ git-lfs pointer (дані витягнуто з `exp2_gsa_results.json`) |
| `results/exp2_patp_alpha_sweep.csv` | ⚠️ git-lfs pointer (дані витягнуто з `exp2_gsa_results.json`) |
| `results_realnoise/exp2_per_detector.csv` | ✅ доступний |
| `results_realnoise/exp2_patp_alpha_sweep.csv` | ✅ доступний |

> Примітка: polygauss CSV є git-lfs pointers. Числові дані успішно витягнуто з `results/exp2_gsa_results.json`, який містить повні результати.

---

## 1.2 Числові значення для таблиць

### 1.2.1 Mean ADD по детекторах — Polygauss
**Джерело:** `results/exp2_gsa_results.json`  
**Статус: ✅ COMPLETE**

| Детектор | SNR=-5 | SNR=-2 | SNR=0 | SNR=+3 | SNR=+5 | **Mean** |
|----------|--------|--------|-------|--------|--------|----------|
| energy | 339.7 | 193.0 | 151.0 | 67.4 | 42.2 | **158.6** |
| gsa_poly_s1 | 970.0 | 976.5 | 1015.0 | 981.6 | 1008.5 | **990.3** |
| gsa_poly_s2 | 811.1 | 699.8 | 575.6 | 399.9 | 283.8 | **554.1** |
| gsa_poly_s3 | 830.8 | 731.8 | 593.8 | 427.2 | 280.5 | **572.8** |
| gsa_poly_s4 | 806.6 | 696.8 | 748.3 | 318.9 | 118.3 | **537.8** |
| gsa_frac_s3 | ∞ | ∞ | 947.4 | 586.6 | 469.1 | **667.7** (finite only) |
| gsa_patp_s3 (α_opt) | 895.4 | 976.5 | 912.2 | 474.9 | 403.1 | **732.4** |

### 1.2.2 Mean ADD по детекторах — Realnoise
**Джерело:** `results_realnoise/exp2_per_detector.csv` та `exp2_gsa_results.json`  
**Статус: ✅ COMPLETE**

| Детектор | SNR=-5 | SNR=-2 | SNR=0 | SNR=+3 | SNR=+5 | **Mean** |
|----------|--------|--------|-------|--------|--------|----------|
| energy | 494.6 | 360.3 | 204.6 | 114.6 | 74.6 | **249.7** |
| gsa_poly_s1 | 747.2 | 787.3 | ∞ | 757.8 | 808.4 | **775.2** (finite only) |
| gsa_poly_s2 | 817.4 | 712.1 | 579.5 | 385.4 | 278.0 | **554.5** |
| gsa_poly_s3 | 817.3 | 711.3 | 580.1 | 385.7 | 277.7 | **554.4** |
| gsa_poly_s4 | 673.1 | 500.5 | 201.7 | 42.0 | 33.0 | **290.1** |
| gsa_frac_s3 | 1019.0 | 722.9 | 810.5 | 514.7 | 521.2 | **717.7** |
| gsa_patp_s3 (α_opt) | 564.6 | 697.4 | 299.6 | 139.5 | 78.4 | **355.9** |

### 1.2.3 α_opt per SNR — Polygauss
**Джерело:** `results/exp2_gsa_results.json` (patp_sweep)  
**Статус: ✅ CONFIRMED — відповідає design.md**

| SNR (dB) | α_opt | Interior? | best ADD |
|----------|-------|-----------|----------|
| −5 | **1.0** | Boundary | 895.4 |
| −2 | **0.5** | Interior | 976.5 |
| 0 | **0.8** | Interior | 912.2 |
| +3 | **0.7** | Interior | 474.9 |
| +5 | **0.9** | Interior | 403.1 |

Design.md очікувало: `{1.0, 0.5, 0.8, 0.7, 0.9}` → **ПІДТВЕРДЖЕНО ✅**

### 1.2.4 α_opt per SNR — Realnoise
**Джерело:** `results_realnoise/exp2_patp_alpha_sweep.csv` та `exp2_gsa_results.json`  
**Статус: ✅ CONFIRMED — відповідає design.md**

| SNR (dB) | α_opt | Interior? | best ADD |
|----------|-------|-----------|----------|
| −5 | **0.85** | Interior | 564.6 |
| −2 | **0.85** | Interior | 697.4 |
| 0 | **0.90** | Interior | 299.6 |
| +3 | **1.00** | Boundary | 139.5 |
| +5 | **1.00** | Boundary | 78.4 |

Design.md очікувало: `{0.85, 0.85, 0.90, 1.00, 1.00}` → **ПІДТВЕРДЖЕНО ✅**

### 1.2.5 Max FAR по детекторах
**Статус: ✅ CONFIRMED**

| Датасет | Max FAR | Детектор | SNR | Design.md | Match? |
|---------|---------|----------|-----|-----------|--------|
| Polygauss | **0.060** | energy | +5 dB | 0.060 | ✅ |
| Realnoise | **0.043** | gsa_poly_s1 | +3 dB | 0.043 | ✅ |

### 1.2.6 J(s) значення та монотонність
**Статус: ✅ CONFIRMED — 0 порушень на обох датасетах**

**Polygauss J(s) по SNR:**

| SNR | J(s=1) | J(s=2) | J(s=3) | J(s=4) | Монотонна? |
|-----|--------|--------|--------|--------|------------|
| −5 | 1.07e-6 | 0.0715 | 0.0719 | 0.0723 | ✅ |
| −2 | 2.70e-8 | 0.2256 | 0.2277 | 0.2325 | ✅ |
| 0 | 6.68e-7 | 0.6299 | 0.6357 | 0.7400 | ✅ |
| +3 | 4.66e-9 | 2.2237 | 2.2816 | 2.6922 | ✅ |
| +5 | 3.83e-6 | 5.7909 | 5.7987 | 16.126 | ✅ |

**Realnoise J(s) по SNR:**

| SNR | J(s=1) | J(s=2) | J(s=3) | J(s=4) | Монотонна? |
|-----|--------|--------|--------|--------|------------|
| −5 | 3.06e-7 | 0.0728 | 0.0728 | 0.1122 | ✅ |
| −2 | 1.56e-6 | 0.2673 | 0.2674 | 0.4660 | ✅ |
| 0 | 1.97e-6 | 0.6804 | 0.6818 | 1.4754 | ✅ |
| +3 | 2.12e-6 | 2.7358 | 2.7361 | 10.653 | ✅ |
| +5 | 9.96e-7 | 6.2739 | 6.2773 | 38.294 | ✅ |

Мінімальна різниця J(s+1)−J(s): polygauss = 0.0004, realnoise = 0.0000 (≈0, але не від'ємна).

---

## 1.3 Checkpoint Tier 1

### 1.3.1 Підтвердження числових значень design.md §3

| Твердження | Статус | Примітка |
|-----------|--------|---------|
| Polygauss: Energy ADD ~280 | ⚠️ DISCREPANCY | Actual mean=158.6; design ~280 може бути для конкретного SNR або іншого запуску |
| Polygauss: GSA-poly s=1 ADD ~990 | ✅ OK | Actual 990.3 |
| Polygauss: GSA-poly s=2 ADD ~558 | ✅ OK | Actual 554.1 |
| Polygauss: GSA-poly s=3 ADD ~573 | ✅ OK | Actual 572.8 |
| Polygauss: GSA-poly s=4 ADD ~400 | ⚠️ DISCREPANCY | Actual mean=537.8; ~400 може бути для SNR=+3 (318.9) або іншого запуску |
| Realnoise: Energy ADD ~280 | ✅ OK | Actual 249.7 |
| Realnoise: GSA-poly s=1 ADD ∞ | ⚠️ PARTIAL | ADD=∞ лише при SNR=0; mean по 4 finite SNR = 775.2 |
| Realnoise: GSA-poly s=2 ADD ~555 | ✅ OK | Actual 554.5 |
| Realnoise: GSA-poly s=3 ADD ~554 | ✅ OK | Actual 554.4 |
| Realnoise: GSA-poly s=4 ADD ~170 | ⚠️ DISCREPANCY | Actual mean=290.1; ~170 може бути для SNR=+5 (33.0) або іншого запуску |
| Polygauss max FAR = 0.060 | ✅ OK | Підтверджено |
| Realnoise max FAR = 0.043 | ✅ OK | Підтверджено |
| Polygauss α_opt = {1.0, 0.5, 0.8, 0.7, 0.9} | ✅ OK | Підтверджено |
| Realnoise α_opt = {0.85, 0.85, 0.90, 1.00, 1.00} | ✅ OK | Підтверджено |
| J(s) монотонна, 0 порушень | ✅ OK | Підтверджено на обох датасетах |
| Pass rate 3/6 (обидва датасети) | ✅ OK | Підтверджено |

### 1.3.2 Відповідність тверджень статті verification_report.txt

| Твердження статті | verification_report | Суперечність? |
|------------------|---------------------|---------------|
| FAR перевищує target 0.01 (max 0.06 poly, 0.043 real) | ✗ far_ok | Немає — чесне представлення |
| ADD(s=3) < ADD(s=1) | ✓ add_monotone_s | Немає |
| J(s) монотонна (0 порушень) | ✓ J_monotone | Немає |
| PATP wins 0/5 SNR (poly), 1/5 (real) | ✗ patp_dominates | Немає — чесне представлення |
| α_opt interior на 4/5 SNR (poly), 3/5 (real) | ✓ alpha_interior | Немає |
| GSA AUC не перевищує Energy на 5% | ✗ gsa_beats_energy | Немає — чесне представлення |

**Висновок: жодне твердження статті не суперечить verification_report.txt ✅**

### 1.3.3 Точні значення ADD для GSA-frac та GSA-PATP (відсутні у чернетці)

**GSA-frac s=3:**

| Датасет | SNR=-5 | SNR=-2 | SNR=0 | SNR=+3 | SNR=+5 | Mean (finite) |
|---------|--------|--------|-------|--------|--------|---------------|
| Polygauss | ∞ | ∞ | 947.4 | 586.6 | 469.1 | **667.7** |
| Realnoise | 1019.0 | 722.9 | 810.5 | 514.7 | 521.2 | **717.7** |

**GSA-PATP s=3 (α_opt per SNR):**

| Датасет | SNR=-5 | SNR=-2 | SNR=0 | SNR=+3 | SNR=+5 | Mean |
|---------|--------|--------|-------|--------|--------|------|
| Polygauss | 895.4 (α=1.0) | 976.5 (α=0.5) | 912.2 (α=0.8) | 474.9 (α=0.7) | 403.1 (α=0.9) | **732.4** |
| Realnoise | 564.6 (α=0.85) | 697.4 (α=0.85) | 299.6 (α=0.90) | 139.5 (α=1.0) | 78.4 (α=1.0) | **355.9** |

---

## Зведена таблиця ADD (для Таблиці III статті)

### Polygauss

| Детектор | SNR=-5 | SNR=-2 | SNR=0 | SNR=+3 | SNR=+5 | Mean |
|----------|--------|--------|-------|--------|--------|------|
| Energy | 339.7 | 193.0 | 151.0 | 67.4 | 42.2 | 158.6 |
| GSA-poly s=1 | 970.0 | 976.5 | 1015.0 | 981.6 | 1008.5 | 990.3 |
| GSA-poly s=2 | 811.1 | 699.8 | 575.6 | 399.9 | 283.8 | 554.1 |
| GSA-poly s=3 | 830.8 | 731.8 | 593.8 | 427.2 | 280.5 | 572.8 |
| GSA-poly s=4 | 806.6 | 696.8 | 748.3 | 318.9 | 118.3 | 537.8 |
| GSA-frac s=3 | ∞ | ∞ | 947.4 | 586.6 | 469.1 | 667.7* |
| GSA-PATP s=3 | 895.4 | 976.5 | 912.2 | 474.9 | 403.1 | 732.4 |

*mean по 3 finite SNR

### Realnoise

| Детектор | SNR=-5 | SNR=-2 | SNR=0 | SNR=+3 | SNR=+5 | Mean |
|----------|--------|--------|-------|--------|--------|------|
| Energy | 494.6 | 360.3 | 204.6 | 114.6 | 74.6 | 249.7 |
| GSA-poly s=1 | 747.2 | 787.3 | ∞ | 757.8 | 808.4 | 775.2* |
| GSA-poly s=2 | 817.4 | 712.1 | 579.5 | 385.4 | 278.0 | 554.5 |
| GSA-poly s=3 | 817.3 | 711.3 | 580.1 | 385.7 | 277.7 | 554.4 |
| GSA-poly s=4 | 673.1 | 500.5 | 201.7 | 42.0 | 33.0 | 290.1 |
| GSA-frac s=3 | 1019.0 | 722.9 | 810.5 | 514.7 | 521.2 | 717.7 |
| GSA-PATP s=3 | 564.6 | 697.4 | 299.6 | 139.5 | 78.4 | 355.9 |

*mean по 4 finite SNR

---

## Розбіжності між design.md та реальними даними

> **Важливо для авторів:** Наступні значення у design.md §3 (Таблиця III) потребують уточнення перед написанням статті.

| Параметр | design.md | Реальні дані | Рекомендація |
|----------|-----------|--------------|--------------|
| Polygauss Energy mean ADD | ~280 | **158.6** | Уточнити: ~280 може бути для SNR=-5 (339.7) або іншого запуску. Використати 158.6 або per-SNR таблицю. |
| Polygauss GSA-poly s=4 mean ADD | ~400 | **537.8** | Уточнити: ~400 може бути для SNR=+3 (318.9). Використати 537.8 або per-SNR таблицю. |
| Realnoise GSA-poly s=1 ADD | ∞ | **∞ лише при SNR=0** | Уточнити: ADD=∞ є детермінованим лише при SNR=0 (mean-zero сигнал). При інших SNR ADD≈775. |
| Realnoise GSA-poly s=4 mean ADD | ~170 | **290.1** | Уточнити: ~170 може бути для SNR=+5 (33.0) або іншого запуску. Використати 290.1 або per-SNR таблицю. |

**Рекомендація:** У Таблиці III статті використовувати per-SNR значення (не mean), щоб уникнути неоднозначності.

---

## Підсумок Tier 1

| Завдання | Статус |
|---------|--------|
| 1.1.1 Figures polygauss (6 PDF) | ✅ |
| 1.1.2 Figures realnoise (6 PDF) | ✅ |
| 1.1.3 verification_report polygauss 3/6 | ✅ |
| 1.1.4 verification_report realnoise 3/6 | ✅ |
| 1.1.5 CSV доступність | ⚠️ (polygauss через JSON) |
| 1.2.1 ADD polygauss витягнуто | ✅ |
| 1.2.2 ADD realnoise витягнуто | ✅ |
| 1.2.3 α_opt polygauss підтверджено | ✅ |
| 1.2.4 α_opt realnoise підтверджено | ✅ |
| 1.2.5 max FAR підтверджено | ✅ |
| 1.2.6 J(s) монотонність підтверджено | ✅ |
| 1.3.1 Числа design.md §3 перевірено | ✅ (з 4 розбіжностями) |
| 1.3.2 Твердження vs verification_report | ✅ (суперечностей немає) |
| 1.3.3 ADD для GSA-frac та GSA-PATP зафіксовано | ✅ |
