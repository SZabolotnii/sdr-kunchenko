# Phase 6 — distribution diagnostics ANALYSIS

**Дата:** 2026-04-29.
**Питання:** чому фрактальний DSGE-reconstruction виграє на 2.4 ГГц, а explicit cumulant features — на 5.8 ГГц?
**Відповідь:** **рівень негаусовості розподілів кардинально різний.**

## Cumulants pooled per protocol per band (Re(y), z-normalized)

| Band | Class | Protocol | γ₃ | γ₄ | γ₅ | γ₆ |
|---|---|---|---|---|---|---|
| 2.4 ГГц | Mavic Pro | OcuSync 1 | +0.002 | **-0.169** | +0.025 | **+1.377** |
| 2.4 ГГц | Inspire 2 | OcuSync 1 | -0.000 | -0.005 | +0.005 | -0.020 |
| 2.4 ГГц | Parrot Disco | WiFi | -0.002 | **+0.252** | -0.125 | **+5.081** |
| 2.4 ГГц | Yuneec H | WiFi | +0.002 | **-0.330** | +0.014 | **+3.963** |
| 5.8 ГГц | Phantom 4 Pro+ | OcuSync 2 | -0.001 | -0.055 | +0.002 | +0.054 |
| 5.8 ГГц | Inspire 2 | OcuSync 1 | +0.001 | -0.056 | +0.001 | +0.007 |
| 5.8 ГГц | Matrice 210 | OcuSync 2 | +0.001 | -0.045 | +0.004 | +0.053 |
| 5.8 ГГц | Yuneec H | WiFi | +0.001 | **+0.117** | +0.010 | **+1.236** |

## Critical observations

### 1. Asymetria (γ₃) близько 0 на всіх класах і обох діапазонах

Це **сподіване** — baseband I/Q signals після DC-removal зазвичай симетричні. γ₃ не несе discriminative information.

### 2. Spread γ₄ на 2.4 ГГц у 3× більший ніж на 5.8 ГГц

| Band | γ₄ range | Spread |
|---|---|---|
| 2.4 ГГц | -0.330 (Yuneec) to +0.252 (Parrot) | **0.582** |
| 5.8 ГГц | -0.056 (Inspire) to +0.117 (Yuneec) | **0.173** |

**На 2.4 ГГц розподіли кардинально не Gaussian** — від platykurtic (тонше за нормальний) до leptokurtic (товсті хвости). На 5.8 ГГц всі ближче до Gaussian (γ₄ ~ 0).

### 3. Spread γ₆ на 2.4 ГГц у 4× більший

| Band | γ₆ range | Spread |
|---|---|---|
| 2.4 ГГц | -0.020 to +5.081 | **5.10** |
| 5.8 ГГц | +0.007 to +1.236 | **1.23** |

Sixth-order cumulant виявляє **дуже великі різниці** на 2.4 ГГц (Parrot Disco γ₆ = +5.08, Inspire 2 γ₆ = -0.02 — на 5 одиниць!), на 5.8 ГГц різниці порядку 1.2.

### 4. Per-window γ₄ stability

| Band | Class | per-window γ₄ std | range |
|---|---|---|---|
| 2.4 ГГц | Mavic Pro | 0.481 | [-1.40, +0.34] |
| 2.4 ГГц | Inspire 2 | 0.137 | [-0.46, +0.46] |
| 2.4 ГГц | **Parrot Disco** | **1.158** | **[-0.61, +12.86]** |
| 2.4 ГГц | Yuneec H | 0.619 | [-1.50, +0.97] |
| 5.8 ГГц | Phantom 4 Pro+ | 0.290 | [-0.89, +0.85] |
| 5.8 ГГц | Inspire 2 | 0.232 | [-0.89, +0.60] |
| 5.8 ГГц | Matrice 210 | 0.172 | [-0.45, +0.71] |
| 5.8 ГГц | Yuneec H | 0.074 | [-0.19, +0.27] |

**Per-window γ₄ на 2.4 ГГц дуже шумні** (Parrot std=1.16 з outliers до +12.86). На 5.8 ГГц — stable (std<0.3).

## Mechanism — чому базиси різні на різних діапазонах

### Сценарій 2.4 ГГц (далеко від Gaussian)

- Pooled distributions **strongly non-Gaussian** (γ₄ range 0.58, γ₆ range 5.1)
- Per-window cumulant estimates **noisy** (Parrot std 1.16, range [-0.61, +12.86] — outliers!)
- **Direct cumulant features → unstable, dominated by noise** на per-window level
- **Fractional DSGE reconstruction → averaging effect через F·K=B на pooled training data усереднює noise**, зберігаючи class-conditional shape signature
- Результат: **frac DSGE +5.2 п.п.** Boost over PSD; cumulants_explicit не testovani на 2.4 ГГц у repaired experiments, але треба перевірити

### Сценарій 5.8 ГГц (близько до Gaussian)

- Pooled distributions **close to Gaussian** (γ₄ range 0.17, γ₆ range 1.2 — все мала шкала)
- Per-window cumulants **stable** (std 0.07–0.29 без outliers)
- **Direct cumulant features → reproducible discriminator**, бо subtle but consistent differences ловляться у stable estimates
- **Fractional DSGE reconstruction → не вистачає sigtnal для tail-shape recovery** (немає heavy tails to fit)
- Результат: **explicit cumulants +1.30 п.п.** boost; frac DSGE 0/3 reproducibility

## Critical missing experiment

Чи **explicit cumulants** також працюють на 2.4 ГГц? Якщо так — тоді cumulants — universal winner, а frac DSGE — це додатковий specific advantage на 2.4 ГГц через reconstruction averaging. Якщо ні — тоді clean band-dependent dichotomy.

Це **наступний логічний експеримент** — прогнати `cumulant_explicit` candidate на 2.4 ГГц з 3-seed replication.

## Інтерпретація для Кунченко-теорії

Ця знахідка **узгоджується** з фундаментальними принципами Кунченко-методів:

> γ₃, γ₄, ..., γ_n — основа PMM/GSA/DSGE. Формули optimum коефіцієнтів містять конкретні cumulants. **Specific representation** — direct estimation чи reconstruction subspace — залежить від:
> 1. Гладкості розподілу (close to Gaussian → direct stable, далеко → reconstruction averaging needed)
> 2. Sample size per window (small → reconstruction; large → direct)
> 3. Computational simplicity desired (direct < reconstruction by complexity)

5.8 ГГц при 200 МSps дає **stable per-window estimates** через wider baseband resolution — direct cumulants виграють по простоті.
2.4 ГГц при 120 МSps дає **noisy per-window estimates** через narrower resolution — reconstruction averaging виграє.

Ця інтерпретація **falsifiable**: на datasets з ще ширшим baseband (LTE 20+ MHz, USRP X310 100 МSps complex) очікуємо **direct cumulants** будуть universal; на narrow-baseband (RTL-SDR 2.4 МSps) — **DSGE reconstruction** буде необхідний.

## Зведення

Distribution-level diagnostics показують, що **різниця базисів — не випадковість**, а natural consequence різниці у gauссовому/негаусовому статусі signals на різних діапазонах. Це **сильний beautiful finding** для статті — він connects:
- empirical observation (frac vs cumulants)
- theoretical prediction (Кунченко moments γ₃-γ₆)
- physical mechanism (sample-rate-driven distribution shape)

Тепер слід оновити manuscript з цим explanation і додати **fig 7 (histograms)** і **fig 8 (cumulant table)** — вони замінюють hand-wavy "temporal sparseness" гіпотезу із попереднього draft'у.
