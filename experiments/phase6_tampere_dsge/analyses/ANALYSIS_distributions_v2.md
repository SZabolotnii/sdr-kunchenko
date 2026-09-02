# Phase 6 — Distribution diagnostics v2 (corrected after Codex review)

**Дата:** 2026-04-29.
**Status:** ANALYSIS_distributions.md (v1) **superseded** через два high-severity defects, виявлені Codex adversarial review. Цей файл — v2 з виправленою методологією і чесним перерахунком mechanism claim.

## Що було виправлено

### F1. Per-window stability — random subsample with seed (не temporal prefix)

**Раніше:** `windows[:200]` бере **перші 200 за часом** після sort. При burst non-stationarity (вмикання, прогрів передавача) ранні windows tail behavior відрізняється від пізніх.

**Тепер:** `rng.choice(N, size=200, replace=False, seed=0)` — random subsample з recorded seed. JSON містить `subsample_seed` і `subsample_indices_total_count`.

### F2. Bootstrap CIs — window-level (block) bootstrap

**Раніше:** sample-level bootstrap на flatten'ed pool — resample individual samples з 4096-sample windows. Ігнорує within-window correlation і normalize-z coupling. CIs занижені особливо для tail-sensitive γ₅, γ₆.

**Тепер:** window-level bootstrap — кожна replicate resample whole windows (with replacement), pools their samples, обчислює cumulants. Respects within-window correlation. R=100, seed=42.

### F3. JSON ↔ compute consistency

**Раніше:** `n_windows: 500` reported while internally using 200.
**Тепер:** `n_windows_loaded: 500` AND `n_subsampled_windows: 200` AND `subsample_seed: 0` — all sync'ed.

## Comparison v1 vs v2 CI widths (γ₆ on Re)

| Band | Class | γ₆ point | sample-CI half (v1) | block-CI half (v2) | Widening |
|---|---|---|---|---|---|
| 2.4 GHz | Mavic Pro | +1.377 | 0.123 | 0.215 | 1.7× |
| 2.4 GHz | Inspire 2 | -0.020 | 0.150 | 0.050 | 0.3× |
| 2.4 GHz | **Parrot Disco** | **+5.081** | 4.384 | **7.585** | 1.7× |
| 2.4 GHz | Yuneec H | +3.963 | 3.297 | 2.731 | 0.8× |
| 5.8 GHz | Phantom 4 Pro+ | +0.054 | 0.129 | 0.106 | 0.8× |
| 5.8 GHz | Inspire 2 | +0.007 | 0.104 | 0.098 | 0.9× |
| 5.8 GHz | Matrice 210 | +0.053 | 0.136 | 0.059 | 0.4× |
| 5.8 GHz | Yuneec H | +1.236 | 1.201 | 1.948 | 1.6× |

CI widening averages **~1.0×** — менше, ніж я хвилювалась (5–10× scope). Чому: v1 sample-level бутстреп вже мав subsampling (100K), а within-window correlation для z-нормалізованих 4096-семплових windows на скінченних cumulants не настільки extreme. **Однак** Parrot Disco γ₆ CI = ±7.6 — точкова оцінка +5.08 з такою uncertainty практично **не reliable**.

## Виправлений mechanism claim — pairwise CI separation

Раніше: "γ₆ spread 5.10 на 2.4 ГГц vs 1.23 на 5.8 ГГц" — це наратив через raw point spreads, який помилково припускав, що всі точкові оцінки однаково надійні.

Тепер: **pairwise CI separation** — конкретно які пари класів дискримінативні через γ₆.

### 2.4 GHz (block-CI 95%)

| Pair | Result |
|---|---|
| Mavic Pro [+1.151, +1.581] vs Inspire 2 [-0.064, +0.037] | **NON-OVERLAP** ★ |
| Mavic Pro vs Parrot Disco [-0.143, +15.027] | overlap (Parrot CI massive) |
| Mavic Pro vs Yuneec H [+2.155, +7.618] | **NON-OVERLAP** ★ |
| Inspire 2 vs Parrot Disco | overlap (Parrot CI) |
| Inspire 2 vs Yuneec H | **NON-OVERLAP** ★ |
| Parrot Disco vs Yuneec H | overlap (Parrot CI) |

**3 з 6 пар non-overlap.** γ₆ дискримінативно для 3 пар, особливо **intra-DJI Mavic vs Inspire**. Parrot Disco γ₆ point estimate +5.08 має CI [-0.14, +15.03] — широка uncertainty робить її **не reliable**.

### 5.8 GHz (block-CI 95%)

| Pair | Result |
|---|---|
| Phantom 4 Pro+ vs Inspire 2 | overlap |
| Phantom 4 Pro+ vs Matrice 210 | overlap |
| Phantom 4 Pro+ vs Yuneec H | overlap |
| Inspire 2 vs Matrice 210 | overlap |
| Inspire 2 vs Yuneec H | overlap |
| Matrice 210 vs Yuneec H | overlap |

**0 з 6 пар non-overlap.** γ₆ **не дискримінативно** на 5.8 GHz.

### Те ж саме для γ₄:

На 2.4 GHz:
- Mavic [-0.20, -0.13] vs Inspire [-0.02, +0.00]: **NON-OVERLAP**
- Mavic vs Parrot [+0.11, +0.46]: **NON-OVERLAP**
- Mavic vs Yuneec [-0.43, -0.18]: overlap (close)
- Inspire vs Parrot: **NON-OVERLAP**
- Inspire vs Yuneec: **NON-OVERLAP**
- Parrot vs Yuneec: **NON-OVERLAP**

На 5.8 GHz:
- 3 DJI all in [-0.08, -0.03] approximately — overlap
- Yuneec [-0.005, +0.251] — partial overlap with DJI

**Висновок:** γ₄ значно **більш discriminative** ніж γ₆ — на 2.4 GHz ~5/6 пар non-overlap, на 5.8 GHz близько 0/6 (3 DJI overlap, Yuneec partially separable).

## Виправлений mechanism claim для manuscript

> "На 2.4 GHz cumulant CIs (window-level bootstrap, R=100) показують **clear class separation**: γ₄ розрізняє 5/6 пар класів, γ₆ розрізняє 3/6 пар з найкращим separation на intra-DJI Mavic Pro vs Inspire 2 (CIs [+1.15, +1.58] і [-0.06, +0.04] **не overlap**). На 5.8 GHz усі CI overlapping — γ₄ і γ₆ **не дискримінативні**.
>
> Це pattern узгоджується з distribution shape: на 2.4 GHz pooled distributions strongly non-Gaussian (γ₄ point spread = 0.58, plus γ₆ point estimates 0...+5 з CIs widely separated for some pairs); на 5.8 GHz близько до Gaussian (γ₄ spread 0.17, γ₆ all CIs near 0).
>
> Caveat: Parrot Disco γ₆ point estimate +5.08 має block-CI [-0.14, +15.03] — uncertainty ~1.5× самої оцінки. Це означає, що **specific value γ₆=+5 для Parrot не reliable**, але класова приналежність CI (positive і wide) ще ловить дискримінативний signal через γ₄."

## Як цей результат relate до empirical findings ([cumulant_4vs6 і frac DSGE])

Empirical results **untouched** Codex critique:

| Empirical claim | Залежить від distribution diagnostics CI? |
|---|---|
| cumulant_order4 на 2.4 GHz: Δ=+4.32 п.п. (3/3 sig) | **NO** — paired bootstrap on test predictions |
| cumulant_order6 на 2.4 GHz: Δ=+5.04 п.п. (3/3 sig) | **NO** |
| cumulant_order4 на 5.8 GHz: Δ=+1.30 п.п. (3/3 sig) | **NO** |
| frac DSGE на 2.4 GHz: Δ=+5.18 п.п. (p<0.001) | **NO** |
| Hybrid variants comparison | **NO** |

Усі main empirical claims **не залежать** від distribution diagnostics CIs. Distribution diagnostics — це **physical mechanism explanation**, не empirical evidence.

**Empirical → mechanism story** consistent post-correction:
- Cumulants працюють reproducibly на обидвох діапазонах ← empirical, untouched
- На 2.4 GHz cumulants CIs показують значну class separation у γ₄ (5/6) і γ₆ (3/6) ← mechanism with corrected CIs
- На 5.8 GHz cumulants CIs всі overlap, але cumulants все одно працюють у класифікаторі ← seemingly contradictory, але resolution: spectral вже дає 97%, cumulants додають 1.3 п.п. через subtle differences що **stable per-class** але **CI-overlapping** через small magnitude
- На 5.8 GHz frac DSGE НЕ виграє ← mechanism: pooled distributions close to Gaussian → reconstruction approach lacks tail-shape signal to fit

## Артефакти

- `run_distribution_diagnostics_v2.py` — corrected script
- `results/phase6_distribution_diagnostics_v2.json` — full numeric output з sync'ed reporting
- `figures/fig8_cumulants_v2.pdf` — fig 8 with corrected window-level CIs

## Дякую Codex review

Виявлені defects були реальні. v1 ANALYSIS_distributions.md і fig8 — superseded. v2 інтегрується у manuscript Phase 6 із пунктом про methodology correction (показує наукову integrity, не недолік).

Methodology lesson: для **time series higher-order moment estimation з multi-window data**:
- ALWAYS use block bootstrap (resample whole windows, not samples)
- ALWAYS document subsample seed and size in JSON
- ALWAYS check JSON ↔ compute consistency

Записую урок у CLAUDE.md проекту.
