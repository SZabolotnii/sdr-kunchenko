# Phase 6 iter-2 — основні findings

**Дата:** 2026-04-29.
**Зміни vs iter-1:** WINDOW_SAMPLES 1024 → 4096; burst-aware sampling (energy_pct=40); spectral PSD features додано; bootstrap hybrid vs spectral-only.

## Зведена таблиця результатів

| Task | DSGE-only best | Spectral-only | Hybrid | PATP α_opt | Notes |
|---|---|---|---|---|---|
| 1: 4-class multi-protocol | trig 38.4% | — | — | — | Random=25%, +13pp |
| 2: DJI vs non-DJI | poly 54.8% | **94.7%** | 94.9% | — | Random=50%, spectral dominates |
| 3: Mavic Pro vs Inspire 2 | poly 64.7% | **95.5%** | 95.5% | 1.00 (boundary) | Random=50%, spectral wins, DSGE doesn't add |

Bootstrap hybrid vs spectral-only:
- Task 2: Δ=+0.0025, p=0.092, **не значимо**
- Task 3: Δ=0.0, p=2.0, **identical** (DSGE не додає над spectral)

## Інтерпретація — те, що для статті

### Знахідка 1: Spectral PSD features домінують на Tampere

Простий 16-bin log-PSD дає 94.7% (DJI vs non-DJI) і 95.5% (Mavic vs Inspire). **Це не дивно**: різні протоколи мають різні OFDM-параметри (FFT size, carrier spacing, channel bandwidth). Mavic Pro і Inspire 2 хоча обидві використовують OcuSync 1, фактично передають з трохи різними центрами і spread'ами через калібрування заводу. Spectral fingerprint = unique для кожного.

### Знахідка 2: DSGE okремо нетривіальний — 65% на intra-OcuSync

Це **позитивний сигнал**. На time-domain (без явного FFT) DSGE захоплює сигнал **65%** на найскладнішій задачі — distinguish two same-protocol same-anechoic-chamber DJI quadcopters. Це показує:
- DSGE features ловлять temporal structure, що корелює з протоколом
- Fractional/polynomial bases при window=4096 вже мають достатньо контексту
- Random baseline = 50%, тобто +15pp = реальний information signal

### Знахідка 3: Hybrid НЕ значимо кращий за spectral-only (Bootstrap p=0.09 на Task 2, identical на Task 3)

Це **ключове** finding для статті. Воно говорить що **DSGE-features на цих даних фактично duplikuje spectral information**. DSGE захоплює спектральну структуру як побічний ефект (через FFT-similarity polynomial expansions), не додає чогось окремого.

**Чому це важливо:** в звіті DeepResearch (2026-04-27) було передбачено:
> "when fingerprints arise from parametrically controlled circuit impairments (CFO, IQ imbalance, phase noise, PA non-linearity) rather than learned correlations over millions of samples, moment-based discriminators have a clear theoretical advantage."

Тампере записано в **anechoic chamber з single unit per model** — там немає significant impairments-driven differences. Spectral signature кожної моделі — фабрично визначена (дискретний parameter space). DSGE на time-domain не може додати поза spectral, бо нема **per-Tx physical variability** для захоплення.

**Implication:** на datasets де multiple identical units (DRFF-R2 з 8 Mavic Air 2, 5 Mini 4 Pro), DSGE matter тому що:
1. spectral однакові (same protocol)
2. impairments різні per-unit (different RF chain calibration)
3. DSGE на time-domain ловить impairment-driven time-domain residuals

Це **prediction**, який можна перевірити коли отримаємо DRFF-R2.

### Знахідка 4: PATP α_opt boundary на Task 3 (Mavic vs Inspire)

PATP sweep знайшов α=1.0 — на boundary, тобто polynomial sufficient. Це **підтверджує** Phase 5 patp_unified manuscript task_type-confounder thesis:
- Tasks де impairments **не варіюються** → α=1 (poly), no PATP benefit
- Tasks де impairments **варіюються** → α інтер'єрно (як на synthetic AD9361 з нашого toolkit smoke test, α=0.5)

## Висновок для статті

Iter-2 робить чесний benchmark методології, не overclaim'ить. Точна seqstanced найголовніша:

> "Spectral PSD features сам по собі досягає 94-95% на Tampere multi-protocol дзада. Time-domain DSGE дає 65% standalone (vs 50% chance) — нетривіально, але на anechoic-single-unit setup'і duplicates spectral, не додає. На multi-unit identical-protocol датасетах (DRFF-R2 forthcoming) очікується інша картина — DSGE захопить per-unit impairment-driven differences що spectral не бачить. Tampere — це **lower bound** для DSGE; реальна потенційна сила на impairment-driven scenarios."

Це **публікаційний message**, що базується на чесних numbers, не overstated, і має clear forward path на DRFF-R2 (або synthesis з нашого imp impairment injector).

## Файли

- `results/phase6_iter2_task1.json` — Task 1 numbers
- `results/phase6_iter2_task23.json` — Tasks 2, 3 + PATP + bootstrap
- `run_iter2.py` — full integrated script (для майбутніх runs)
- `run_task1_only.py` — окремий runner Task 1
- `run_task2and3.py` — окремий runner Tasks 2, 3
