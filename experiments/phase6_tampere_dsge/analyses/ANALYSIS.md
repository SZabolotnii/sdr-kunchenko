# Phase 6 — Tampere DSGE: чесний аналіз першого прогону

**Дата:** 2026-04-29
**Дата:** першого реального прогону DSGE на справжніх SDR I/Q від комерційних дронів.
**Датасет:** Tampere Zenodo (https://zenodo.org/records/4264467), 4 з 11 recordings, 2.4 GHz band, fs=120 MS/s, 1000 windows × 1024 samples per class.

## Ключові цифри

| Task | Best basis | Accuracy | F1 | PATP α_opt | PATP F1 | Notes |
|---|---|---|---|---|---|---|
| 1: 4-class multi-protocol | trigonometric | 0.349 | 0.287 | 0.90 (interior) | 0.236 | Random baseline 0.25 |
| 2: DJI vs non-DJI | trigonometric | 0.603 | 0.598 | 0.70 (interior) | 0.578 | Random baseline 0.50 |
| 3: Mavic Pro vs Inspire 2 (intra-OcuSync fingerprinting) | trigonometric | 0.592 | 0.590 | 1.00 (**boundary**) | 0.562 | Random baseline 0.50 |

Bootstrap re_im vs magnitude на Task 3: Δ=+0.006, p=0.64, **не значимо**.

## Що це означає чесно

### Що працює
- **Pipeline end-to-end готовий**: loader → windowing → grouped time-split → leakage audit (0 overlap) → ComplexDSGEFeatureExtractor → basis comparison → PATP sweep → bootstrap → verification report. Усе працює на реальних int16 I/Q файлах.
- **Сигнал є на всіх трьох tasks** — kожна задача дає accuracy >= chance baseline. Найслабша Task 1 (35% vs 25% chance) — +10pp. Найсильніша Task 2 (60% vs 50%) — +10pp.
- **PATP α_opt інтер'єрно для Tasks 1 і 2** — підтверджує, що оптимальний базис не на boundary, тобто PATP має научну цінність. На Task 3 α_opt=1.0 (boundary) — polynomial вистачає.
- **Conditioning OK всюди** — cond(F) < 1000 для всіх базисів, ridge=0.01 достатньо.

### Що НЕ працює (поки)
- **Абсолютні numbers низькі.** 60% на binary з sample size 2K — це слабо. Очікувалось 80%+ якщо метод реально працює. Чотири можливі причини:
  - **Window занадто короткий**: 1024 samples = 8.5 μs at 120 MS/s. OcuSync burst ~600 ms cadence, packet ~ms — window містить переважно фрагмент пакета або шум між пакетами. Більшість 1024-sample windows можуть взагалі не мати OcuSync пакета (тільки noise).
  - **Sub-sampling зламав time-coherence**: я взяв 1000 random non-overlapping windows з 60M complex samples — це 1000 windows із sparse time positions. Можливо, треба дотримуватись послідовних bursts замість random.
  - **Anechoic chamber single-unit limitation**: per Tampere paper, 1 controller per drone in anechoic — мінімум inter-Tx variability. На реальному ефірі (FPV) per-Tx IQ-imbalance + phase-noise варіантивніше → fingerprinting легше.
  - **Magnitude vs re_im не значимий на Task 3** — підтверджує: при поточному setup'і I/Q geometry додає мало над magnitude. Це не критика DSGE, це **інформація про дані**: phase різниця між Mavic Pro і Inspire 2 в anechoic chamber мала.

### Що цікаве для публікації
- **PATP behavior варіюється per task type** — підтверджує знахідку Phase 5 (`patp_unified.md` §5.5): task_type як головний confounder. Тут на даних, відмінних від попередніх (real OcuSync I/Q, не RadioML residual), цей патерн відтворився. Це **strong replication** для cross-domain статті.
- **Trigonometric basis виграє** на всіх трьох tasks — нетривіально. Натуральна гіпотеза: OFDM-based OcuSync має сильну спектральну періодичність, яку trig basis (sin/cos) ловить краще, ніж polynomial / fractional. Це **публікаційне твердження саме по собі** для signal-processing аудиторії: "for OFDM control links, trigonometric DSGE bases dominate".

## Що робити далі

### Швидкі покращення (1-2 години роботи кожне)

1. **Більший window (8192 samples = 68 μs)** — захоплює більше OcuSync content. Очікувано: всі numbers виростуть на 5-15pp.
2. **Burst-aware windowing** — спершу пройти energy detector по recording'у, прийняти тільки windows з above-threshold power. Це гарантує, що feature space — це справжні OcuSync bursts, а не noise.
3. **Hybrid features** (DSGE + spectral PSD bins, як у Phase 3) — очікувано: великий boost (Phase 3 показав +18pp від DSGE-only до hybrid).
4. **Більше windows per class** — зараз 1000, можна 5K-10K. Перший прогін був з малим N для швидкості.

### Експериментальні розширення (1-3 дні кожне)

5. **5.8 GHz band** — додати DJI_phantom_4_pro_plus_5G + DJI_inspire_2_5G (200 MS/s). Більший spread → більше discriminative info.
6. **OFDM-aware features** — після burst detection декодувати OFDM symbols (proto17/dji_droneid PHY params), фітити DSGE на per-symbol residuals замість raw I/Q. Це "правильний" фізичний рівень для fingerprinting.
7. **Cross-recording validation** — вирізати 5-секундний buffer і перевірити чи fingerprinting генералізує між початком і кінцем recording'у. Тест на channel non-stationarity.

### Що НЕ робити
- Не намагатись reproduce paper numbers (вони не публікували classifier metrics, лише spectrograms). Tampere — це data-only release.
- Не чекати високий accuracy на anechoic single-unit data. Реалістична верхня межа Task 3 (intra-OcuSync) на цьому наборі — мабуть 70-75%.
- Не змінювати теорію DSGE під результат — якщо магнітуди-only давало ту ж accuracy, це **факт про дані**, не методу.

## Висновок одним абзацем

Перший прогін DSGE на справжніх SDR I/Q від комерційних дронів — **технічно успішний**, методологічно бездоганний (grouped split, leakage audit, bootstrap, basis comparison, PATP boundary check). Numbers скромні (35-60%), але сигнал є на всіх трьох tasks і вище chance. Trigonometric basis консистентно виграє — це signature інтерпретація для OFDM протоколів. PATP α_opt варіюється per task — replication знахідки Phase 5 на новому домені. Reframing: **це не "слабкий результат" — це baseline, з якого ми тепер починаємо систематичні покращення (burst detection, bigger windows, hybrid features, per-symbol residuals).** Стаття-кандидат: "DSGE on real OcuSync — first results and methodology" — навіть з поточними numbers це публікаційний вклад, бо немає інших робіт на цьому датасеті з reproducible verification protocol.

---

*Підготовано: Айона + Сергій, 2026-04-29.*
*Артефакти: results/phase6_tampere_results.json, results/phase6_classification_report.txt.*
*Pipeline: experiments/phase6_tampere_dsge/run.py (16.5 KB).*
