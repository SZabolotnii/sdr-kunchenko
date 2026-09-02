# phase6_tampere_dsge / iterations

Експлораторні скрипти Phase 6, що накопичились по ходу дослідження real-OcuSync I/Q
на Tampere Zenodo (2.4 GHz та 5.8 GHz bands). **Не входять у канонічний пайплайн.**
Канонічний — `../run.py` (entrypoint) + `../plot.py` (figures) + `../analyses/` (writeups).

Усі скрипти пишуть у `../results/` та `../figures/` (path resolution через
`Path(__file__).parent.parent`).

## Карта скриптів

### Iter 1 (initial Tampere)
| Скрипт | Призначення |
|---|---|
| `run_iter2.py` | DSGE на Tampere з burst detection + hybrid features (iter-2 baseline) |
| `run_task1_only.py` | Standalone runner для Task 1 (4-class) — fast feedback loop |
| `run_task2and3.py` | Task 2 (DJI vs non-DJI) + Task 3 (Mavic vs Inspire intra-OcuSync) + PATP α-sweep + bootstrap |

### 5.8 GHz band sweep
| Скрипт | Призначення |
|---|---|
| `run_5G.py` | Phase 6 — 5.8 GHz band experiment (broad replication) |
| `run_5G_basis_search.py` | Extended basis search для 5.8 GHz Task 2 (DJI vs non-DJI) |
| `run_5G_patp.py` | PATP α-sweep на 5.8 GHz Task 2 |
| `verify_5G_findings.py` | Sanity-check 5.8 GHz findings (3-seed replication) |

### Cumulant / distribution diagnostics
| Скрипт | Призначення |
|---|---|
| `run_cumulant_4vs6.py` | Cumulant explicit features — order 4 vs 6, 2.4 vs 5.8 GHz |
| `run_distribution_diagnostics.py` | Distribution diagnostics — чому frac виграє на 2.4 GHz, а cumulants на 5.8 |
| `run_distribution_diagnostics_v2.py` | Те саме v2 — block bootstrap + corrected sampling |
| `run_frac_cumulants_synergy.py` | Final synergy test: frac DSGE + cumulant_order6 hybrid |
| `run_hybrid_variants.py` | Hybrid variants — Сергієва hypothesis check |
| `run_patp_complementarity.py` | PATP α-sweep з complementarity-aware setup |

### Phase 7 burst-level (built on Phase 6 infrastructure)
| Скрипт | Призначення |
|---|---|
| `run_burst_classifier_phase7.py` | Phase 7 — burst-level classifier з per-class metrics + Mavic CI |
| `run_burst_level_phase7.py` | Phase 7 PoC v2 — burst-level (replaces window-bandwidth filter) |
| `run_rc_only_proof.py` | **[SUPERSEDED by run_burst_classifier_phase7.py after Codex review 2026-04-29]** |

### Spectrogram inspection
| Скрипт | Призначення |
|---|---|
| `run_spectrogram_multi_segment.py` | Multi-segment spectrogram analysis (Codex D fix) |
| `run_spectrogram_verify.py` | Spectrogram verification — RC vs video burst patterns на Tampere |

### Figure builders (one-shot)
| Скрипт | Цільова фігура |
|---|---|
| `make_fig2_corrected.py` | Fig 2 — hybrid variants з complementarity story |
| `make_fig3_bimodal.py` | Fig 3 — bimodal PATP α-sweep із degenerate-α=0.5 dip |
| `make_fig5_cross_band.py` | Fig 5 — cross-band 2.4 vs 5.8 GHz PATP α-sweeps |
| `make_fig6_basis_search.py` | Fig 6 — extended basis search на 5.8 GHz |
| `make_fig8_v2.py` | Fig 8 v2 — corrected з window-level block bootstrap CIs |
| `make_fig9_cumulant_orders.py` | Fig 9 — cumulant explicit, order 4 vs 6 на обох bands |

## Звіти за результатами

Після цих ітерацій рефлексія записана у `../analyses/`:
- `ANALYSIS.md` — iter-1 baseline
- `ANALYSIS_iter2.md` / `ANALYSIS_iter2_corrected.md` — поправки iter-2
- `ANALYSIS_distributions.md` / `_v2.md` — distribution-level diagnostics (v2 canonical)
- `ANALYSIS_cumulant_orders.md` — cumulant order analysis (universal winner)
