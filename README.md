# sdr-kunchenko — code supplement

Reproducibility package for a series of articles applying Kunchenko's theory of stochastic
polynomials — the polynomial maximization method (PMM), the parametrically-adaptive
transition polynomial (PATP), and decomposition in a space with a generating element
(DSGE) — to software-defined-radio baseband processing.

It holds the estimator and feature code, the Monte Carlo drivers and configurations, and
the **stored results** behind every published table and figure. Because results are
committed, each number can be checked without re-running the pipelines.

## Articles and where their code lives

| Article | Tag | Directories |
|---|---|---|
| **Amplitude estimation for SDR baseband under non-Gaussian RF noise: polynomial maximization and a parametrically-adaptive transition basis** — *Visnyk NTUU KPI, Ser. Radiotekhnika, Radioaparatobuduvannia* (RADAP), ms. 2149 | `radap-2149-v2` | `experiments/exp1_pmm_estimation/`, `experiments/exp1b_heavy_tail/` |
| **DSGE Features for Noise-Type Classification: Continuous Basis Selection via PATP** — *Radio Electronics, Computer Science, Control* (RIC) | `exp3-final` | `experiments/exp3_dsge_classification/` |
| **PATP-adaptive denoising of SDR baseband signals based on subspace projection and Lp-IRLS** — RIC | `exp4-final` | `experiments/exp4_patp_denoiser/` |
| **Higher-Order Cumulants and Kunchenko-Space Reconstruction as Complements to Spectral Features for OcuSync Drone Classification** — *Radioelectronics*, ms. 361891 | `exp6-final` | `experiments/phase6_tampere_dsge/` |

Two further directories support the four above rather than a submitted article of their
own, and are published because the articles' cross-references and pipelines depend on
them: `experiments/exp2_gsa_detection/` (tag `exp2-final`; GSA-CUSUM burst detection —
its α sweeps are an input to Phase 5) and `experiments/phase5_cross_cutting/`
(tag `exp5-final`, cited as a manuscript in preparation by the OcuSync article).

**About the tags.** This repository is a narrow extraction from the authors' full research
repository, which also holds unpublished work and all manuscript sources; only what the
articles depend on is published here. A filtered repository necessarily has its own
history, so the tags above mark the published snapshot that reproduces each article rather
than the original development commits. In particular, the RADAP article's *Code
availability* section cites commit `6d712f1ae27b`, which belongs to the full repository;
the corresponding state here is `radap-2149-v2`, and a corrected reference will be
supplied with the final version of that manuscript.

All PMM computations run through the R package **EstemPMM**, independently available on
CRAN: <https://CRAN.R-project.org/package=EstemPMM>. It is a declared dependency, not
vendored here. The RADAP article pins version 0.4.0.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -e ".[dev]"
```

R side — required only for the PMM2/PMM3 columns of Exp 1; the fractional PATP branch,
the DSGE features and the denoiser are pure Python:

```r
install.packages("remotes")
remotes::install_version("EstemPMM", version = "0.4.0")   # version pinned by the RADAP article
```

`tests/test_pmm_oracle.py::test_estempmm_version_match` is a deliberate guard: it fails if
the installed EstemPMM differs from the `0.4.0` recorded in
`tests/fixtures/pmm_reference.json`. Under the pinned version the whole suite passes.

## 2. Data

The drivers read SDR baseband datasets that are **not redistributed here** (size, and
third-party licensing for the RadioML-derived residuals):

- `KU_SDR_DATA_ROOT` — the synthetic polygaussian, real-noise and RadioML-derived datasets
  used by Exp 1–5. See [`.env.example`](.env.example); `python scripts/p0_verify_datasets.py`
  checks the expected layout and invariants.
- `KU_TAMPERE_DATA_ROOT` — the Tampere OcuSync drone I/Q extract used by the OcuSync
  article, from [Zenodo 4264467](https://zenodo.org/records/4264467). Defaults to
  `data/tampere_extract/` inside this repository.

Without the data you can still read every stored result and run the unit tests that do not
touch the datasets.

## 3. Reproducing the published artifacts

Each experiment directory follows the same triple: `run.py` (Monte Carlo / pipeline driver,
writes `results/` plus a `manifest.json` recording the git SHA, timestamp, platform and
full configuration), `verify.py` (pass-criteria check), `plot.py` (figures).

### RADAP — PMM/PATP amplitude estimation

| Article artifact | Command | Output |
|---|---|---|
| Table 2 (MSE, sandwich/HAC coverage) — polygaussian | `python experiments/exp1_pmm_estimation/run.py` | `results/exp1A_amplitude.csv` |
| Table 2 — real noise | `run.py --config experiments/exp1_pmm_estimation/config_realnoise.yaml` | `results_realnoise/exp1A5_amplitude.csv` |
| Table 6, §5.3 — selection-rule audit and regret sweep | `python experiments/exp1_pmm_estimation/analyze_dispatch.py` | `results_dispatch/` |
| Tables 3, 4, S1 — PATP α-transition | `python experiments/exp1b_heavy_tail/run.py --config experiments/exp1b_heavy_tail/config_patp_{mc,platy,real}.yaml` | `results_patp_{mc,platy,real}/` |
| Tables 3–5 — median/trimmed RE with block-bootstrap CI | `python experiments/exp1b_heavy_tail/robust_re.py` | `results_robust_re/robust_re.csv` |
| Figures 1–3 | `plot.py` (add `--csv results_realnoise/exp1A5_amplitude.csv --fig-dir figures_realnoise` for Figs. 2–3) | `figures/`, `figures_realnoise/` |

`results{,_realnoise}/pre_pmm3hac_backup/` holds the pre-revision run, kept so that the
response letter's claim — published MSE values are bit-identical after adding the PMM3 HAC
standard error — can be checked directly.

### RIC — DSGE features for noise-type classification (Exp 3)

```bash
python experiments/exp3_dsge_classification/run.py       # -> results/
python experiments/exp3_dsge_classification/verify.py    # 4/5 criteria
python experiments/exp3_dsge_classification/plot.py      # -> figures/
python experiments/exp3_dsge_classification/review_response.py   # -> results_review_response/
```

`results_review_response/` contains the extra analyses prepared for the editor: block-size
and S/λ sensitivity, an augmented PATP α sweep, per-block predictions and runtimes.

### RIC — PATP-adaptive denoising (Exp 4)

```bash
python experiments/exp4_patp_denoiser/run.py       # -> results/exp4_denoiser_metrics.csv
python experiments/exp4_patp_denoiser/verify.py    # 5/6 criteria
python experiments/exp4_patp_denoiser/plot.py      # -> figures/
```

### Radioelectronics — OcuSync drone classification (Exp 6 / Phase 6)

```bash
export KU_TAMPERE_DATA_ROOT=/path/to/tampere_extract
python experiments/phase6_tampere_dsge/run.py               # canonical 4-class and binary tasks
python experiments/phase6_tampere_dsge/iterations/run_iter2.py   # iter-2 pipeline behind the headline numbers
python experiments/phase6_tampere_dsge/verify.py            # checks the canonical artifacts are present
```

The article's four figures come from `iterations/make_fig*.py` and are stored in
`figures/` (`fig9_cumulant_orders`, `fig3_patp_bimodal`, `fig8_cumulants_v2`,
`fig10_order_increment`, in article order). The narrative reading of the pass criteria
lives in `analyses/ANALYSIS_iter2_corrected.md`, the canonical source for the
99.83 % / 95.5 % hybrid results.

This pipeline imports five modules from the author's separate DSGE toolkit. In the
research tree they were reached through an absolute path that existed only on the machine
where the experiments were run, which made the pipeline unrunnable elsewhere. They are
vendored unmodified under `experiments/phase6_tampere_dsge/toolkit/`, and the importing
scripts now resolve that directory relative to their own location — see
[`toolkit/README.md`](experiments/phase6_tampere_dsge/toolkit/README.md). This is the only
functional change made to any script during extraction.

## 4. Code layout

```
src/sdr_kunchenko/
├── pmm/               # PMM estimation
│   ├── oracle.py        # rpy2 bridge to EstemPMM (lm_pmm2/lm_pmm3, pmm_dispatch, moments)
│   ├── inference.py     # sandwich and Newey-West HAC standard errors, incl. PMM3
│   ├── pooled.py        # cumulant pooling for frozen PMM weights
│   ├── fractional.py    # PMM_FRAC / PATP score psi(e) = sign(e)|e|^p
│   └── dispatch_ext.py  # AUTO_EXT selection rule (OLS / PMM2 / PMM3 / PMM_FRAC)
├── gsa/               # basis construction, GSA coefficients, CUSUM detector, thresholds
├── dsge/features.py   # per-class F.K = B reconstruction-error features
├── denoiser/          # PATP subspace denoiser, classical baselines, alpha mapping
├── patp/cross_domain.py  # cross-task alpha-mapping and transferability
├── data_gen/heavy_tail.py
├── utils/cumulants.py # standardized cumulants via Fisher k-statistics
└── data_loader.py
```

Not included, because no published article depends on them: the Phase 7
direction-finding / RF-fingerprinting work, and all manuscript and dissertation sources.

Source files are byte-identical to the research repository (the one exception is the
toolkit path fix described in §3), so a few comments and docstrings still point at files
that live only there — `CLAUDE.md`, `docs/specs/kunchenko_sdr_experiments.md`,
`manuscript/…`. Two of those pointers are worth spelling out, because the articles rely on
the conventions behind them:

- **Spec §10.1** — each experiment has explicit pass criteria; `verify.py` writes a
  `verification_report.txt` marking each PASS or FAIL. Reports are committed, failures
  included: Exp 3 is 4/5, Exp 4 is 5/6, and the reports say which criterion missed and by
  how much.
- **Verification Standard 9 — the resampling unit is the dependence unit.** I/Q segments
  from one burst are near-duplicates, and bursts from one recording share a channel and a
  receiver, so resampling rows would treat `n_segments` as the sample size when the real
  sample size is `n_bursts` or `n_recordings`. Confidence intervals are therefore built by
  cluster bootstrap over groups, and rows backed by a single recording are labelled
  `descriptive_only` and carry no p-value — see `results_robust_re/robust_re.csv`, whose
  schema records `resampling_unit` and `n_recordings` next to every interval. Monte Carlo
  replicates in Exp 1 and 1b are genuinely i.i.d. and are the one exception.

## 5. Tests

```bash
pytest tests/ -q
```

93 tests covering the EstemPMM oracle bridge (bit-equivalence against a stored R reference
fixture), the sandwich/HAC standard errors, cumulant pooling, the fractional PATP score,
the selection rule, the GSA basis and detector, the DSGE features, the denoiser, the
cross-domain α-mapping, and the dataset loader. Four are skipped when `KU_SDR_DATA_ROOT`
is unset; the rest run without the datasets.

## 6. License and citation

Code is released under the MIT license ([`LICENSE`](LICENSE)). Please cite the relevant
article above and the EstemPMM package:

> Zabolotnii, S. (2026). *EstemPMM: Polynomial Maximization Method for Non-Gaussian
> Regression.* R package version 0.4.0. https://CRAN.R-project.org/package=EstemPMM
