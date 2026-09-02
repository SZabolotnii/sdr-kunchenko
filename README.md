# sdr-kunchenko — code supplement (RADAP)

Reproducibility package for

> **Amplitude estimation for SDR baseband under non-Gaussian radio-frequency noise:
> polynomial maximization and a parametrically-adaptive transition basis**
> S. Zabolotnii, V. Koval, A. Chepynoha
> *Visnyk NTUU KPI, Seriia Radiotekhnika, Radioaparatobuduvannia* (RADAP), manuscript **2149**.

It contains the estimator code, the Monte Carlo drivers and configurations, and the
stored results behind every table and figure of the article. Stored results are committed,
so each published number can be checked **without** re-running the Monte Carlo.

---

## 1. Provenance and the commit reference in the article

This repository is a narrow extraction from the authors' full research repository, which
also holds unrelated experiments, manuscripts and unpublished work. Only the code and
results this article depends on are published here.

The article's *Code availability* section cites commit `6d712f1ae27b`. That hash belongs
to the **full** research repository, not to this extraction — a filtered repository
necessarily has its own history. The corresponding state is published here under the tag
**`radap-2149-v2`**, which reproduces the article's numbers. A corrected reference will be
supplied with the final version of the manuscript.

All PMM computations run through the R package **EstemPMM**, which is independently
available on CRAN: <https://CRAN.R-project.org/package=EstemPMM>. This package is *not*
vendored here; it is a declared dependency. The article pins version **0.4.0**.

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -e ".[dev]"
```

R side (required for the PMM2/PMM3 columns; the fractional PATP branch is pure Python):

```r
install.packages("remotes")
remotes::install_version("EstemPMM", version = "0.4.0")   # version pinned by the article
```

`tests/test_pmm_oracle.py::test_estempmm_version_match` is a deliberate guard: it fails
if the installed EstemPMM differs from the `0.4.0` recorded in
`tests/fixtures/pmm_reference.json`. Under the pinned version the whole suite passes.

## 3. Data

The Monte Carlo drivers read SDR baseband datasets that are **not redistributed here**
(size, and third-party licensing for the RadioML-derived residuals). Point
`KU_SDR_DATA_ROOT` at a directory holding the expected layout — see
[`.env.example`](.env.example) and §4 of the article for how the datasets were built:

```bash
export KU_SDR_DATA_ROOT="/path/to/datasets"
python scripts/p0_verify_datasets.py     # checks the dataset invariants
```

Without the data you can still read every stored result (§5) and run the unit tests that
do not touch the datasets.

## 4. Reproducing the article's artifacts

| Article artifact | Command | Output |
|---|---|---|
| **Table 2** (MSE, sandwich/HAC coverage) — polygaussian | `python experiments/exp1_pmm_estimation/run.py` | `experiments/exp1_pmm_estimation/results/exp1A_amplitude.csv` |
| **Table 2** — real-noise | `python experiments/exp1_pmm_estimation/run.py --config experiments/exp1_pmm_estimation/config_realnoise.yaml` | `experiments/exp1_pmm_estimation/results_realnoise/exp1A5_amplitude.csv` |
| **Table 6**, §5.3 (selection-rule audit, regret sweep over `k`) | `python experiments/exp1_pmm_estimation/analyze_dispatch.py` | `experiments/exp1_pmm_estimation/results_dispatch/` |
| **Table 3** (heavy-tail RE) | `python experiments/exp1b_heavy_tail/run.py --config experiments/exp1b_heavy_tail/config_patp_mc.yaml` | `experiments/exp1b_heavy_tail/results_patp_mc/` |
| **Table 4** (platykurtic RE) | `python experiments/exp1b_heavy_tail/run.py --config experiments/exp1b_heavy_tail/config_patp_platy.yaml` | `experiments/exp1b_heavy_tail/results_patp_platy/` |
| **Table S1** (real RF, descriptive only) | `python experiments/exp1b_heavy_tail/run.py --config experiments/exp1b_heavy_tail/config_patp_real.yaml` | `experiments/exp1b_heavy_tail/results_patp_real/` |
| **Tables 3–5** median/trimmed RE + block-bootstrap CI | `python experiments/exp1b_heavy_tail/robust_re.py` | `experiments/exp1b_heavy_tail/results_robust_re/robust_re.csv` |
| Pass-criteria report, Exp 1 | `python experiments/exp1_pmm_estimation/verify.py` (add `--csv .../results_realnoise/exp1A5_amplitude.csv` for the real-noise run) | `results{,_realnoise}/verification_report.txt` |
| Pass-criteria report, Exp 1b | `python experiments/exp1b_heavy_tail/verify.py --results-dir experiments/exp1b_heavy_tail/results_patp_mc/` (likewise for `_platy`, `_real`) | `results_patp_*/verification_report_heavy_tail.txt` |

Figures, and where each one is produced:

| Figure in the article | Produced by | File |
|---|---|---|
| Fig. 1 — MSE vs SNR (polygaussian) | `exp1_pmm_estimation/plot.py` | `experiments/exp1_pmm_estimation/figures/fig1A_mse_vs_snr.pdf` |
| Fig. 2 — RE vs excess kurtosis (real noise) | `exp1_pmm_estimation/plot.py --csv results_realnoise/exp1A5_amplitude.csv --fig-dir figures_realnoise` | `experiments/exp1_pmm_estimation/figures_realnoise/fig1A_RE_vs_kurtosis.pdf` |
| Fig. 3 — interval coverage | same command as Fig. 2 | `experiments/exp1_pmm_estimation/figures_realnoise/fig1A_coverage.pdf` |
| Regret vs threshold multiplier `k` | `exp1_pmm_estimation/analyze_dispatch.py` | `experiments/exp1_pmm_estimation/results_dispatch/fig1A_dispatch_regret_vs_k.pdf` |
| RE(α) transition curve | `exp1b_heavy_tail/plot_patp.py` | `experiments/exp1b_heavy_tail/figures_patp_mc/fig_patp_re_vs_alpha.pdf` |
| α\* vs kurtosis | `exp1b_heavy_tail/plot_patp.py` | `experiments/exp1b_heavy_tail/figures_patp_mc/fig_patp_alpha_star_vs_kurtosis.pdf` |
| Regime map | `exp1b_heavy_tail/plot_patp_regime.py` | `experiments/exp1b_heavy_tail/figures_patp_mc/fig_patp_regime_map.pdf` |

Every `run.py` writes a `manifest.json` next to its results recording the git SHA, the
timestamp, the platform and the full configuration used.

## 5. Stored results, and what to look at first

- `experiments/exp1_pmm_estimation/results{,_realnoise}/` — per-block Monte Carlo output,
  `verification_report.txt`, and `pre_pmm3hac_backup/` (the pre-revision run kept so that
  the "published MSE values are bit-identical after adding the PMM3 HAC standard error"
  claim of the response letter can be checked directly).
- `experiments/exp1_pmm_estimation/results_dispatch/` — the full audit of the selection
  rule, including the sweep over the threshold multiplier.
- `experiments/exp1b_heavy_tail/results_robust_re/robust_re.csv` — median and trimmed
  relative efficiencies with block-bootstrap intervals. Rows for real RF are flagged
  `descriptive_only`: there is one recording per drone, so blocks are not independent and
  no p-value or generalization claim is made from them.

## 6. Code layout

```
src/sdr_kunchenko/
├── pmm/
│   ├── oracle.py        # rpy2 bridge to EstemPMM (lm_pmm2/lm_pmm3, pmm_dispatch, moments)
│   ├── inference.py     # sandwich and Newey–West HAC standard errors, incl. PMM3
│   ├── pooled.py        # cumulant pooling for frozen PMM weights
│   ├── fractional.py    # PMM_FRAC / PATP score psi(e) = sign(e)|e|^p
│   └── dispatch_ext.py  # AUTO_EXT selection rule (OLS / PMM2 / PMM3 / PMM_FRAC)
├── gsa/basis.py         # patp_power — the exponent map p_S(alpha)
├── data_gen/heavy_tail.py
├── utils/cumulants.py   # standardized cumulants via Fisher k-statistics
└── data_loader.py
```

The `gsa/` package is shipped whole rather than reduced to `basis.py`, so that the source
files are byte-identical to the research repository.

Not included, because the article does not depend on them: the GSA change-point, DSGE
classification, denoiser, and direction-finding experiments, and all manuscript sources.

## 7. Tests

```bash
pytest tests/ -q
```

68 tests covering the EstemPMM oracle bridge (bit-equivalence against a stored R
reference fixture), the sandwich/HAC standard errors, cumulant pooling, the fractional
PATP score, the selection rule, and the dataset loader. Four are skipped when
`KU_SDR_DATA_ROOT` is unset; the rest run without the datasets.

## 8. License and citation

Code is released under the MIT license ([`LICENSE`](LICENSE)). If you use it, please cite
the article above and the EstemPMM package:

> Zabolotnii, S. (2026). *EstemPMM: Polynomial Maximization Method for Non-Gaussian
> Regression.* R package version 0.4.0. https://CRAN.R-project.org/package=EstemPMM
