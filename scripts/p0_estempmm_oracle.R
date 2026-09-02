#!/usr/bin/env Rscript
# Phase 0 — EstemPMM oracle reference dump.
#
# Generates two artifacts:
#   tests/fixtures/pmm_reference.json — scalar reference values
#   tests/fixtures/pmm_reference.csv  — eps & x arrays for all 3 DGPs
#
# DGPs:
#   1. Gaussian (γ₃=0, γ₄=0)        — must give g₂ ≈ 1 (no PMM gain)
#   2. Two-piece normal (γ₃≈-1)     — PMM2 should win
#   3. Polygauss K=3                — γ₄ varies with random SDs
#
# Python tests load these arrays, route through src/sdr_kunchenko/pmm/oracle.py,
# and assert bit-equivalence (or ≤ 1e-10 relative deviation) with reference.

suppressPackageStartupMessages({
  library(EstemPMM)
  library(jsonlite)
})

stopifnot(packageVersion("EstemPMM") >= "0.3.0")
cat("EstemPMM version:", as.character(packageVersion("EstemPMM")), "\n")

set.seed(42)
N <- 2000

# ---------- DGPs ----------
make_dgp <- function(kind, n) {
  x <- rnorm(n)
  if (kind == "gaussian") {
    eps <- rnorm(n, sd = 0.5)
  } else if (kind == "tpn") {
    z <- rnorm(n)
    raw <- ifelse(z < 0, z * 1.6, z * 0.7)
    eps <- (raw - mean(raw)) / sd(raw) * 0.5
  } else if (kind == "polygauss_k3") {
    sds <- runif(3, 0.3, 1.5)
    comp <- sample(seq_len(3), n, replace = TRUE)
    raw <- rnorm(n) * sds[comp]
    eps <- (raw - mean(raw)) / sd(raw) * 0.5
  }
  list(x = x, eps = eps, y = 1.5 * x + eps, kind = kind)
}

dgps <- list(
  dgp1 = make_dgp("gaussian", N),
  dgp2 = make_dgp("tpn", N),
  dgp3 = make_dgp("polygauss_k3", N)
)

# ---------- Reference values ----------
safe <- function(expr) tryCatch(expr, error = function(e) NA_real_)
extract_g <- function(vf) if (is.list(vf) && !is.null(vf$g)) vf$g else NA_real_

ref <- list()
for (key in names(dgps)) {
  d <- dgps[[key]]

  # compute_moments returns list(m2, m3, m4, c3, c4, g) — PMM2 decomposition.
  m2_full <- compute_moments(d$eps)
  pmm2_moments <- list(m2 = m2_full$m2, m3 = m2_full$m3, m4 = m2_full$m4)

  # compute_moments_pmm3 returns list(m2, m4, m6, gamma4, gamma6, g3, kappa).
  m3_full <- compute_moments_pmm3(d$eps)
  pmm3_moments <- list(m2 = m3_full$m2, m4 = m3_full$m4, m6 = m3_full$m6)

  fit2 <- lm_pmm2(y ~ x - 1, data = data.frame(y = d$y, x = d$x), verbose = FALSE)
  fit3 <- safe(lm_pmm3(y ~ x - 1, data = data.frame(y = d$y, x = d$x), verbose = FALSE))

  vf2 <- pmm2_variance_factor(pmm2_moments$m2, pmm2_moments$m3, pmm2_moments$m4)
  vf3 <- safe(pmm3_variance_factor(pmm3_moments$m2, pmm3_moments$m4, pmm3_moments$m6))
  vf3 <- if (is.list(vf3)) vf3 else list(g = vf3)

  ref[[key]] <- list(
    description = d$kind,
    n = N,
    moments_pmm2 = pmm2_moments,
    moments_pmm3 = pmm3_moments,
    compute_moments_full = list(m2 = m2_full$m2, m3 = m2_full$m3, m4 = m2_full$m4,
                                 c3 = m2_full$c3, c4 = m2_full$c4, g = m2_full$g),
    compute_moments_pmm3_full = list(m2 = m3_full$m2, m4 = m3_full$m4, m6 = m3_full$m6,
                                      gamma4 = m3_full$gamma4, gamma6 = m3_full$gamma6,
                                      g3 = m3_full$g3, kappa = m3_full$kappa),
    cumulants = list(
      gamma3 = pmm_skewness(d$eps),
      gamma4 = pmm_kurtosis(d$eps),
      gamma6 = pmm_gamma6(d$eps)
    ),
    variance_factor_g2 = list(c3 = vf2$c3, c4 = vf2$c4, g = vf2$g),
    variance_factor_g3 = list(g = extract_g(vf3)),
    pmm2_fit = list(
      coef = unname(coef(fit2)),
      m2 = slot(fit2, "m2"),
      m3 = slot(fit2, "m3"),
      m4 = slot(fit2, "m4"),
      convergence = slot(fit2, "convergence"),
      iterations = slot(fit2, "iterations")
    ),
    pmm3_fit = if (inherits(fit3, "PMM3fit")) list(
      coef = unname(coef(fit3)),
      m2 = slot(fit3, "m2"),
      m4 = slot(fit3, "m4"),
      m6 = slot(fit3, "m6"),
      gamma4 = slot(fit3, "gamma4"),
      gamma6 = slot(fit3, "gamma6"),
      g_coefficient = slot(fit3, "g_coefficient"),
      kappa = slot(fit3, "kappa"),
      convergence = slot(fit3, "convergence"),
      iterations = slot(fit3, "iterations")
    ) else NULL
  )
}

ref[["meta"]] <- list(
  estempmm_version = as.character(packageVersion("EstemPMM")),
  R_version = R.version.string,
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  seed = 42,
  N = N
)

# ---------- Resolve project root ----------
script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grepl("^--file=", script_args)]
if (length(file_arg) == 1) {
  script_path <- normalizePath(sub("^--file=", "", file_arg))
} else {
  script_path <- normalizePath("scripts/p0_estempmm_oracle.R")
}
project_root <- dirname(dirname(script_path))
fixtures_dir <- file.path(project_root, "tests", "fixtures")
dir.create(fixtures_dir, recursive = TRUE, showWarnings = FALSE)

# ---------- Write JSON references ----------
json_path <- file.path(fixtures_dir, "pmm_reference.json")
write_json(ref, json_path, pretty = TRUE, auto_unbox = TRUE, digits = 16)
cat("Reference JSON written to:", json_path, "\n")

# ---------- Write arrays as CSV (one row per dgp) ----------
arr_path <- file.path(fixtures_dir, "pmm_reference_arrays.csv")
arr_df <- do.call(rbind, lapply(names(dgps), function(k) {
  d <- dgps[[k]]
  data.frame(dgp = k, idx = seq_len(N), x = d$x, eps = d$eps, y = d$y)
}))
write.csv(arr_df, arr_path, row.names = FALSE)
cat("Reference arrays written to:", arr_path, "\n")
cat("Done.\n")
