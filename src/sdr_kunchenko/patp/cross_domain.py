"""Cross-domain α-mapping aggregation, regression, transferability, Pareto.

Spec §8 — uniting α* from Exp 2 (GSA-CUSUM detection), Exp 3 (DSGE noise-type
classification), Exp 4 (PATP-adaptive denoiser) into a single regression
α* = f(γ₃, γ₄, SNR, task_type) and testing three properties:

  1. Monotonicity  ∂α/∂γ₄ has stable sign on a γ₄-grid (Spec §8.3 #1).
  2. Transferability  classifier-trained mapping → detection task gives ≥90%
     of native α* metric (Spec §8.3 #2).
  3. Pareto frontier  in (α-distance-from-linear, normalised-metric, SNR-coverage)
     non-trivial per task_type (Spec §8.3 #3).

Aggregation choice for Exp 4 (per AskUserQuestion): mode + 25/75-percentile.
Regression choice: poly degree 2 with task-type interaction.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# ---------- per-dataset noise-statistics priors (from PHASE0_REPORT.md) ----

DATASET_GAMMA_PRIOR: dict[str, dict[str, float]] = {
    "fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8": {
        "gamma3": 0.08, "gamma4": 0.50, "noise": "polygauss"},
    "deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7": {
        "gamma3": -0.01, "gamma4": 0.41, "noise": "polygauss"},
    "fpv_realnoise_bpsk_qpsk": {
        "gamma3": 0.00, "gamma4": -0.46, "noise": "real_radioml_residual"},
    "deep_space_realnoise_bpsk_qpsk": {
        "gamma3": 0.01, "gamma4": -0.30, "noise": "real_radioml_residual"},
    "radioml2018_bpsk_qpsk_fpv": {
        "gamma3": 0.01, "gamma4": -0.48, "noise": "real_sdr"},
    "radioml2018_bpsk_qpsk_deep_space": {
        "gamma3": -0.01, "gamma4": -0.34, "noise": "real_sdr"},
}

# Aliases used in experiment configs.
EXP2_PRIMARY_DS = "fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8"
EXP2_REALNOISE_DS = "fpv_realnoise_bpsk_qpsk"
EXP3_CLASS_A_DS = "fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8"
EXP3_CLASS_B_DS = "fpv_realnoise_bpsk_qpsk"


# --------------------------------------------------------- Ingest layer


def load_exp2_alphas(csv_path: str | Path, dataset_name: str) -> pd.DataFrame:
    """Load Exp 2 PATP α-sweep CSV and pick optimal α per SNR via max AUC.

    Returns one row per SNR with chosen α and the metric value.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if _looks_like_lfs_pointer(df):
        df = _load_exp2_alpha_sweep_from_json(csv_path.with_name("exp2_gsa_results.json"))
    rows = []
    for snr, grp in df.groupby("snr_db"):
        best = grp.loc[grp["AUC"].idxmax()]
        rows.append({
            "experiment": "exp2_gsa",
            "task_type": "detection",
            "dataset": dataset_name,
            "snr_db": float(snr),
            "alpha_opt": float(best["alpha"]),
            "primary_metric": "AUC",
            "metric_value": float(best["AUC"]),
            "secondary_metric_ADD": float(best["ADD"]),
            "secondary_metric_FAR": float(best["FAR"]),
        })
    return pd.DataFrame(rows)


def load_exp3_alphas(csv_path: str | Path) -> pd.DataFrame:
    """Load Exp 3 DSGE α-sweep CSV — best α per SNR via max accuracy.

    Exp 3 has a single noise-type classification task pooling both classes,
    so dataset is `<class_a>+<class_b>`.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if _looks_like_lfs_pointer(df):
        local_csv = csv_path.with_name(csv_path.stem + "_local.csv")
        if not local_csv.exists():
            raise ValueError(f"{csv_path} is a Git LFS pointer and {local_csv} is missing")
        df = pd.read_csv(local_csv)
    rows = []
    dataset_str = f"{EXP3_CLASS_A_DS} + {EXP3_CLASS_B_DS}"
    for snr, grp in df.groupby("snr_db"):
        best = grp.loc[grp["accuracy"].idxmax()]
        rows.append({
            "experiment": "exp3_dsge",
            "task_type": "classification",
            "dataset": dataset_str,
            "snr_db": float(snr),
            "alpha_opt": float(best["alpha"]),
            "primary_metric": "accuracy",
            "metric_value": float(best["accuracy"]),
            "secondary_metric_AUC": float(best.get("auc", np.nan)),
        })
    return pd.DataFrame(rows)


def _looks_like_lfs_pointer(df: pd.DataFrame) -> bool:
    """Detect a Git LFS pointer accidentally read as a CSV table."""
    return "snr_db" not in df.columns and any("git-lfs" in str(c) for c in df.columns)


def _load_exp2_alpha_sweep_from_json(json_path: Path) -> pd.DataFrame:
    """Recover Exp 2 PATP sweep rows from the JSON artifact when CSV is LFS-only."""
    if not json_path.exists():
        raise ValueError(f"alpha-sweep CSV is a Git LFS pointer and {json_path} is missing")
    data = json.loads(json_path.read_text())
    rows = []
    for snr_payload in data.get("snr_results", []):
        snr_db = float(snr_payload["snr_db"])
        for point in snr_payload.get("patp_sweep", []):
            rows.append({
                "snr_db": snr_db,
                "alpha": float(point["alpha"]),
                "FAR": float(point.get("FAR", np.nan)),
                "ADD": float(point.get("ADD", np.nan)),
                "P_detect": float(point.get("P_detect", np.nan)),
                "AUC": float(point.get("AUC", np.nan)),
                "MTBF_samples": float(point.get("MTBF_samples", np.nan)),
                "J": float(point.get("J", np.nan)),
                "cond_F": float(point.get("cond_F", np.nan)),
                "threshold": float(point.get("threshold", np.nan)),
            })
    if not rows:
        raise ValueError(f"no patp_sweep records found in {json_path}")
    return pd.DataFrame(rows)


def load_exp4_blocks(json_path: str | Path) -> pd.DataFrame:
    """Unpack hierarchical Exp 4 mapping JSON into a flat per-block table."""
    data = json.loads(Path(json_path).read_text())
    rows = []
    for alias, snrs in data.items():
        ds_name = {
            "primary": EXP2_PRIMARY_DS,
            "realnoise": EXP2_REALNOISE_DS,
        }.get(alias, alias)
        for snr_str, payload in snrs.items():
            try:
                snr_db = float(snr_str)
            except ValueError:
                continue
            alphas = payload.get("training_alpha", [])
            gamma4s = payload.get("training_gamma4", [])
            for i, (a, g4) in enumerate(zip(alphas, gamma4s, strict=True)):
                rows.append({
                    "dataset_alias": alias,
                    "dataset": ds_name,
                    "snr_db": snr_db,
                    "block_idx": i,
                    "training_alpha": float(a),
                    "training_gamma4": float(g4),
                })
    return pd.DataFrame(rows)


def _mode_continuous(values: np.ndarray, ndigits: int = 1) -> float:
    """Mode of continuous values via discretisation to `ndigits` decimal grid.

    Exp 4 stores alphas as 0.0/0.1/.../1.0, so ndigits=1 recovers the natural
    grid without artefacts.
    """
    if len(values) == 0:
        return float("nan")
    rounded = np.round(values, ndigits)
    counts = Counter(rounded)
    return float(counts.most_common(1)[0][0])


def aggregate_exp4_to_snr(blocks_df: pd.DataFrame) -> pd.DataFrame:
    """Mode + 25/75-percentile per (snr_db, dataset). Per AskUserQuestion."""
    rows = []
    for (snr, ds), grp in blocks_df.groupby(["snr_db", "dataset"]):
        a = grp["training_alpha"].to_numpy()
        g4 = grp["training_gamma4"].to_numpy()
        rows.append({
            "experiment": "exp4_denoiser",
            "task_type": "denoising",
            "dataset": ds,
            "snr_db": float(snr),
            "alpha_opt": _mode_continuous(a, ndigits=1),
            "alpha_p25": float(np.percentile(a, 25)),
            "alpha_p75": float(np.percentile(a, 75)),
            "gamma4_block_mean": float(np.mean(g4)),
            "gamma4_block_std": float(np.std(g4, ddof=1)),
            "n_blocks": int(len(a)),
            "primary_metric": "alpha_concentration",
            "metric_value": float(np.mean(a >= 0.95)),
        })
    return pd.DataFrame(rows)


def _attach_priors(df: pd.DataFrame) -> pd.DataFrame:
    """Add gamma3/gamma4 columns from DATASET_GAMMA_PRIOR.

    For Exp 3 (classification across two datasets) we use the *contrast* of
    γ₄ between classes — that's what DSGE actually discriminates on.
    """
    out = df.copy()
    g3, g4 = [], []
    for _, row in out.iterrows():
        ds = row["dataset"]
        if " + " in ds:
            a, b = ds.split(" + ")
            pa, pb = DATASET_GAMMA_PRIOR.get(a, {}), DATASET_GAMMA_PRIOR.get(b, {})
            if pa and pb:
                g3.append(0.5 * (pa["gamma3"] + pb["gamma3"]))
                g4.append(pa["gamma4"] - pb["gamma4"])  # contrast
                continue
        prior = DATASET_GAMMA_PRIOR.get(ds, {})
        g3.append(prior.get("gamma3", float("nan")))
        g4.append(prior.get("gamma4", float("nan")))
    out["gamma3"] = g3
    out["gamma4"] = g4
    return out


def build_unified_table(exp2_pri: pd.DataFrame,
                          exp2_real: pd.DataFrame,
                          exp3: pd.DataFrame,
                          exp4_agg: pd.DataFrame) -> pd.DataFrame:
    """Stack all sources, attach γ₃/γ₄ priors, normalise metrics per task."""
    parts = [exp2_pri, exp2_real, exp3, exp4_agg]
    df = pd.concat(parts, ignore_index=True, sort=False)
    df = _attach_priors(df)

    # Per-experiment min-max normalised metric for Pareto comparability.
    df["metric_normalized"] = float("nan")
    for _exp_id, grp in df.groupby("experiment"):
        v = grp["metric_value"].to_numpy()
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        if vmax - vmin > 1e-12:
            df.loc[grp.index, "metric_normalized"] = (v - vmin) / (vmax - vmin)
        else:
            df.loc[grp.index, "metric_normalized"] = 0.5

    df["alpha_distance_to_05"] = (df["alpha_opt"] - 0.5).abs()
    return df


# --------------------------------------------------- Regression layer


def _design_matrix(df: pd.DataFrame, *, include_task_dummies: bool = True
                    ) -> tuple[np.ndarray, list[str]]:
    """Build a poly2-with-interaction design matrix on (γ₃, γ₄, SNR) + task."""
    cont = df[["gamma3", "gamma4", "snr_db"]].to_numpy(dtype=float)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    x_cont = poly.fit_transform(cont)
    cont_names = list(poly.get_feature_names_out(
        input_features=["gamma3", "gamma4", "snr_db"]))
    if not include_task_dummies:
        return x_cont, cont_names
    tasks = sorted(df["task_type"].unique())
    dummies = pd.get_dummies(df["task_type"], drop_first=True).reindex(
        columns=tasks[1:], fill_value=0).to_numpy(dtype=float)
    dummy_names = [f"task={t}" for t in tasks[1:]]
    x_matrix = np.hstack([x_cont, dummies])
    return x_matrix, cont_names + dummy_names


def fit_alpha_mapping(unified_df: pd.DataFrame, *,
                       include_task_dummies: bool = True,
                       ridge_alpha: float = 1e-3) -> dict:
    """Fit α* = poly₂(γ₃, γ₄, SNR) + task-type dummies.

    Returns dict with model, predict callable, R², per-task residual stats.
    """
    df = unified_df.dropna(subset=["alpha_opt", "gamma3", "gamma4"]).copy()
    x_matrix, names = _design_matrix(df, include_task_dummies=include_task_dummies)
    y = df["alpha_opt"].to_numpy(dtype=float)
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(x_matrix, y)
    yhat = model.predict(x_matrix)
    df["alpha_predicted"] = yhat
    df["residual"] = y - yhat
    ss_res = float(np.sum(df["residual"] ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Per-task residual diagnostics.
    per_task = (df.groupby("task_type")["residual"]
                  .agg(["mean", "std", "count"]).to_dict("index"))

    def predict(gamma3: float, gamma4: float, snr_db: float,
                 task_type: str = "detection") -> float:
        row = pd.DataFrame([{
            "gamma3": gamma3, "gamma4": gamma4, "snr_db": snr_db,
            "task_type": task_type,
        }])
        # Reproduce same dummy ordering as training set.
        all_tasks = sorted(unified_df["task_type"].unique())
        dummy = np.array([[1.0 if task_type == t else 0.0
                            for t in all_tasks[1:]]]) if include_task_dummies else np.empty((1, 0))
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        cont = row[["gamma3", "gamma4", "snr_db"]].to_numpy(dtype=float)
        x_one = np.hstack([poly.fit_transform(cont), dummy])
        return float(np.clip(model.predict(x_one)[0], 0.0, 1.0))

    return {
        "model": model,
        "feature_names": names,
        "predict": predict,
        "r2": r2,
        "residuals_per_task": per_task,
        "fitted_df": df,
        "include_task_dummies": include_task_dummies,
    }


def check_monotonicity(predict: Callable[..., float], *,
                        gamma4_grid: np.ndarray | None = None,
                        gamma3: float = 0.0,
                        snr_db: float = 0.0,
                        task_type: str = "detection") -> dict:
    """Test whether α(γ₄) has a stable sign of derivative on `gamma4_grid`."""
    if gamma4_grid is None:
        gamma4_grid = np.arange(-2.0, 3.01, 0.25)
    alphas = np.array([predict(gamma3, g4, snr_db, task_type)
                        for g4 in gamma4_grid])
    diffs = np.diff(alphas)
    pos = int(np.sum(diffs > 1e-6))
    neg = int(np.sum(diffs < -1e-6))
    n_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
    monotonic = (pos == 0) or (neg == 0)
    return {
        "monotonic": bool(monotonic),
        "n_sign_changes": n_changes,
        "n_pos_diffs": pos,
        "n_neg_diffs": neg,
        "alphas": alphas.tolist(),
        "gamma4_grid": gamma4_grid.tolist(),
        "fixed_gamma3": gamma3, "fixed_snr_db": snr_db, "task_type": task_type,
    }


# ---------------------------------------------------- Transferability


def _exp2_metric_at_alpha(csv_path: str | Path, snr_db: float, alpha: float,
                           metric: str = "AUC") -> float:
    """Interpolate Exp 2 metric vs α at given (snr_db, α)."""
    df = pd.read_csv(csv_path)
    grp = df[np.isclose(df["snr_db"], snr_db)].sort_values("alpha")
    if len(grp) < 2:
        return float("nan")
    f = interp1d(grp["alpha"].to_numpy(), grp[metric].to_numpy(),
                  kind="linear", bounds_error=False, fill_value="extrapolate")
    return float(f(np.clip(alpha, 0.0, 1.0)))


def transferability_test(unified_df: pd.DataFrame,
                          exp2_csv: str | Path,
                          exp2_dataset_name: str,
                          *, train_on: str = "classification",
                          test_on: str = "detection",
                          metric: str = "AUC") -> dict:
    """Train mapping on `train_on` task, predict α for `test_on` SNRs,
    look up actual metric at that α via interpolation, compare to native max.

    Returns ratio-per-SNR + mean ratio. Spec §8.3 #2 target ≥ 0.90.
    """
    train_df = unified_df[unified_df["task_type"] == train_on].copy()
    if train_df.empty:
        return {"error": f"no rows with task_type={train_on}"}
    fit = fit_alpha_mapping(train_df, include_task_dummies=False)
    predict = fit["predict"]

    # Test points: native alphas in `test_on` task, restricted to dataset.
    test_df = unified_df[(unified_df["task_type"] == test_on)
                          & (unified_df["dataset"] == exp2_dataset_name)].copy()
    if test_df.empty:
        return {"error": f"no test rows with task_type={test_on}"}

    rows = []
    for _, r in test_df.iterrows():
        a_pred = predict(r["gamma3"], r["gamma4"], r["snr_db"], train_on)
        m_pred = _exp2_metric_at_alpha(exp2_csv, r["snr_db"], a_pred, metric=metric)
        m_native = _exp2_metric_at_alpha(exp2_csv, r["snr_db"], r["alpha_opt"], metric=metric)
        ratio = (m_pred / m_native) if (m_native > 1e-9) else float("nan")
        rows.append({
            "snr_db": r["snr_db"],
            "alpha_native": r["alpha_opt"],
            "alpha_predicted": a_pred,
            "metric_native": m_native,
            "metric_predicted": m_pred,
            "ratio": ratio,
        })
    rdf = pd.DataFrame(rows)
    return {
        "per_snr": rdf,
        "mean_ratio": float(rdf["ratio"].mean()),
        "median_ratio": float(rdf["ratio"].median()),
        "min_ratio": float(rdf["ratio"].min()),
        "passes_90pct": bool(rdf["ratio"].mean() >= 0.90),
        "train_on": train_on, "test_on": test_on, "metric": metric,
    }


# ------------------------------------------------------------- Pareto


def compute_pareto(unified_df: pd.DataFrame,
                    *, axes: tuple[str, str, str] = (
                        "alpha_distance_to_05",
                        "metric_normalized",
                        "snr_db")) -> pd.DataFrame:
    """Per-task Pareto frontier on a 2D projection (axes[0], axes[1]).

    Convention: we MAXIMISE axes[1] (metric_normalized) and MINIMISE axes[0]
    (alpha-distance from linear regime — proxy for parsimony).

    Axis 2 is reported but not used in dominance check; useful for plotting.
    """
    df = unified_df.dropna(subset=list(axes)).copy()
    df["pareto_optimal"] = False
    for _task, grp in df.groupby("task_type"):
        for i, row_i in grp.iterrows():
            x_i, y_i = row_i[axes[0]], row_i[axes[1]]
            dominated = False
            for j, row_j in grp.iterrows():
                if i == j:
                    continue
                x_j, y_j = row_j[axes[0]], row_j[axes[1]]
                if (x_j <= x_i and y_j >= y_i) and (x_j < x_i or y_j > y_i):
                    dominated = True
                    break
            if not dominated:
                df.at[i, "pareto_optimal"] = True
    return df


# ------------------------------------------------------ public API


__all__ = [
    "DATASET_GAMMA_PRIOR",
    "load_exp2_alphas", "load_exp3_alphas",
    "load_exp4_blocks", "aggregate_exp4_to_snr",
    "build_unified_table",
    "fit_alpha_mapping", "check_monotonicity",
    "transferability_test", "compute_pareto",
]
