"""Verification of Exp 1A pass-criteria (Spec §4.2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def coverage(A_hat: np.ndarray, A_true: np.ndarray, se: np.ndarray,
              z: float = 1.96) -> float:
    """Empirical coverage of (A_hat ± z·se) for A_true."""
    lower = A_hat - z * se
    upper = A_hat + z * se
    inside = (A_true >= lower) & (A_true <= upper)
    return float(np.mean(inside))


def variance_ratio(A_ref: np.ndarray, A_alt: np.ndarray) -> float:
    """RE = Var(ref) / Var(alt). RE>1 means alt has lower variance."""
    var_ref = np.var(A_ref, ddof=1)
    var_alt = np.var(A_alt, ddof=1)
    if var_alt <= 0:
        return float("inf")
    return var_ref / var_alt


def verify(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    res: dict = {"n_trials": int(len(df)), "checks": []}

    # ---- Bias of PMM2 ----
    bias_pmm2 = float((df["A_PMM2"] - df["A_true"]).mean())
    res["checks"].append({
        "id": "bias_pmm2",
        "metric": "|Bias(PMM2)|",
        "value": abs(bias_pmm2),
        "threshold": 0.01,
        "pass": bool(abs(bias_pmm2) < 0.01),
        "note": "Spec §4.2 pass-criterion #1",
    })

    # ---- |g_emp - g_theory| / g_theory ----
    # Empirical g_emp for PMM2 = Var(A_OLS - A_true) / Var(A_PMM2 - A_true)
    re_pmm2 = variance_ratio(df["A_OLS"] - df["A_true"], df["A_PMM2"] - df["A_true"])
    g_emp_pmm2 = 1.0 / re_pmm2 if re_pmm2 > 0 else float("nan")
    g_th_mean = float(df["g2_theory"].mean())
    rel_err = abs(g_emp_pmm2 - g_th_mean) / g_th_mean if g_th_mean > 0 else float("inf")
    res["checks"].append({
        "id": "g_emp_vs_theory",
        "metric": "|g_emp - g_theory| / g_theory (PMM2)",
        "value": rel_err,
        "threshold": 0.05,
        "pass": bool(rel_err < 0.05),
        "note": f"g_emp={g_emp_pmm2:.4f}, g_theory_mean={g_th_mean:.4f}",
    })

    # ---- Coverage(95%) — multiple methods ----
    cov_methods = {}
    method_se = [
        ("OLS", "A_OLS", "SE_OLS"),
        ("OLS_HAC", "A_OLS", "SE_OLS_HAC"),
        ("PMM2", "A_PMM2", "SE_PMM2"),
        ("PMM2_HAC", "A_PMM2", "SE_PMM2_HAC"),
        ("PMM3", "A_PMM3", "SE_PMM3"),
        ("PMM3_HAC", "A_PMM3", "SE_PMM3_HAC"),
        ("AUTO", "A_AUTO", "SE_AUTO"),
        ("PMM2_POOL", "A_PMM2_POOL", "SE_PMM2_POOL"),
        ("PMM3_POOL", "A_PMM3_POOL", "SE_PMM3_POOL"),
    ]
    for label, a_col, se_col in method_se:
        if a_col in df.columns and se_col in df.columns:
            mask = df[a_col].notna() & df[se_col].notna() & np.isfinite(df[se_col])
            if mask.sum() > 0:
                a = df.loc[mask, a_col].to_numpy()
                at = df.loc[mask, "A_true"].to_numpy()
                se = df.loc[mask, se_col].to_numpy()
                cov_methods[label] = coverage(a, at, se)

    # Primary check: PMM2 sandwich coverage (per Spec §4.2 #3).
    cov_pmm2 = cov_methods.get("PMM2", float("nan"))
    res["checks"].append({
        "id": "coverage_pmm2",
        "metric": "Coverage(95%) PMM2",
        "value": cov_pmm2,
        "threshold_lower": 0.93,
        "threshold_upper": 0.97,
        "pass": bool(0.93 <= cov_pmm2 <= 0.97) if not np.isnan(cov_pmm2) else None,
    })
    res["coverage_supplement"] = cov_methods

    # ---- HAC-SE recovers coverage on autocorrelated residuals (Phase-1.5 finding) ----
    if "OLS_HAC" in cov_methods:
        cov_hac = cov_methods["OLS_HAC"]
        res["checks"].append({
            "id": "ols_hac_coverage",
            "metric": "Coverage(95%) OLS+HAC",
            "value": cov_hac,
            "threshold_lower": 0.90,
            "threshold_upper": 0.97,
            "pass": bool(0.90 <= cov_hac <= 0.97),
            "note": "HAC-SE closes the autocorrelation-induced under-coverage of naive OLS-SE.",
        })

    # ---- Pooling fixes per-block PMM2 disaster ----
    if "A_PMM2_POOL" in df.columns:
        mse_pmm2 = float(((df["A_PMM2"] - df["A_true"]) ** 2).mean())
        mse_pool = float(((df["A_PMM2_POOL"] - df["A_true"]) ** 2).mean())
        mse_ols = float(((df["A_OLS"] - df["A_true"]) ** 2).mean())
        res["checks"].append({
            "id": "pool_fixes_pmm2",
            "metric": "MSE(PMM2_POOL) ≤ MSE(OLS) (pooling closes per-block disaster)",
            "value": mse_pool,
            "threshold": mse_ols * 1.05,
            "pass": bool(mse_pool <= mse_ols * 1.05),
            "note": f"per-block PMM2 MSE={mse_pmm2:.5f}, pooled={mse_pool:.5f}, OLS={mse_ols:.5f}",
        })

    # ---- RE > 1.2 when |γ₃| > 1.0 ----
    asym = df[df["gamma3_xi"].abs() > 1.0]
    if len(asym) >= 30:
        re_asym = variance_ratio(asym["A_OLS"] - asym["A_true"],
                                  asym["A_PMM2"] - asym["A_true"])
        res["checks"].append({
            "id": "re_pmm2_asymmetric",
            "metric": "RE(PMM2) when |γ₃|>1",
            "value": re_asym,
            "threshold": 1.2,
            "pass": bool(re_asym > 1.2),
            "note": f"n={len(asym)}",
        })
    else:
        res["checks"].append({
            "id": "re_pmm2_asymmetric",
            "metric": "RE(PMM2) when |γ₃|>1",
            "value": None,
            "threshold": 1.2,
            "pass": None,
            "note": f"INSUFFICIENT DATA (n={len(asym)} < 30)",
        })

    # ---- RE > 1.2 when γ₄ < -0.7 and |γ₃| < 0.3 ----
    plat = df[(df["gamma4_xi"] < -0.7) & (df["gamma3_xi"].abs() < 0.3)]
    if len(plat) >= 30 and plat["pmm3_converged"].any():
        plat_ok = plat[plat["pmm3_converged"]]
        re_plat = variance_ratio(plat_ok["A_OLS"] - plat_ok["A_true"],
                                  plat_ok["A_PMM3"] - plat_ok["A_true"])
        res["checks"].append({
            "id": "re_pmm3_platykurtic",
            "metric": "RE(PMM3) when γ₄<-0.7 & |γ₃|<0.3",
            "value": re_plat,
            "threshold": 1.2,
            "pass": bool(re_plat > 1.2),
            "note": f"n={len(plat_ok)}",
        })
    else:
        res["checks"].append({
            "id": "re_pmm3_platykurtic",
            "metric": "RE(PMM3) when γ₄<-0.7 & |γ₃|<0.3",
            "value": None,
            "threshold": 1.2,
            "pass": None,
            "note": f"INSUFFICIENT DATA (n={len(plat)} < 30)",
        })

    # ---- Auto-PMM not worse than best fixed in mean MSE ----
    mse = {m: float(((df[f"A_{m}"] - df["A_true"]) ** 2).mean())
           for m in ["OLS", "PMM2", "PMM3", "AUTO"]}
    best_fixed = min(mse["OLS"], mse["PMM2"], mse["PMM3"])
    res["checks"].append({
        "id": "auto_not_worse",
        "metric": "MSE(AUTO) ≤ min(OLS, PMM2, PMM3)",
        "value": mse["AUTO"],
        "threshold": best_fixed * 1.05,  # 5% slack
        "pass": bool(mse["AUTO"] <= best_fixed * 1.05),
        "note": f"MSE: {mse}",
    })

    # ---- Summary ----
    n_pass = sum(1 for c in res["checks"] if c.get("pass") is True)
    n_total = len(res["checks"])
    n_skip = sum(1 for c in res["checks"] if c.get("pass") is None)
    res["summary"] = {
        "passed": n_pass,
        "total": n_total,
        "skipped": n_skip,
        "pass_rate": f"{n_pass}/{n_total - n_skip}",
    }
    return res


def fmt_report(res: dict) -> str:
    lines = ["# Exp 1A Verification Report", ""]
    lines.append(f"**N trials:** {res['n_trials']}")
    s = res["summary"]
    lines.append(f"**Pass rate:** {s['pass_rate']} (skipped: {s['skipped']})")
    lines.append("")
    lines.append("| ID | Metric | Value | Threshold | Pass | Note |")
    lines.append("|---|---|---|---|---|---|")
    for c in res["checks"]:
        v = c.get("value")
        if v is None:
            v_s = "—"
        elif isinstance(v, (int, float)):
            v_s = f"{v:.4f}"
        else:
            v_s = str(v)
        if "threshold" in c:
            t = c["threshold"]
            t_s = f"{t:.4f}" if isinstance(t, (int, float)) else str(t)
        elif "threshold_lower" in c:
            t_s = f"[{c['threshold_lower']}, {c['threshold_upper']}]"
        else:
            t_s = "—"
        p = c.get("pass")
        p_s = "✓" if p is True else ("✗" if p is False else "—")
        note = c.get("note", "")
        lines.append(f"| {c['id']} | {c['metric']} | {v_s} | {t_s} | {p_s} | {note} |")
    lines.append("")
    if "coverage_supplement" in res:
        cs = res["coverage_supplement"]
        cov_str = ", ".join(f"{k}={v:.3f}" for k, v in cs.items())
        lines.append(f"**Coverage (all methods):** {cov_str}")
    return "\n".join(lines)


def main(csv_path: Path | None = None) -> None:
    if csv_path is None:
        csv_path = PROJECT_ROOT / "experiments" / "exp1_pmm_estimation" / "results" / "exp1A_amplitude.csv"
    res = verify(csv_path)
    out_dir = csv_path.parent
    (out_dir / "verification.json").write_text(json.dumps(res, indent=2, default=str))
    report = fmt_report(res)
    (out_dir / "verification_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()
    main(args.csv)
