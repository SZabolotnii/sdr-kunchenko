"""Verification of Exp 3 pass-criteria (Spec §6.8)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def verify(results_dir: Path) -> dict:
    df = pd.read_csv(results_dir / "exp3_basis_comparison.csv")
    patp = pd.read_csv(results_dir / "exp3_patp_alpha_sweep.csv")
    res: dict = {"checks": []}

    # 1. DSGE-only ≥ 70% accuracy.
    dsge_lr = df[(df["type"] == "dsge") & (df["classifier"] == "rf")]
    if not dsge_lr.empty:
        # Best basis at each SNR.
        best_dsge = dsge_lr.groupby("snr_db")["accuracy"].max()
        mean_acc = float(best_dsge.mean())
        res["checks"].append({
            "id": "dsge_70pct",
            "metric": "DSGE-only RF best-basis accuracy ≥ 0.70 (mean over SNR)",
            "value": mean_acc,
            "threshold": 0.70,
            "pass": bool(mean_acc >= 0.70),
            "note": f"per-SNR best DSGE acc: {best_dsge.to_dict()}",
        })

    # 2. Hybrid ≥ DSGE AND ≥ spectral baseline.
    base_rf = df[(df["type"] == "baseline") & (df["classifier"] == "rf")]
    hybrid_rf = df[(df["type"] == "hybrid") & (df["classifier"] == "rf")]
    if not base_rf.empty and not hybrid_rf.empty:
        merged = (hybrid_rf.groupby("snr_db")["accuracy"].max()
                   .rename("hybrid")
                   .to_frame()
                   .join(base_rf.groupby("snr_db")["accuracy"].mean().rename("spectral"))
                   .join(dsge_lr.groupby("snr_db")["accuracy"].max().rename("dsge")))
        merged["beats_spectral"] = merged["hybrid"] >= merged["spectral"]
        merged["beats_dsge"] = merged["hybrid"] >= merged["dsge"]
        res["checks"].append({
            "id": "hybrid_beats_baselines",
            "metric": "Hybrid ≥ Spectral-only AND ≥ DSGE-only on all SNR",
            "value": int(merged["beats_spectral"].sum() + merged["beats_dsge"].sum()),
            "threshold": int(2 * len(merged)),
            "pass": bool(merged["beats_spectral"].all() and merged["beats_dsge"].all()),
            "note": (f"hybrid mean={merged['hybrid'].mean():.3f}, "
                      f"spectral mean={merged['spectral'].mean():.3f}, "
                      f"dsge mean={merged['dsge'].mean():.3f}"),
        })

        # 2b. Hybrid > spectral by paired bootstrap (per-SNR).
        diffs = (merged["hybrid"] - merged["spectral"]).values
        # H0: diffs ≤ 0. One-sided test.
        rng = np.random.default_rng(0)
        B = 2000
        boot_means = np.empty(B)
        for i in range(B):
            samp = rng.choice(diffs, size=len(diffs), replace=True)
            boot_means[i] = samp.mean()
        # p-value: P(boot_means ≤ 0)
        p = float((boot_means <= 0).mean())
        res["checks"].append({
            "id": "hybrid_vs_spectral_significant",
            "metric": "Paired bootstrap p-value (hybrid − spectral > 0)",
            "value": p,
            "threshold": 0.05,
            "pass": bool(p < 0.05),
            "note": f"observed mean diff = {diffs.mean():.4f}",
        })

    # 3. PATP α_opt strictly interior.
    if not patp.empty:
        # Best α per SNR
        rows = []
        for snr, grp in patp.groupby("snr_db"):
            best = grp.loc[grp["accuracy"].idxmax()]
            rows.append({"snr_db": snr, "alpha_opt": best["alpha"],
                          "acc": best["accuracy"]})
        df_a = pd.DataFrame(rows)
        interior = ((df_a["alpha_opt"] > 0.05) & (df_a["alpha_opt"] < 0.95)).sum()
        res["checks"].append({
            "id": "patp_alpha_interior",
            "metric": "PATP α* strictly interior on ALL SNR",
            "value": int(interior),
            "threshold": int(len(df_a)),
            "pass": bool(interior == len(df_a)),
            "note": (f"α* per SNR: "
                      f"{df_a.set_index('snr_db')['alpha_opt'].to_dict()}"),
        })

        # 3b. PATP best AT α* better than poly-s3 alone (DSGE comparison).
        if not dsge_lr.empty:
            poly_dsge_lr = df[(df["type"] == "dsge") & (df["basis"] == "poly_s3")
                               & (df["classifier"] == "lr")]
            patp_best_per_snr = patp.groupby("snr_db")["accuracy"].max()
            poly_per_snr = poly_dsge_lr.groupby("snr_db")["accuracy"].mean()
            joined = patp_best_per_snr.to_frame("patp").join(poly_per_snr.rename("poly"))
            wins = (joined["patp"] >= joined["poly"]).sum()
            res["checks"].append({
                "id": "patp_dominates_poly",
                "metric": "PATP@α* ≥ poly-s3 (DSGE-LR), per SNR count",
                "value": int(wins),
                "threshold": int(len(joined) // 2),
                "pass": bool(wins >= len(joined) // 2),
                "note": f"patp mean={joined['patp'].mean():.3f}, poly mean={joined['poly'].mean():.3f}",
            })

    n_pass = sum(1 for c in res["checks"] if c.get("pass") is True)
    n_total = len(res["checks"])
    res["summary"] = {"passed": n_pass, "total": n_total,
                       "pass_rate": f"{n_pass}/{n_total}"}
    return res


def fmt_report(res: dict) -> str:
    lines = ["# Exp 3 Verification Report", "",
             f"**Pass rate:** {res['summary']['pass_rate']}", "",
             "| ID | Metric | Value | Threshold | Pass | Note |",
             "|---|---|---|---|---|---|"]
    for c in res["checks"]:
        v = c.get("value")
        v_s = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        t = c.get("threshold")
        t_s = f"{t:.4f}" if isinstance(t, (int, float)) else str(t)
        p = c.get("pass")
        p_s = "✓" if p is True else ("✗" if p is False else "—")
        note = c.get("note", "")
        lines.append(f"| {c['id']} | {c['metric']} | {v_s} | {t_s} | {p_s} | {note} |")
    return "\n".join(lines)


def main(results_dir: Path) -> None:
    res = verify(results_dir)
    (results_dir / "verification.json").write_text(json.dumps(res, indent=2, default=str))
    report = fmt_report(res)
    (results_dir / "verification_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                    default=PROJECT_ROOT / "experiments" / "exp3_dsge_classification" / "results")
    args = p.parse_args()
    main(args.results_dir)
