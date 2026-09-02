"""Dispatcher audit for Exp 1A — AUTO (EstemPMM::pmm_dispatch) vs oracle-best.

Answers RADAP-2149 reviewer point 2: the dispatch thresholds |γ̂₃| > 0.3 and
γ̂₄ < −0.7 are undocumented constants, and the paper never reports how often the
rule picks the wrong estimator.

Uses only the already-computed MC results (no re-run):
  results/exp1A_amplitude.csv            (polygauss, N=1024, 2500 blocks)
  results_realnoise/exp1A5_amplitude.csv (realnoise,  N=1024, 2500 blocks)

Outputs -> results_dispatch/:
  dispatch_sweep.csv            per-k aggregate MSE / regret / method mix
  dispatch_summary.json         confusion matrices + headline numbers
  dispatch_report.txt           human-readable summary
  fig1A_dispatch_regret_vs_k.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "results_dispatch"

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

METHODS = ["OLS", "PMM2", "PMM3"]
BLOCK_N = 1024               # dataset block size (both datasets)
G2_THRESHOLD = 0.95          # EstemPMM::pmm_dispatch default, kept fixed in sweep
STRONG_ASYM = 1.0            # |γ̂₃| > 1 branch, kept fixed in sweep
N_G4_BINS = 10               # quantile bins for γ̂₄ breakdown / binned oracle

DATASETS = {
    "polygauss": EXP_DIR / "results" / "exp1A_amplitude.csv",
    "realnoise": EXP_DIR / "results_realnoise" / "exp1A5_amplitude.csv",
}


# ------------------------------------------------------------ dispatch rule


def g2_factor(gamma3: np.ndarray, gamma4: np.ndarray) -> np.ndarray:
    """EstemPMM:::.g2_factor — 1 − γ₃²/(2+γ₄), clamped to 1 on bad denominator."""
    denom = 2.0 + gamma4
    out = np.ones_like(gamma3, dtype=float)
    ok = np.isfinite(denom) & (denom > 0)
    out[ok] = 1.0 - gamma3[ok] ** 2 / denom[ok]
    return out


def dispatch_rule(gamma3: np.ndarray, gamma4: np.ndarray, *,
                  symmetry_threshold: float,
                  kurtosis_threshold: float) -> np.ndarray:
    """Re-implementation of EstemPMM::pmm_dispatch branch logic (R order kept).

    Verified bit-exact against the `method_auto` column at the package defaults
    (symmetry_threshold=0.3, kurtosis_threshold=−0.7) on both datasets.
    """
    a3 = np.abs(gamma3)
    g2 = g2_factor(gamma3, gamma4)
    asym = a3 > symmetry_threshold
    platy = gamma4 < kurtosis_threshold
    out = np.full(a3.shape, "OLS", dtype=object)
    out[asym & platy] = "PMM2"                       # branch 4 (asym dominates)
    out[(~asym) & platy] = "PMM3"                    # branch 3
    out[asym & (g2 < G2_THRESHOLD)] = "PMM2"         # branch 2
    out[a3 > STRONG_ASYM] = "PMM2"                   # branch 1
    return out


def k_of(threshold: float, cumulant_var: float, n: int = BLOCK_N) -> float:
    """Convert an absolute threshold to the N-scaled multiplier k."""
    return float(threshold / np.sqrt(cumulant_var / n))


# ------------------------------------------------------------ core analysis


def squared_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Per-block (Â_method − A_true)² for each estimator plus AUTO."""
    err = pd.DataFrame(index=df.index)
    for m in METHODS + ["AUTO"]:
        err[m] = (df[f"A_{m}"] - df["A_true"]) ** 2
    return err


def policy_mse(err: pd.DataFrame, choice: np.ndarray) -> float:
    """MSE of a per-block selection policy given the chosen method names."""
    picked = np.array([err.iloc[i][choice[i]] for i in range(len(choice))])
    return float(np.mean(picked))


def policy_median_se(err: pd.DataFrame, choice: np.ndarray) -> float:
    """Median squared error of a policy — outlier-insensitive companion to MSE."""
    picked = np.array([err.iloc[i][choice[i]] for i in range(len(choice))])
    return float(np.median(picked))


def tail_share(e: np.ndarray, top: int = 10) -> float:
    """Fraction of the total squared error carried by the `top` worst blocks."""
    return float(np.sort(e)[::-1][:top].sum() / e.sum())


def binned_oracle(err: pd.DataFrame, gamma4: np.ndarray,
                  n_bins: int = N_G4_BINS) -> tuple[np.ndarray, pd.DataFrame]:
    """Oracle restricted to one decision per γ̂₄ quantile bin.

    Far less optimistic than the per-block argmin, which partly selects on
    estimation noise; here the oracle may only condition on the same statistic
    the practical rule sees.
    """
    bins = pd.qcut(pd.Series(gamma4), n_bins, duplicates="drop")
    choice = np.empty(len(err), dtype=object)
    rows = []
    for b in bins.cat.categories:
        mask = (bins == b).to_numpy()
        if not mask.any():
            continue
        mses = {m: float(err.loc[mask, m].mean()) for m in METHODS}
        best = min(mses, key=mses.get)
        choice[mask] = best
        rows.append({"g4_bin": str(b), "g4_mid": float(b.mid), "n": int(mask.sum()),
                     "bin_oracle": best,
                     **{f"MSE_{m}": mses[m] for m in METHODS}})
    return choice, pd.DataFrame(rows)


def confusion(auto: np.ndarray, oracle: np.ndarray) -> dict:
    """Confusion counts dispatcher-choice × oracle-best, plus row/overall rates."""
    mat = {a: {o: int(np.sum((auto == a) & (oracle == o))) for o in METHODS}
           for a in METHODS}
    correct = sum(mat[m][m] for m in METHODS)
    return {"matrix": mat, "n": int(len(auto)),
            "misclassification_rate": 1.0 - correct / len(auto)}


def bootstrap_ci(err: pd.DataFrame, choice_auto: np.ndarray,
                 choice_oracle: np.ndarray, *, n_boot: int = 2000,
                 seed: int = 1234) -> tuple[float, float]:
    """Percentile CI for regret = MSE(AUTO)/MSE(oracle) over MC replicates.

    Resampling unit = MC replicate. Legitimate here: Exp 1A blocks are drawn
    without replacement from disjoint dataset rows, i.e. i.i.d. replicates
    (the CLAUDE.md Verification-Standard-9 exception).
    """
    rng = np.random.default_rng(seed)
    e_auto = err["AUTO"].to_numpy()
    e_orac = np.array([err.iloc[i][choice_oracle[i]] for i in range(len(err))])
    del choice_auto
    n = len(e_auto)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = e_auto[idx].mean() / e_orac[idx].mean()
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sweep(df: pd.DataFrame, err: pd.DataFrame, mse_oracle: float,
          mse_bin_oracle: float, med_oracle: float,
          k_grid: np.ndarray) -> pd.DataFrame:
    """Re-derive the dispatch decision offline for each k; aggregate MSE/regret."""
    g3 = df["gamma3_xi"].to_numpy()
    g4 = df["gamma4_xi"].to_numpy()
    rows = []
    for k in k_grid:
        sym = k * np.sqrt(6.0 / BLOCK_N)
        kur = -k * np.sqrt(24.0 / BLOCK_N)
        choice = dispatch_rule(g3, g4, symmetry_threshold=sym,
                               kurtosis_threshold=kur)
        mse = policy_mse(err, choice)
        med = policy_median_se(err, choice)
        rows.append({
            "k": float(k),
            "symmetry_threshold": float(sym),
            "kurtosis_threshold": float(kur),
            "MSE": mse,
            "regret_vs_block_oracle": mse / mse_oracle,
            "regret_vs_bin_oracle": mse / mse_bin_oracle,
            "MSE_ratio_vs_OLS": mse / float(err["OLS"].mean()),
            "medSE": med,
            "med_regret_vs_block_oracle": med / med_oracle,
            "medSE_ratio_vs_OLS": med / float(err["OLS"].median()),
            **{f"frac_{m}": float(np.mean(choice == m)) for m in METHODS},
        })
    return pd.DataFrame(rows)


def analyse(name: str, csv_path: Path, k_grid: np.ndarray) -> dict:
    df = pd.read_csv(csv_path)
    err = squared_errors(df)
    auto = df["method_auto"].to_numpy().astype(object)

    # sanity: our re-implementation must reproduce the recorded choice exactly
    repro = dispatch_rule(df["gamma3_xi"].to_numpy(), df["gamma4_xi"].to_numpy(),
                          symmetry_threshold=0.3, kurtosis_threshold=-0.7)
    repro_match = float(np.mean(repro == auto))

    # the rule as *written in the paper* omits the g2<0.95 guard and the
    # |γ̂₃|>1 branch — check how far that simplification is from the code
    a3 = np.abs(df["gamma3_xi"].to_numpy())
    paper_rule = np.where(a3 > 0.3, "PMM2",
                          np.where(df["gamma4_xi"].to_numpy() < -0.7, "PMM3", "OLS"))
    paper_rule_match = float(np.mean(paper_rule == auto))

    oracle = err[METHODS].to_numpy().argmin(axis=1)
    oracle = np.array([METHODS[i] for i in oracle], dtype=object)
    bin_choice, bin_table = binned_oracle(err, df["gamma4_xi"].to_numpy())

    mse = {m: float(err[m].mean()) for m in METHODS + ["AUTO"]}
    med = {m: float(err[m].median()) for m in METHODS + ["AUTO"]}
    tails = {m: tail_share(err[m].to_numpy()) for m in METHODS + ["AUTO"]}
    mse_oracle = policy_mse(err, oracle)
    mse_bin_oracle = policy_mse(err, bin_choice)
    med_oracle = policy_median_se(err, oracle)

    conf = confusion(auto, oracle)
    ci = bootstrap_ci(err, auto, oracle)

    # misclassification by γ̂₄ decile
    bins = pd.qcut(df["gamma4_xi"], N_G4_BINS, duplicates="drop")
    by_bin = []
    for b in bins.cat.categories:
        mask = (bins == b).to_numpy()
        if not mask.any():
            continue
        by_bin.append({
            "g4_bin": str(b), "g4_mid": float(b.mid), "n": int(mask.sum()),
            "misclass_rate": float(np.mean(auto[mask] != oracle[mask])),
            "regret": float(err.loc[mask, "AUTO"].mean()
                            / np.mean([err.iloc[i][oracle[i]]
                                       for i in np.where(mask)[0]])),
            "auto_mix": {m: int(np.sum(auto[mask] == m)) for m in METHODS},
            "oracle_mix": {m: int(np.sum(oracle[mask] == m)) for m in METHODS},
        })

    sw = sweep(df, err, mse_oracle, mse_bin_oracle, med_oracle, k_grid)
    sw.insert(0, "dataset", name)
    best = sw.loc[sw["regret_vs_block_oracle"].idxmin()]
    best_med = sw.loc[sw["med_regret_vs_block_oracle"].idxmin()]

    k_paper_sym = k_of(0.3, 6.0)
    k_paper_kur = k_of(0.7, 24.0)

    return {
        "dataset": name,
        "csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "n_blocks": int(len(df)),
        "block_N": BLOCK_N,
        "reimplementation_match": repro_match,
        "paper_stated_rule_match": paper_rule_match,
        "MSE": mse,
        "median_SE": med,
        "top10_block_share_of_total_SE": tails,
        "MSE_oracle_per_block": mse_oracle,
        "MSE_oracle_per_g4bin": mse_bin_oracle,
        "medSE_oracle_per_block": med_oracle,
        "regret_block_oracle": mse["AUTO"] / mse_oracle,
        "regret_block_oracle_ci95": list(ci),
        "regret_bin_oracle": mse["AUTO"] / mse_bin_oracle,
        "median_regret_block_oracle": med["AUTO"] / med_oracle,
        "mse_ratio_auto_over_ols": mse["AUTO"] / mse["OLS"],
        "median_ratio_auto_over_ols": med["AUTO"] / med["OLS"],
        # constant (no-dispatch) baselines — the bar the rule must clear
        "constant_policy_regret_block_oracle": {
            m: mse[m] / mse_oracle for m in METHODS + ["AUTO"]},
        "constant_policy_mse_ratio_vs_ols": {
            m: mse[m] / mse["OLS"] for m in METHODS + ["AUTO"]},
        "auto_distribution": {m: int(np.sum(auto == m)) for m in METHODS},
        "oracle_distribution": {m: int(np.sum(oracle == m)) for m in METHODS},
        "confusion": conf,
        "k_paper": {"skewness": k_paper_sym, "kurtosis": k_paper_kur},
        "k_best": float(best["k"]),
        "regret_at_k_best": float(best["regret_vs_block_oracle"]),
        "k_best_median": float(best_med["k"]),
        "median_regret_at_k_best": float(best_med["med_regret_vs_block_oracle"]),
        "by_g4_bin": by_bin,
        "bin_oracle_table": bin_table.to_dict(orient="records"),
        "_sweep": sw,
    }


# ------------------------------------------------------------------ outputs


def fig_regret_vs_k(results: list[dict], out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"polygauss": "#3274a1", "realnoise": "#e1812c"}
    for r in results:
        sw = r["_sweep"]
        c = colors.get(r["dataset"], None)
        ax.plot(sw["k"], sw["regret_vs_block_oracle"], "-", color=c,
                label=f"{r['dataset']} (n={r['n_blocks']})")
        kb = r["k_best"]
        ax.plot([kb], [r["regret_at_k_best"]], "o", color=c, ms=7)
    k_sym = k_of(0.3, 6.0)
    k_kur = k_of(0.7, 24.0)
    ax.axvline(k_sym, color="grey", linestyle="--", alpha=0.7,
               label=f"paper |γ̂₃|>0.3  (k={k_sym:.2f})")
    ax.axvline(k_kur, color="grey", linestyle=":", alpha=0.7,
               label=f"paper γ̂₄<−0.7  (k={k_kur:.2f})")
    ax.set_xlabel("k  (thresholds |γ̂₃| > k·√(6/N),  γ̂₄ < −k·√(24/N),  N=1024)")
    ax.set_ylabel("regret = MSE(rule) / MSE(per-block oracle)")
    ax.set_yscale("log")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5,
               label="regret = 1 (oracle)")
    ax.set_title("Exp 1A: dispatcher regret vs threshold scale k")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)


def fmt_report(results: list[dict]) -> str:
    lines = ["# Exp 1A — dispatcher audit (AUTO vs oracle-best)", ""]
    for r in results:
        lines.append(f"## {r['dataset']}  ({r['csv']}, n={r['n_blocks']}, N={r['block_N']})")
        lines.append(f"rule re-implementation matches `method_auto`: "
                     f"{100 * r['reimplementation_match']:.1f}%; "
                     f"paper-as-written rule matches: "
                     f"{100 * r['paper_stated_rule_match']:.1f}%")
        lines.append("")
        lines.append("MSE:    " + ", ".join(f"{k}={v:.3e}" for k, v in r["MSE"].items()))
        lines.append("medSE:  " + ", ".join(f"{k}={v:.3e}"
                                            for k, v in r["median_SE"].items()))
        lines.append("top-10-block share of ΣSE: "
                     + ", ".join(f"{k}={v:.3f}"
                                 for k, v in r["top10_block_share_of_total_SE"].items()))
        lines.append(f"MSE(oracle per-block)={r['MSE_oracle_per_block']:.3e}, "
                     f"MSE(oracle per γ̂₄-bin)={r['MSE_oracle_per_g4bin']:.3e}")
        ci = r["regret_block_oracle_ci95"]
        lines.append(f"regret vs per-block oracle = {r['regret_block_oracle']:.4f} "
                     f"[{ci[0]:.4f}; {ci[1]:.4f}]")
        lines.append(f"regret vs per-γ̂₄-bin oracle = {r['regret_bin_oracle']:.4f}")
        lines.append(f"median-based regret vs per-block oracle = "
                     f"{r['median_regret_block_oracle']:.4f}")
        lines.append(f"MSE(AUTO)/MSE(OLS) = {r['mse_ratio_auto_over_ols']:.4f}, "
                     f"medSE(AUTO)/medSE(OLS) = "
                     f"{r['median_ratio_auto_over_ols']:.4f}")
        lines.append("constant-policy regret (no dispatch): " + ", ".join(
            f"{m}={v:.3f}" for m, v in
            r["constant_policy_regret_block_oracle"].items()))
        lines.append(f"misclassification rate = "
                     f"{100 * r['confusion']['misclassification_rate']:.1f}%")
        lines.append("")
        lines.append("confusion (rows = AUTO, cols = oracle-best):")
        lines.append("| AUTO \\ oracle | " + " | ".join(METHODS) + " |")
        lines.append("|---|" + "---|" * len(METHODS))
        for a in METHODS:
            lines.append(f"| {a} | " +
                         " | ".join(str(r["confusion"]["matrix"][a][o]) for o in METHODS)
                         + " |")
        lines.append("")
        lines.append(f"k(paper): skew={r['k_paper']['skewness']:.3f}, "
                     f"kurt={r['k_paper']['kurtosis']:.3f}; "
                     f"k*={r['k_best']:.2f} (regret {r['regret_at_k_best']:.4f}); "
                     f"k*_median={r['k_best_median']:.2f} "
                     f"(median regret {r['median_regret_at_k_best']:.4f})")
        lines.append("")
        lines.append("misclassification by γ̂₄ decile:")
        lines.append("| γ̂₄ bin | n | misclass | regret |")
        lines.append("|---|---|---|---|")
        for b in r["by_g4_bin"]:
            lines.append(f"| {b['g4_bin']} | {b['n']} | "
                         f"{100 * b['misclass_rate']:.1f}% | {b['regret']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    k_grid = np.arange(0.5, 8.0 + 1e-9, 0.25)

    results = [analyse(name, path, k_grid) for name, path in DATASETS.items()]

    sweep_df = pd.concat([r["_sweep"] for r in results], ignore_index=True)
    sweep_df.to_csv(out_dir / "dispatch_sweep.csv", index=False)

    summary = [{k: v for k, v in r.items() if k != "_sweep"} for r in results]
    (out_dir / "dispatch_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    fig_regret_vs_k(results, out_dir / "fig1A_dispatch_regret_vs_k")

    report = fmt_report(results)
    (out_dir / "dispatch_report.txt").write_text(report)
    print(report)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    main(args.out_dir)
