"""Verification protocols for DSGE experiments.

These checks codify the verification standards from the
kunchenko-research-workflow skill (Verification Standard 8 + PATP checks):

    - bootstrap p-value for hybrid > baseline (row-level OR cluster/grouped)
    - 95% CI for the macro-F1 improvement
    - basis comparison ablation
    - PATP boundary check (best alpha not at 0 or 1)
    - F-matrix conditioning report

Use these BEFORE submitting a paper or pushing a release. They take a
few seconds to run and catch common publication-killer mistakes.

Public API:
    bootstrap_compare(y_true, y_pred_baseline, y_pred_hybrid, R=1000,
                      groups=None)
        -> {'delta', 'p_value', 'ci_low', 'ci_high', 'significant',
            'resampling_unit', 'n_groups'}

    verification_report(results, ...) -> str
        Pretty-prints PASS/FAIL for each Verification Standard.

Verification Standard 9 — the resampling unit must be the dependence unit.
Pass `groups=` (subject_id for HAR, burst_id/recording_id for RF, specimen or
campaign for acoustic emission, author for NLP) whenever test rows are not
independent. Row-level resampling on grouped data treats `n_rows` as the sample
size when it is `n_groups`: the CI shrinks by roughly sqrt(rows per group) and
the p-value collapses. See `references/verification.md` section V3a.
"""

from __future__ import annotations

import warnings

import numpy as np

MIN_GROUPS_FOR_CLUSTER_BOOTSTRAP = 10


def _metric_fn(metric: str):
    from sklearn.metrics import accuracy_score, f1_score
    if metric == "macro_f1":
        return lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    if metric == "accuracy":
        return accuracy_score
    raise ValueError(f"Unknown metric '{metric}'.")


def bootstrap_compare(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_hybrid: np.ndarray,
    R: int = 1000,
    metric: str = "macro_f1",
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> dict:
    """Paired bootstrap test on macro-F1 (or accuracy) improvement.

    H0: hybrid model gives the same metric as baseline.
    Returns p-value (two-sided), 95% CI, and `significant` flag at p < 0.05.

    R = 1000 is the standard from the NLP paper (matches the
    kunchenko-research-workflow Verification Standard 8). Pass R = 5000+
    for tighter CIs on small test sets.

    Args:
        groups: dependence-unit id per test row (subject / burst / recording /
            specimen / campaign / author). When given, GROUPS are resampled
            with replacement and all rows of each drawn group are kept
            (cluster bootstrap) — this is Verification Standard 9 and is
            **required** for grouped data. When ``None``, rows are resampled
            i.i.d., which is only valid if the rows really are independent
            (e.g. fresh Monte Carlo replicates).

    Raises:
        ValueError: if `groups` has the wrong length, or `y_true` is empty.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_baseline = np.asarray(y_pred_baseline)
    y_pred_hybrid = np.asarray(y_pred_hybrid)
    n = len(y_true)
    if n == 0:
        raise ValueError("y_true is empty.")

    score = _metric_fn(metric)

    if groups is None:
        unit, n_groups = "row", n
        draw = lambda: rng.integers(0, n, size=n)
    else:
        groups = np.asarray(groups)
        if groups.shape[0] != n:
            raise ValueError(
                f"groups has length {groups.shape[0]}, expected {n} "
                "(one dependence-unit id per test row)."
            )
        uniq = np.unique(groups)
        n_groups = int(uniq.size)
        idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
        if n_groups < MIN_GROUPS_FOR_CLUSTER_BOOTSTRAP:
            warnings.warn(
                f"n_groups = {n_groups} < {MIN_GROUPS_FOR_CLUSTER_BOOTSTRAP}: the "
                "cluster bootstrap is unreliable here. Report a paired test over "
                "repeated leakage-safe splits (Wilcoxon signed-rank) instead, and "
                "treat this CI as indicative only.",
                RuntimeWarning,
                stacklevel=2,
            )
        unit = "group"
        draw = lambda: np.concatenate(
            [idx_by_group[g] for g in rng.choice(uniq, size=uniq.size, replace=True)]
        )

    deltas = np.zeros(R)
    for r in range(R):
        idx = draw()
        s_b = score(y_true[idx], y_pred_baseline[idx])
        s_h = score(y_true[idx], y_pred_hybrid[idx])
        deltas[r] = s_h - s_b

    point = score(y_true, y_pred_hybrid) - score(y_true, y_pred_baseline)
    # Two-sided p-value: fraction of bootstrap deltas at least as extreme
    # as the observed point estimate, under the null of no difference.
    # Clipped at 1.0 — when delta is exactly 0 both tail fractions include the
    # ties, so 2*min(...) can exceed 1 (observed as p = 1.198 in the drilling
    # control of results/stats_basis/stats_basis.json before this fix).
    p_value = float(min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))))
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(point),
        "p_value": p_value,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool(p_value < 0.05 and point > 0),
        "metric": metric,
        "R": R,
        "resampling_unit": unit,
        "n_groups": n_groups,
    }


def patp_boundary_check(alphas: list[float], scores: list[float],
                        margin: float = 0.05) -> dict:
    """Confirm that the optimal alpha is not at the [0, 1] boundary.

    If alpha_opt sits at 0 or 1 (within `margin`), the corresponding fixed
    discrete basis is sufficient and PATP gives no benefit.

    Returns:
        {'best_alpha', 'at_boundary', 'recommendation'}
    """
    best_idx = int(np.argmax(scores))
    best_alpha = float(alphas[best_idx])
    at_boundary = best_alpha < margin or best_alpha > (1.0 - margin)
    rec = (
        f"alpha_opt = {best_alpha:.2f} is at the boundary; use the "
        f"corresponding fixed basis ({'fractional' if best_alpha < margin else 'integer-power'})."
        if at_boundary else
        f"alpha_opt = {best_alpha:.2f} is interior; PATP gives genuine benefit "
        "over discrete bases."
    )
    return {"best_alpha": best_alpha, "at_boundary": at_boundary, "recommendation": rec}


def conditioning_report(extractor) -> dict:
    """Return per-class cond(F_reg) and a recommendation if poorly conditioned."""
    cond = extractor.conditioning_report
    high = {c: v for c, v in cond.items() if v > 1e6}
    rec = "OK" if not high else (
        f"Classes {list(high)} have cond(F) > 1e6. "
        "Consider increasing `ridge`, decreasing `n`, or trying a smoother basis."
    )
    return {"cond_per_class": cond, "warning": rec}


def verification_report(
    *,
    accuracy_dsge_only: float | None = None,
    accuracy_baseline: float | None = None,
    accuracy_hybrid: float | None = None,
    bootstrap_result: dict | None = None,
    basis_comparison: list[dict] | None = None,
    patp_check: dict | None = None,
    cond_report: dict | None = None,
    independent_rows: bool = False,
) -> str:
    """Produce a PASS/FAIL summary aligned with the kunchenko-research-workflow
    Verification Standards 8 (DSGE), 9 (resampling unit) and N (PATP).

    Pass any subset of arguments — only the corresponding checks are run.

    Args:
        independent_rows: set True only if test rows genuinely are independent
            (fresh Monte Carlo replicates, synthetic data sampled per row). It
            makes the Standard 9 check pass on a row-level bootstrap. Asserting
            it on grouped data is how pseudo-replication gets published.
    """
    lines = ["=== DSGE Verification Report ==="]
    passed = total = 0

    def _check(label: str, condition: bool, detail: str = ""):
        nonlocal passed, total
        total += 1
        passed += int(condition)
        tag = "[PASS]" if condition else "[FAIL]"
        lines.append(f"{tag} {label}{(' — ' + detail) if detail else ''}")

    if accuracy_dsge_only is not None and accuracy_baseline is not None:
        _check(
            "DSGE-only beats trivial baseline",
            accuracy_dsge_only > 0.5 + 1e-6,
            f"acc_dsge = {accuracy_dsge_only:.3f}",
        )

    if accuracy_hybrid is not None and accuracy_baseline is not None:
        _check(
            "Hybrid > baseline",
            accuracy_hybrid > accuracy_baseline,
            f"hybrid = {accuracy_hybrid:.3f} vs baseline = {accuracy_baseline:.3f}",
        )

    if bootstrap_result is not None:
        _check(
            "Bootstrap p < 0.05 (Verification Standard 8)",
            bootstrap_result["significant"],
            f"p = {bootstrap_result['p_value']:.4f}, "
            f"95% CI [{bootstrap_result['ci_low']:+.4f}, {bootstrap_result['ci_high']:+.4f}]",
        )
        unit = bootstrap_result.get("resampling_unit", "row")
        n_groups = bootstrap_result.get("n_groups")
        _check(
            "Resampling unit = dependence unit (Verification Standard 9)",
            unit == "group" or independent_rows,
            (
                f"unit = {unit}, n_groups = {n_groups}"
                if unit == "group" else
                f"unit = row, n = {n_groups}; rows declared independent"
                if independent_rows else
                "rows resampled i.i.d. and independence NOT declared — pass "
                "`groups=` to bootstrap_compare, or set independent_rows=True "
                "if the rows really are independent"
            ),
        )
        if unit == "group" and isinstance(n_groups, int) \
                and n_groups < MIN_GROUPS_FOR_CLUSTER_BOOTSTRAP:
            lines.append(
                f"[WARN] n_groups = {n_groups} < {MIN_GROUPS_FOR_CLUSTER_BOOTSTRAP}: "
                "report a paired test over repeated leakage-safe splits "
                "(Wilcoxon signed-rank); treat this CI as indicative."
            )

    if basis_comparison:
        bases = [r["basis"] for r in basis_comparison]
        f1s = [r["macro_f1"] for r in basis_comparison]
        best = bases[int(np.argmax(f1s))]
        _check(
            "At least one basis gives non-trivial F1",
            max(f1s) > 0.5,
            f"best basis = {best} (F1 = {max(f1s):.3f})",
        )

    if patp_check is not None:
        _check(
            "PATP alpha_opt is interior (not at boundary)",
            not patp_check["at_boundary"],
            patp_check["recommendation"],
        )

    if cond_report is not None:
        _check(
            "F-matrix well-conditioned (cond < 1e6 per class)",
            "OK" in cond_report["warning"],
            cond_report["warning"],
        )

    lines.append(f"=== {passed}/{total} checks passed ===")
    return "\n".join(lines)
