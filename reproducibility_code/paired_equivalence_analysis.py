"""Paired seed-level equivalence analysis for the controlled experiments.

The analysis is intentionally separate from manuscript generation.  It compares
the dual-head model without a consistency penalty against the flat model on
matched seeds and writes both the individual paired differences and a complete
TOST summary.

The practical equivalence margin is fixed here, rather than estimated from the
observed results.  All metrics in ``e2e_test_runs.csv`` are already expressed in
percentage points, so ``0.25`` means an absolute quarter percentage point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


EQUIVALENCE_MARGIN_PP = 0.25
ALPHA = 0.05
DEFAULT_TARGET_PAIRS = 10
MARGIN_RATIONALE = (
    "A quarter percentage point is 0.0025 on the unit scale and, for accuracy, "
    "corresponds to one top-1 decision per 400 test images. The same deliberately "
    "strict absolute bound is applied to macro-F1 and cross-host error so the "
    "criterion cannot be relaxed metric by metric. It was fixed before the five "
    "additional paired seeds are analysed and is not estimated from their results."
)
METRICS = {
    "accuracy_pct": "higher_is_better",
    "macro_f1_pct": "higher_is_better",
    "cross_host_error_pct": "lower_is_better",
}
METRIC_LABELS = {
    "accuracy_pct": "Accuracy",
    "macro_f1_pct": "Macro-F1",
    "cross_host_error_pct": "CHER",
}
DISPLAY_NAMES = {
    "PV": "NPD-Rice-COCO-43",
    "DS1": "MultiCrop-88",
    "DS2": "Agri-Foundation-145k",
}


def mean_confidence_interval(
    differences: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    """Return a two-sided Student-t interval for the paired mean."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two paired differences are required")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    if standard_error == 0.0:
        return mean, mean
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    half_width = critical * standard_error
    return mean - half_width, mean + half_width


def bootstrap_mean_interval(
    differences: np.ndarray,
    confidence: float,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic paired percentile-bootstrap interval."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two paired differences are required")
    if iterations < 1000:
        raise ValueError("Use at least 1,000 bootstrap iterations")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def tost_paired(
    differences: np.ndarray,
    margin: float = EQUIVALENCE_MARGIN_PP,
    alpha: float = ALPHA,
) -> dict[str, float | bool]:
    """Two one-sided t tests for ``-margin < mean difference < margin``.

    The two TOST null hypotheses are ``mean <= -margin`` and
    ``mean >= +margin``.  Equivalence is established only when both are
    rejected.  The equivalent confidence-interval decision uses the 90%
    interval when ``alpha=0.05``.
    """

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two paired differences are required")
    if margin <= 0:
        raise ValueError("The equivalence margin must be positive")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    degrees_freedom = len(values) - 1
    if standard_error == 0.0:
        p_lower = 0.0 if mean > -margin else (0.5 if mean == -margin else 1.0)
        p_upper = 0.0 if mean < margin else (0.5 if mean == margin else 1.0)
    else:
        lower_statistic = (mean + margin) / standard_error
        upper_statistic = (mean - margin) / standard_error
        p_lower = float(student_t.sf(lower_statistic, degrees_freedom))
        p_upper = float(student_t.cdf(upper_statistic, degrees_freedom))
    interval_confidence = 1.0 - 2.0 * alpha
    ci_lower, ci_upper = mean_confidence_interval(values, interval_confidence)
    tost_p = max(p_lower, p_upper)
    equivalent = bool(
        p_lower < alpha
        and p_upper < alpha
        and ci_lower > -margin
        and ci_upper < margin
    )
    return {
        "tost_lower_p": p_lower,
        "tost_upper_p": p_upper,
        "tost_p": tost_p,
        "tost_ci_confidence": interval_confidence,
        "tost_ci_lower_pp": ci_lower,
        "tost_ci_upper_pp": ci_upper,
        "equivalent_unadjusted": equivalent,
    }


def holm_adjusted_pvalues(p_values: np.ndarray) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    adjusted_sorted = np.empty(len(values), dtype=float)
    running_maximum = 0.0
    number = len(values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (number - rank) * values[original_index])
        running_maximum = max(running_maximum, candidate)
        adjusted_sorted[rank] = running_maximum
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def validate_runs(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    required = {"dataset", "configuration", "seed", "mode", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    selected = frame[frame["mode"] == mode].copy()
    if selected.empty:
        raise ValueError(f"No rows use mode={mode!r}")
    duplicate = selected.duplicated(["dataset", "configuration", "seed"], keep=False)
    if duplicate.any():
        examples = selected.loc[
            duplicate, ["dataset", "configuration", "seed"]
        ].drop_duplicates()
        raise ValueError(
            "Each dataset/configuration/seed must be unique after mode filtering; "
            f"duplicates include {examples.head().to_dict(orient='records')}"
        )
    for metric in METRICS:
        selected[metric] = pd.to_numeric(selected[metric], errors="raise")
        if not np.isfinite(selected[metric]).all():
            raise ValueError(f"{metric} contains a non-finite value")
    selected["seed"] = pd.to_numeric(selected["seed"], errors="raise").astype(int)
    return selected


def load_controlled_runs(runs_root: Path) -> tuple[pd.DataFrame, str]:
    """Load legacy rows and any compatible manifest-run completions.

    Combining both sources is required when seeds 2026--2030 come from the
    original controlled study and the preregistered extension contributes seeds
    2031--2035 under the same runs root.
    """

    legacy_path = runs_root / "e2e_test_runs.csv"
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    if legacy_path.is_file():
        frames.append(pd.read_csv(legacy_path))
        sources.append(str(legacy_path))
    rows: list[dict] = []
    completion_paths = sorted(runs_root.rglob("COMPLETE.json"))
    for completion_path in completion_paths:
        record = json.loads(completion_path.read_text(encoding="utf-8"))
        metrics = record.get("metrics", {}).get("test")
        if not isinstance(metrics, dict):
            continue
        dataset = record.get("dataset_label") or record.get("dataset")
        configuration = record.get("configuration")
        seed = record.get("seed")
        if dataset is None or configuration is None or seed is None:
            continue
        rows.append(
            {
                "dataset": str(dataset),
                "configuration": str(configuration),
                "seed": int(seed),
                "mode": "standard",
                **metrics,
                "completion_path": str(completion_path.resolve()),
            }
        )
    if rows:
        frames.append(pd.DataFrame(rows))
        sources.append(f"{runs_root.resolve()}::COMPLETE.json")
    if not frames:
        raise FileNotFoundError(
            f"Neither {legacy_path} nor compatible COMPLETE.json files were found"
        )
    return pd.concat(frames, ignore_index=True, sort=False), " + ".join(sources)


def analyse(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    target_pairs: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    difference_rows: list[dict] = []
    summary_rows: list[dict] = []
    pairing_audit: list[dict] = []
    configurations = set(frame["configuration"])
    for name in (candidate, reference):
        if name not in configurations:
            raise ValueError(f"Configuration {name!r} is absent from the input")

    for dataset_index, (dataset, group) in enumerate(
        sorted(frame.groupby("dataset"), key=lambda item: item[0])
    ):
        candidate_frame = group[group["configuration"] == candidate].set_index("seed")
        reference_frame = group[group["configuration"] == reference].set_index("seed")
        candidate_seeds = set(candidate_frame.index)
        reference_seeds = set(reference_frame.index)
        paired_seeds = sorted(candidate_seeds.intersection(reference_seeds))
        pairing_audit.append(
            {
                "dataset": dataset,
                "paired_seeds": paired_seeds,
                "candidate_only_seeds": sorted(candidate_seeds - reference_seeds),
                "reference_only_seeds": sorted(reference_seeds - candidate_seeds),
                "target_pairs": target_pairs,
                "target_pairs_met": len(paired_seeds) >= target_pairs,
            }
        )
        if len(paired_seeds) < 2:
            continue
        for metric_index, (metric, direction) in enumerate(METRICS.items()):
            candidate_values = candidate_frame.loc[paired_seeds, metric].to_numpy(float)
            reference_values = reference_frame.loc[paired_seeds, metric].to_numpy(float)
            differences = candidate_values - reference_values
            for seed, candidate_value, reference_value, difference in zip(
                paired_seeds,
                candidate_values,
                reference_values,
                differences,
            ):
                difference_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{candidate}_minus_{reference}",
                        "candidate": candidate,
                        "reference": reference,
                        "seed": seed,
                        "metric": metric,
                        "direction_of_improvement": direction,
                        "candidate_value_pct": candidate_value,
                        "reference_value_pct": reference_value,
                        "paired_difference_pp": difference,
                        "within_equivalence_margin": bool(
                            -EQUIVALENCE_MARGIN_PP
                            < difference
                            < EQUIVALENCE_MARGIN_PP
                        ),
                    }
                )
            mean = float(differences.mean())
            standard_deviation = float(differences.std(ddof=1))
            standard_error = standard_deviation / np.sqrt(len(differences))
            t95_lower, t95_upper = mean_confidence_interval(differences, 0.95)
            bootstrap_lower, bootstrap_upper = bootstrap_mean_interval(
                differences,
                confidence=0.95,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + 100 * dataset_index + metric_index,
            )
            tost = tost_paired(differences)
            target_met = len(paired_seeds) >= target_pairs
            summary_rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{candidate}_minus_{reference}",
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "direction_of_improvement": direction,
                    "n_pairs": len(paired_seeds),
                    "paired_seeds": " ".join(str(seed) for seed in paired_seeds),
                    "target_pairs": target_pairs,
                    "target_pairs_met": target_met,
                    "equivalence_margin_lower_pp": -EQUIVALENCE_MARGIN_PP,
                    "equivalence_margin_upper_pp": EQUIVALENCE_MARGIN_PP,
                    "mean_difference_pp": mean,
                    "std_difference_pp": standard_deviation,
                    "standard_error_pp": standard_error,
                    "t_95ci_lower_pp": t95_lower,
                    "t_95ci_upper_pp": t95_upper,
                    "bootstrap_95ci_lower_pp": bootstrap_lower,
                    "bootstrap_95ci_upper_pp": bootstrap_upper,
                    **tost,
                }
            )

    differences_frame = pd.DataFrame(difference_rows)
    summary_frame = pd.DataFrame(summary_rows)
    if summary_frame.empty:
        raise RuntimeError("No dataset had at least two matched candidate/reference seeds")
    summary_frame["holm_adjusted_tost_p"] = holm_adjusted_pvalues(
        summary_frame["tost_p"].to_numpy(float)
    )
    summary_frame["equivalent_holm_0_05"] = (
        summary_frame["equivalent_unadjusted"]
        & (summary_frame["holm_adjusted_tost_p"] < ALPHA)
    )
    summary_frame["claim_ready"] = (
        summary_frame["target_pairs_met"]
        & summary_frame["equivalent_holm_0_05"]
    )
    summary_frame["analysis_status"] = np.where(
        ~summary_frame["target_pairs_met"],
        "interim_only_fewer_than_target_pairs",
        np.where(
            summary_frame["equivalent_holm_0_05"],
            "equivalent_within_predeclared_margin",
            "equivalence_not_established",
        ),
    )
    return differences_frame, summary_frame, pairing_audit


def make_equivalence_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    available = set(summary["dataset"].unique())
    datasets = [dataset for dataset in ("PV", "DS1", "DS2") if dataset in available]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(7.3, 2.75),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for column, dataset in enumerate(datasets):
        axis = axes[0, column]
        selected = summary[summary["dataset"] == dataset].set_index("metric")
        axis.axvspan(
            -EQUIVALENCE_MARGIN_PP,
            EQUIVALENCE_MARGIN_PP,
            color="#DCEBDD",
            alpha=0.8,
            zorder=0,
            label="Equivalence region",
        )
        axis.axvline(0.0, color="#52616B", linewidth=0.9, linestyle=":")
        for row_index, metric in enumerate(METRICS):
            if metric not in selected.index:
                continue
            row = selected.loc[metric]
            mean = float(row["mean_difference_pp"])
            lower = float(row["t_95ci_lower_pp"])
            upper = float(row["t_95ci_upper_pp"])
            if not bool(row["target_pairs_met"]):
                colour = "#6B7785"
            elif bool(row["equivalent_holm_0_05"]):
                colour = "#2E7D5B"
            else:
                colour = "#B85C38"
            axis.errorbar(
                mean,
                row_index,
                xerr=[[mean - lower], [upper - mean]],
                fmt="o",
                color=colour,
                ecolor=colour,
                capsize=3,
                markersize=5,
                linewidth=1.25,
                zorder=2,
            )
        axis.set_title(DISPLAY_NAMES.get(dataset, dataset), fontweight="bold", fontsize=9)
        axis.set_xlabel("Dual - flat (percentage points)")
        axis.grid(axis="x", alpha=0.18)
        axis.set_yticks(
            np.arange(len(METRICS)),
            [METRIC_LABELS[metric] for metric in METRICS],
        )
        axis.invert_yaxis()
        axis.text(
            0.98,
            0.03,
            f"n={int(selected['n_pairs'].min())} paired seeds",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#52616B",
        )
    for suffix, options in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(
            output_dir / f"paired_equivalence_forest.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **options,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate", default="dual_none")
    parser.add_argument("--reference", default="flat")
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--target-pairs", type=int, default=DEFAULT_TARGET_PAIRS)
    parser.add_argument("--bootstrap-iterations", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1701)
    parser.add_argument(
        "--strict-target",
        action="store_true",
        help="Exit unsuccessfully if any dataset has fewer than --target-pairs.",
    )
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    if args.target_pairs < 2:
        raise ValueError("--target-pairs must be at least two")

    runs_root = args.runs_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else runs_root / "paired_equivalence"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_frame, input_source = load_controlled_runs(runs_root)
    frame = validate_runs(raw_frame, args.mode)
    differences, summary, pairing_audit = analyse(
        frame,
        candidate=args.candidate,
        reference=args.reference,
        target_pairs=args.target_pairs,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    differences.to_csv(output_dir / "paired_seed_differences.csv", index=False)
    summary.to_csv(output_dir / "paired_equivalence_summary.csv", index=False)
    protocol = {
        "input": input_source,
        "candidate_minus_reference": f"{args.candidate}_minus_{args.reference}",
        "mode": args.mode,
        "metrics": list(METRICS),
        "metric_units": "percentage_points",
        "equivalence_margin_pp": EQUIVALENCE_MARGIN_PP,
        "alpha": ALPHA,
        "tost_interval_confidence": 1.0 - 2.0 * ALPHA,
        "reported_interval_confidence": 0.95,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "multiplicity_control": (
            "Holm adjustment across every dataset-by-metric TOST p-value"
        ),
        "target_pairs": args.target_pairs,
        "margin_provenance": (
            "Fixed by the revision protocol before analysis of additional seeds; "
            "not estimated from the observed paired differences."
        ),
        "margin_rationale": MARGIN_RATIONALE,
        "difference_definition": "candidate value minus reference value",
        "pairing_audit": pairing_audit,
    }
    (output_dir / "paired_equivalence_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    if not args.no_figure:
        make_equivalence_figure(summary, output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote paired equivalence artifacts to {output_dir}")
    if args.strict_target and not summary["target_pairs_met"].all():
        missing = summary.loc[
            ~summary["target_pairs_met"], ["dataset", "n_pairs", "target_pairs"]
        ].drop_duplicates()
        raise SystemExit(
            "Target pair count is not yet met:\n" + missing.to_string(index=False)
        )


if __name__ == "__main__":
    main()
