"""Generate probability-quality diagnostics from the cached Swin logits.

The script performs no model inference.  It reads the development and test
logits written by ``advanced_inference_analysis.py`` and regenerates the
calibration, selective-prediction, class-support, and hierarchy-aware artifacts
used by the manuscript and supplement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


DATASETS = ("PV", "DS1", "DS2")
DISPLAY_NAMES = {
    "PV": "NPD–Rice–COCO–43",
    "DS1": "MultiCrop–88",
    "DS2": "Agri–Foundation–145k",
}
RAW = "#C46A3A"
SCALED = "#176B87"
NAVY = "#17365D"
GREY = "#6B7785"
PALE_GREY = "#D7DEE5"
TIER_COLOURS = {
    "tail": "#C46A3A",
    "medium": "#D7A928",
    "head": "#2E7D5B",
}


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(value)


def nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(selected.clip(1e-15)).mean())


def multiclass_brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    selected = probabilities[np.arange(len(labels)), labels]
    squared_probability = np.square(probabilities).sum(axis=1)
    return float(np.mean(squared_probability - 2.0 * selected + 1.0))


def per_class_average_precision(
    labels: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    scores = np.zeros(probabilities.shape[1], dtype=float)
    for class_index in range(probabilities.shape[1]):
        binary_labels = labels == class_index
        scores[class_index] = average_precision_score(
            binary_labels, probabilities[:, class_index]
        )
    return scores


def aurc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Area under the confidence-ranked selective-risk curve.

    Coverage is swept over every retained prefix after sorting examples by
    decreasing maximum probability.  The returned value is a proportion, not a
    percentage.
    """

    confidence = probabilities.max(axis=1)
    errors = probabilities.argmax(axis=1) != labels
    order = np.argsort(-confidence, kind="mergesort")
    cumulative_errors = np.cumsum(errors[order])
    retained = np.arange(1, len(labels) + 1)
    risk = cumulative_errors / retained
    coverage = retained / len(labels)
    return float(np.trapezoid(risk, coverage))


def reliability_bins(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    mean_confidence: list[float] = []
    mean_accuracy: list[float] = []
    mass: list[float] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            mean_confidence.append(float(confidence[selected].mean()))
            mean_accuracy.append(float(correct[selected].mean()))
            mass.append(float(selected.mean()))
    return (
        np.asarray(mean_confidence),
        np.asarray(mean_accuracy),
        np.asarray(mass),
    )


def clean_condition_name(name: str, length: int = 29) -> str:
    value = name.replace("___", ": ").replace("__", ": ").replace("_", " ")
    value = " ".join(value.split())
    if len(value) > length:
        return value[: length - 1].rstrip() + "…"
    return value


def load_metadata(results_dir: Path, dataset: str) -> dict:
    path = results_dir / f"metrics_{dataset}_test_hierarchical.json"
    return json.loads(path.read_text(encoding="utf-8"))["model_metadata"]


def make_reliability_figure(
    cache: dict[str, dict[str, np.ndarray | float]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.3, 4.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 0.75]},
    )
    for column, dataset in enumerate(DATASETS):
        labels = cache[dataset]["labels"]
        raw_probability = cache[dataset]["raw"]
        scaled_probability = cache[dataset]["scaled"]
        axis = axes[0, column]
        axis.plot(
            [0.5, 1.0],
            [0.5, 1.0],
            color=GREY,
            linewidth=1.1,
            linestyle=":",
            label="Perfect",
        )
        for name, probability, colour, marker, linestyle in (
            ("Raw", raw_probability, RAW, "o", "--"),
            ("Scaled", scaled_probability, SCALED, "s", "-"),
        ):
            confidence, accuracy, mass = reliability_bins(labels, probability)
            retained = confidence >= 0.5
            sizes = 16.0 + 72.0 * np.sqrt(mass[retained])
            axis.plot(
                confidence[retained],
                accuracy[retained],
                color=colour,
                marker=marker,
                markersize=4.2,
                linewidth=1.45,
                linestyle=linestyle,
                label=name,
                zorder=2,
            )
            axis.scatter(
                confidence[retained],
                accuracy[retained],
                s=sizes,
                facecolor="white",
                edgecolor=colour,
                linewidth=0.8,
                zorder=3,
            )
        axis.set_xlim(0.5, 1.005)
        axis.set_ylim(0.5, 1.005)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18)
        axis.set_title(DISPLAY_NAMES[dataset], fontsize=11, fontweight="bold")
        if column == 0:
            axis.set_ylabel("Observed accuracy")
        axis.set_xlabel("Mean confidence")
        axis.text(
            0.03,
            0.96,
            (
                f"ECE: {100 * ece(labels, raw_probability):.3f}"
                f" → {100 * ece(labels, scaled_probability):.3f}%"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color=NAVY,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": PALE_GREY,
                "alpha": 0.92,
            },
        )
        if column == 2:
            axis.legend(loc="lower right", frameon=False, fontsize=8)

        hist_axis = axes[1, column]
        bins = np.linspace(0.5, 1.0, 26)
        hist_axis.hist(
            raw_probability.max(axis=1),
            bins=bins,
            histtype="step",
            linewidth=1.5,
            color=RAW,
            label="Raw",
        )
        hist_axis.hist(
            scaled_probability.max(axis=1),
            bins=bins,
            histtype="stepfilled",
            linewidth=1.2,
            facecolor=f"{SCALED}35",
            edgecolor=SCALED,
            label="Scaled",
        )
        hist_axis.set_yscale("log")
        hist_axis.set_xlim(0.5, 1.0)
        hist_axis.grid(axis="y", alpha=0.18)
        hist_axis.set_xlabel("Top-class confidence")
        if column == 0:
            hist_axis.set_ylabel("Test images (log)")
        if column == 2:
            hist_axis.legend(loc="upper left", frameon=False, fontsize=8)

    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(
            output_dir / f"calibration_diagnostics.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def make_class_hierarchy_figure(
    per_class: pd.DataFrame,
    hierarchy: pd.DataFrame,
    output_dir: Path,
) -> None:
    dataset = "DS2"
    class_frame = per_class[per_class["dataset"] == dataset].copy()
    pair_frame = (
        hierarchy[hierarchy["dataset"] == dataset]
        .sort_values(["error_count", "true_host_rate_pct"], ascending=False)
        .head(10)
        .sort_values("error_count")
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.35), constrained_layout=True)

    left = axes[0]
    for tier in ("tail", "medium", "head"):
        selected = class_frame[class_frame["frequency_tier"] == tier]
        left.scatter(
            selected["development_support"],
            100 * selected["average_precision"],
            s=25 + 2.2 * np.sqrt(selected["test_support"]),
            alpha=0.78,
            color=TIER_COLOURS[tier],
            edgecolor="white",
            linewidth=0.45,
            label=tier.title(),
        )
    macro_ap = 100 * class_frame["average_precision"].mean()
    left.axhline(
        macro_ap,
        color=NAVY,
        linewidth=1.1,
        linestyle=":",
        label=f"Macro AP {macro_ap:.2f}%",
    )
    left.set_xscale("log")
    left.set_xlabel("Development images per class (log scale)")
    left.set_ylabel("Test average precision (%)")
    left.set_title(
        "Class probability quality vs. support",
        fontsize=10.5,
        fontweight="bold",
    )
    left.set_ylim(48, 101)
    left.grid(alpha=0.2)
    left.legend(frameon=False, fontsize=7.7, loc="lower right")

    right = axes[1]
    labels = [
        f"{true.replace('_', ' ').title()} → {pred.replace('_', ' ').title()}"
        for true, pred in zip(pair_frame["true_host"], pair_frame["predicted_host"])
    ]
    bars = right.barh(
        np.arange(len(pair_frame)),
        pair_frame["error_count"],
        color="#3D7E9A",
        alpha=0.9,
    )
    right.set_yticks(np.arange(len(pair_frame)), labels, fontsize=7.5)
    right.set_xlabel("Cross-host errors (test images)")
    right.set_title(
        "Largest directed crop-host confusions",
        fontsize=10.5,
        fontweight="bold",
    )
    right.grid(axis="x", alpha=0.2)
    for bar, (_, row) in zip(bars, pair_frame.iterrows()):
        right.text(
            bar.get_width() + max(pair_frame["error_count"]) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['error_count'])}  ({row['true_host_rate_pct']:.1f}%)",
            va="center",
            fontsize=7.2,
            color="#374151",
        )
    right.set_xlim(0, max(pair_frame["error_count"]) * 1.27)

    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(
            output_dir / f"class_hierarchy_diagnostics.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.package_root.resolve()
    results_dir = root / "results/revised"
    analysis_dir = results_dir / "advanced_inference"
    output_dir = root / "figures/revised"
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = json.loads(
        (analysis_dir / "selection.json").read_text(encoding="utf-8")
    )

    metric_rows: list[dict] = []
    class_rows: list[dict] = []
    hierarchy_rows: list[dict] = []
    operating_rows: list[dict] = []
    cache: dict[str, dict[str, np.ndarray | float]] = {}

    for dataset in DATASETS:
        archive = np.load(analysis_dir / f"logits_{dataset}.npz")
        metadata = load_metadata(results_dir, dataset)
        mapping = np.asarray(metadata["condition_to_species"], dtype=int)
        class_names = list(metadata["condition_classes"])
        host_names = list(metadata["species_names"])
        labels = archive["test_labels"]
        raw_probability = softmax(archive["test_condition"], axis=1)
        temperature = float(selections[dataset]["condition_temperature"])
        scaled_probability = softmax(
            archive["test_condition"] / temperature, axis=1
        )
        cache[dataset] = {
            "labels": labels,
            "raw": raw_probability,
            "scaled": scaled_probability,
            "temperature": temperature,
        }

        for calibration, probability in (
            ("raw", raw_probability),
            ("temperature_scaled", scaled_probability),
        ):
            class_ap = per_class_average_precision(labels, probability)
            metric_rows.append(
                {
                    "dataset": dataset,
                    "calibration": calibration,
                    "temperature": 1.0 if calibration == "raw" else temperature,
                    "accuracy_pct": 100
                    * np.mean(probability.argmax(axis=1) == labels),
                    "ece15_pct": 100 * ece(labels, probability),
                    "nll": nll(labels, probability),
                    "multiclass_brier": multiclass_brier(labels, probability),
                    "macro_ovr_average_precision_pct": 100 * class_ap.mean(),
                    "minimum_class_average_precision_pct": 100 * class_ap.min(),
                    "aurc": aurc(labels, probability),
                }
            )

        scaled_predictions = scaled_probability.argmax(axis=1)
        precision, recall, f1, test_support = precision_recall_fscore_support(
            labels,
            scaled_predictions,
            labels=np.arange(len(class_names)),
            zero_division=0,
        )
        class_ap = per_class_average_precision(labels, scaled_probability)
        development_support = np.bincount(
            archive["val_labels"], minlength=len(class_names)
        )
        positive_support = development_support[development_support > 0]
        first_quartile, third_quartile = np.quantile(
            positive_support, (0.25, 0.75)
        )
        for class_index, class_name in enumerate(class_names):
            support = development_support[class_index]
            if support <= first_quartile:
                tier = "tail"
            elif support >= third_quartile:
                tier = "head"
            else:
                tier = "medium"
            class_rows.append(
                {
                    "dataset": dataset,
                    "class_index": class_index,
                    "class_name": class_name,
                    "host_index": int(mapping[class_index]),
                    "host_name": host_names[mapping[class_index]],
                    "development_support": int(support),
                    "test_support": int(test_support[class_index]),
                    "precision": float(precision[class_index]),
                    "recall": float(recall[class_index]),
                    "f1": float(f1[class_index]),
                    "average_precision": float(class_ap[class_index]),
                    "frequency_tier": tier,
                }
            )

        true_hosts = mapping[labels]
        predicted_hosts = mapping[scaled_predictions]
        true_host_counts = np.bincount(true_hosts, minlength=len(host_names))
        for true_host in range(len(host_names)):
            for predicted_host in range(len(host_names)):
                if true_host == predicted_host:
                    continue
                count = int(
                    np.sum(
                        (true_hosts == true_host)
                        & (predicted_hosts == predicted_host)
                    )
                )
                if count == 0:
                    continue
                hierarchy_rows.append(
                    {
                        "dataset": dataset,
                        "true_host_index": true_host,
                        "true_host": host_names[true_host],
                        "predicted_host_index": predicted_host,
                        "predicted_host": host_names[predicted_host],
                        "error_count": count,
                        "true_host_support": int(true_host_counts[true_host]),
                        "true_host_rate_pct": 100
                        * count
                        / true_host_counts[true_host],
                    }
                )

        confidence = scaled_probability.max(axis=1)
        errors = scaled_predictions != labels
        order = np.argsort(-confidence, kind="mergesort")
        cumulative_error = np.cumsum(errors[order])
        retained = np.arange(1, len(labels) + 1)
        risk = cumulative_error / retained
        coverage = retained / len(labels)
        for target_risk in (0.01, 0.02):
            feasible = np.flatnonzero(risk <= target_risk)
            index = int(feasible[-1])
            operating_rows.append(
                {
                    "dataset": dataset,
                    "target_risk_pct": 100 * target_risk,
                    "confidence_threshold": float(confidence[order][index]),
                    "coverage_pct": 100 * float(coverage[index]),
                    "retained_images": int(retained[index]),
                    "selective_risk_pct": 100 * float(risk[index]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(class_rows)
    hierarchy = pd.DataFrame(hierarchy_rows)
    operating = pd.DataFrame(operating_rows)

    metrics.to_csv(
        analysis_dir / "probability_quality_metrics.csv", index=False
    )
    per_class.to_csv(
        analysis_dir / "per_class_probability_metrics.csv", index=False
    )
    hierarchy.to_csv(
        analysis_dir / "hierarchy_confusion_pairs.csv", index=False
    )
    operating.to_csv(
        analysis_dir / "selective_operating_points.csv", index=False
    )

    support_rows = []
    for dataset in DATASETS:
        selected = per_class[per_class["dataset"] == dataset]
        correlation, p_value = spearmanr(
            np.log10(selected["development_support"] + 1),
            selected["average_precision"],
        )
        support_rows.append(
            {
                "dataset": dataset,
                "spearman_log_support_vs_average_precision": correlation,
                "two_sided_p_value_descriptive": p_value,
                "classes": len(selected),
            }
        )
    pd.DataFrame(support_rows).to_csv(
        analysis_dir / "support_ap_association.csv", index=False
    )

    efficiency_rows = []
    for metrics_path in sorted(results_dir.glob("metrics_*_test_*.json")):
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        latency = record.get("latency", {})
        if not latency.get("available"):
            continue
        checkpoint = root / "checkpoints" / record["checkpoint"]
        milliseconds = float(latency["mean_ms_per_image"])
        efficiency_rows.append(
            {
                "dataset": record["dataset"],
                "model": record["model_label"],
                "parameters": int(record["parameters"]),
                "parameters_millions": int(record["parameters"]) / 1e6,
                "archived_checkpoint_mib": (
                    checkpoint.stat().st_size / 1024**2
                    if checkpoint.exists()
                    else np.nan
                ),
                "batch_size": int(latency["batch_size"]),
                "warmup_batches": int(latency["warmup"]),
                "timed_batches": int(latency["repeats"]),
                "mean_ms_per_image": milliseconds,
                "throughput_images_per_second": 1000.0 / milliseconds,
            }
        )
    pd.DataFrame(efficiency_rows).to_csv(
        analysis_dir / "released_model_efficiency.csv", index=False
    )

    make_reliability_figure(cache, output_dir)
    make_class_hierarchy_figure(per_class, hierarchy, output_dir)
    print(metrics.to_string(index=False))
    print(f"Wrote diagnostics to {analysis_dir} and {output_dir}")


if __name__ == "__main__":
    main()
