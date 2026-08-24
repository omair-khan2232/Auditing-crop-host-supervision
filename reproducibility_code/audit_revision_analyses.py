"""Reviewer-requested audits computed from fixed predictions and logits.

No model is trained or selected here. The script adds equivalence-margin
sensitivity, NPD family-balanced metrics, and Agri-Foundation-145k class,
host, disagreement, and oracle-host diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import f1_score


SEEDS = tuple(range(2026, 2036))
MARGINS = (0.10, 0.25, 0.50)
METRICS = ("accuracy_pct", "macro_f1_pct", "cross_host_error_pct")
TEST_SIZES = {
    "NPD-Rice-COCO-43": 9333,
    "MultiCrop-88": 7769,
    "Agri-Foundation-145k": 29041,
}
CONFIG_LABELS = {"flat": "Flat", "dual_none": "Dual, no penalty"}
FREQUENCY_BINS = (
    ("1--25", 1, 25),
    ("26--100", 26, 100),
    ("101--500", 101, 500),
    (">500", 501, np.inf),
)


def mean_ci(values: np.ndarray, confidence: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def tost_p_value(values: np.ndarray, margin: float) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    if standard_error == 0:
        lower_p = 0.0 if mean > -margin else 1.0
        upper_p = 0.0 if mean < margin else 1.0
    else:
        degrees = len(values) - 1
        lower_p = float(student_t.sf((mean + margin) / standard_error, degrees))
        upper_p = float(student_t.cdf((mean - margin) / standard_error, degrees))
    return lower_p, upper_p, max(lower_p, upper_p)


def equivalence_sensitivity(evidence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    differences = pd.read_csv(evidence_dir / "core_paired_seed_differences.csv")
    rows: list[dict[str, object]] = []
    for margin in MARGINS:
        margin_rows: list[dict[str, object]] = []
        grouped = differences.groupby(["dataset", "metric"], sort=False)
        for (dataset, metric), group in grouped:
            values = group["paired_difference_pp"].to_numpy(float)
            lower90, upper90 = mean_ci(values, 0.90)
            lower_p, upper_p, p_value = tost_p_value(values, margin)
            margin_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "margin_pp": margin,
                    "n_pairs": len(values),
                    "mean_difference_pp": values.mean(),
                    "tost_90ci_lower_pp": lower90,
                    "tost_90ci_upper_pp": upper90,
                    "tost_lower_p": lower_p,
                    "tost_upper_p": upper_p,
                    "tost_p": p_value,
                }
            )
        adjusted = holm_adjust(np.asarray([row["tost_p"] for row in margin_rows]))
        for row, adjusted_p in zip(margin_rows, adjusted):
            row["holm_adjusted_tost_p"] = adjusted_p
            row["equivalent_holm_0_05"] = bool(adjusted_p < 0.05)
            rows.append(row)
    sensitivity = pd.DataFrame(rows)

    event_rows = []
    selected = differences[differences["metric"] == "cross_host_error_pct"]
    for dataset, group in selected.groupby("dataset", sort=False):
        test_size = TEST_SIZES[dataset]
        flat_events = group["reference_value_pct"].to_numpy(float) * test_size / 100.0
        dual_events = group["candidate_value_pct"].to_numpy(float) * test_size / 100.0
        event_rows.append(
            {
                "dataset": dataset,
                "test_images": test_size,
                "flat_mean_cross_host_events": flat_events.mean(),
                "dual_mean_cross_host_events": dual_events.mean(),
                "dual_minus_flat_mean_events": (dual_events - flat_events).mean(),
                "dual_to_flat_rate_ratio": dual_events.mean() / flat_events.mean(),
                "events_at_0_10pp_margin": test_size * 0.001,
                "events_at_0_25pp_margin": test_size * 0.0025,
                "events_at_0_50pp_margin": test_size * 0.005,
            }
        )
    return sensitivity, pd.DataFrame(event_rows)


def load_taxonomy(package_root: Path, key: str) -> tuple[np.ndarray, list[str], list[str]]:
    taxonomy = json.loads(
        (package_root / f"config/taxonomy_{key}.json").read_text(encoding="utf-8")
    )
    return (
        np.asarray(taxonomy["condition_to_species"], dtype=int),
        list(taxonomy["condition_classes"]),
        list(taxonomy["species_names"]),
    )


def npd_run_directory(package_root: Path, configuration: str, seed: int) -> Path:
    return (
        package_root
        / "results/critical_reruns/training/clean_grouped_v1/PV"
        / configuration
        / f"seed_{seed}"
    )


def npd_family_balanced(package_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(
        package_root / "results/critical_reruns/clean_splits/PV/split_manifest.csv",
        usecols=["new_split", "source_relative_path", "family_id"],
    )
    test_manifest = manifest.loc[manifest["new_split"] == "test"].copy()
    family_sizes = test_manifest.groupby("family_id").size()
    test_manifest["family_weight"] = test_manifest["family_id"].map(
        lambda value: 1.0 / family_sizes[value]
    )
    path_to_weight = test_manifest.set_index("source_relative_path")["family_weight"]
    mapping, classes, _ = load_taxonomy(package_root, "PV")

    rows = []
    for configuration in ("flat", "dual_none"):
        for seed in SEEDS:
            archive = np.load(
                npd_run_directory(package_root, configuration, seed) / "test_logits.npz"
            )
            labels = archive["labels"].astype(int)
            predictions = archive["condition_logits"].argmax(axis=1)
            paths = archive["relative_paths"].astype(str)
            if len(paths) != len(test_manifest) or not set(paths).issubset(path_to_weight.index):
                raise RuntimeError("NPD test paths do not match the family manifest")
            weights = path_to_weight.loc[paths].to_numpy(float)
            rows.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    "test_images": len(labels),
                    "test_families": len(family_sizes),
                    "family_balanced_accuracy_pct": 100.0
                    * np.average(predictions == labels, weights=weights),
                    "family_balanced_macro_f1_pct": 100.0
                    * f1_score(
                        labels,
                        predictions,
                        labels=np.arange(len(classes)),
                        average="macro",
                        sample_weight=weights,
                        zero_division=0,
                    ),
                    "family_balanced_cher_pct": 100.0
                    * np.average(mapping[predictions] != mapping[labels], weights=weights),
                }
            )
    seed_level = pd.DataFrame(rows)

    summary_rows = []
    for metric in (
        "family_balanced_accuracy_pct",
        "family_balanced_macro_f1_pct",
        "family_balanced_cher_pct",
    ):
        pivot = seed_level.pivot(index="seed", columns="configuration", values=metric)
        differences = (pivot["dual_none"] - pivot["flat"]).to_numpy(float)
        lower, upper = mean_ci(differences, 0.95)
        for configuration in ("flat", "dual_none"):
            values = pivot[configuration].to_numpy(float)
            summary_rows.append(
                {
                    "metric": metric,
                    "configuration": configuration,
                    "seeds": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "dual_minus_flat_mean_pp": differences.mean(),
                    "dual_minus_flat_95ci_lower_pp": lower,
                    "dual_minus_flat_95ci_upper_pp": upper,
                }
            )
    return seed_level, pd.DataFrame(summary_rows)


def agri_run_directory(package_root: Path, configuration: str, seed: int) -> Path:
    if seed <= 2030:
        return package_root / f"results/revised/e2e_ablation/DS2/{configuration}/seed_{seed}"
    return (
        package_root
        / "results/revised/e2e_ablation/critical_extension_v1/DS2"
        / configuration
        / f"seed_{seed}"
    )


def load_agri_logits(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    combined = run_dir / "controlled_logits.npz"
    split = run_dir / "test_logits.npz"
    archive = np.load(combined if combined.exists() else split)
    if "test_labels" in archive.files:
        labels = archive["test_labels"].astype(int)
        condition_logits = archive["test_condition_logits"].astype(float)
        host_logits = archive["test_host_logits"].astype(float)
    else:
        labels = archive["labels"].astype(int)
        condition_logits = archive["condition_logits"].astype(float)
        host_logits = archive["host_logits"].astype(float) if "host_logits" in archive.files else None
    return labels, condition_logits, host_logits


def oracle_predictions(
    condition_logits: np.ndarray,
    true_hosts: np.ndarray,
    mapping: np.ndarray,
) -> np.ndarray:
    masked = np.where(mapping[None, :] == true_hosts[:, None], condition_logits, -np.inf)
    return masked.argmax(axis=1)


def agri_diagnostics(
    package_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping, class_names, host_names = load_taxonomy(package_root, "DS2")
    manifest = pd.read_csv(
        package_root / "results/critical_reruns/legacy_controlled_splits/DS2/split_manifest.csv",
        usecols=["new_split", "class_index"],
    )
    train_counts = (
        manifest.loc[manifest["new_split"] == "train"]
        .groupby("class_index")
        .size()
        .reindex(np.arange(len(class_names)), fill_value=0)
    )
    test_counts = (
        manifest.loc[manifest["new_split"] == "test"]
        .groupby("class_index")
        .size()
        .reindex(np.arange(len(class_names)), fill_value=0)
    )
    bin_by_class: dict[int, str] = {}
    for class_index, count in train_counts.items():
        for label, lower, upper in FREQUENCY_BINS:
            if lower <= count <= upper:
                bin_by_class[int(class_index)] = label
                break

    class_rows = []
    host_rows = []
    oracle_rows = []
    disagreement_rows = []
    bin_rows = []
    for configuration in ("flat", "dual_none"):
        for seed in SEEDS:
            labels, condition_logits, host_logits = load_agri_logits(
                agri_run_directory(package_root, configuration, seed)
            )
            predictions = condition_logits.argmax(axis=1)
            true_hosts = mapping[labels]
            predicted_condition_hosts = mapping[predictions]
            oracle = oracle_predictions(condition_logits, true_hosts, mapping)
            class_f1 = 100.0 * f1_score(
                labels,
                predictions,
                labels=np.arange(len(class_names)),
                average=None,
                zero_division=0,
            )
            for class_index, value in enumerate(class_f1):
                class_rows.append(
                    {
                        "configuration": configuration,
                        "seed": seed,
                        "class_index": class_index,
                        "class_name": class_names[class_index],
                        "host_name": host_names[mapping[class_index]],
                        "training_images": int(train_counts[class_index]),
                        "test_images": int(test_counts[class_index]),
                        "frequency_bin": bin_by_class[class_index],
                        "class_f1_pct": value,
                    }
                )
            for label, _, _ in FREQUENCY_BINS:
                indices = np.asarray(
                    [index for index, bin_label in bin_by_class.items() if bin_label == label]
                )
                bin_rows.append(
                    {
                        "configuration": configuration,
                        "seed": seed,
                        "frequency_bin": label,
                        "classes": len(indices),
                        "training_images": int(train_counts.loc[indices].sum()),
                        "macro_f1_pct": float(class_f1[indices].mean()),
                    }
                )
            for host_index, host_name in enumerate(host_names):
                selected = true_hosts == host_index
                host_rows.append(
                    {
                        "configuration": configuration,
                        "seed": seed,
                        "host_index": host_index,
                        "host_name": host_name,
                        "test_images": int(selected.sum()),
                        "condition_accuracy_pct": 100.0
                        * np.mean(predictions[selected] == labels[selected]),
                        "cher_pct": 100.0
                        * np.mean(predicted_condition_hosts[selected] != true_hosts[selected]),
                    }
                )
            oracle_rows.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    "standard_accuracy_pct": 100.0 * np.mean(predictions == labels),
                    "standard_macro_f1_pct": 100.0
                    * f1_score(
                        labels,
                        predictions,
                        labels=np.arange(len(class_names)),
                        average="macro",
                        zero_division=0,
                    ),
                    "standard_cher_pct": 100.0
                    * np.mean(predicted_condition_hosts != true_hosts),
                    "oracle_accuracy_pct": 100.0 * np.mean(oracle == labels),
                    "oracle_macro_f1_pct": 100.0
                    * f1_score(
                        labels,
                        oracle,
                        labels=np.arange(len(class_names)),
                        average="macro",
                        zero_division=0,
                    ),
                    "oracle_accuracy_gain_pp": 100.0
                    * (np.mean(oracle == labels) - np.mean(predictions == labels)),
                    "oracle_macro_f1_gain_pp": 100.0
                    * (
                        f1_score(labels, oracle, labels=np.arange(len(class_names)), average="macro", zero_division=0)
                        - f1_score(labels, predictions, labels=np.arange(len(class_names)), average="macro", zero_division=0)
                    ),
                }
            )
            if configuration == "dual_none":
                if host_logits is None:
                    raise RuntimeError("Dual-head run has no host logits")
                predicted_hosts = host_logits.argmax(axis=1)
                tci = np.where(
                    mapping[None, :] == predicted_hosts[:, None],
                    condition_logits,
                    -np.inf,
                ).argmax(axis=1)
                disagreement = predicted_condition_hosts != predicted_hosts
                categories = {
                    "host_correct_condition_host_wrong": disagreement
                    & (predicted_hosts == true_hosts),
                    "condition_host_correct_host_wrong": disagreement
                    & (predicted_condition_hosts == true_hosts),
                    "both_hosts_wrong": disagreement
                    & (predicted_hosts != true_hosts)
                    & (predicted_condition_hosts != true_hosts),
                }
                total_disagreements = int(disagreement.sum())
                for category, selected in categories.items():
                    disagreement_rows.append(
                        {
                            "seed": seed,
                            "category": category,
                            "count": int(selected.sum()),
                            "pct_all_test_images": 100.0 * np.mean(selected),
                            "pct_of_disagreements": 100.0
                            * selected.sum()
                            / total_disagreements,
                        }
                    )
                for category, selected in {
                    "top1_gained_by_tci": (predictions != labels) & (tci == labels),
                    "top1_lost_by_tci": (predictions == labels) & (tci != labels),
                    "prediction_changed": predictions != tci,
                }.items():
                    disagreement_rows.append(
                        {
                            "seed": seed,
                            "category": category,
                            "count": int(selected.sum()),
                            "pct_all_test_images": 100.0 * np.mean(selected),
                            "pct_of_disagreements": 100.0
                            * selected.sum()
                            / total_disagreements,
                        }
                    )

    class_seed = pd.DataFrame(class_rows)
    class_summary = (
        class_seed.groupby(
            [
                "configuration",
                "class_index",
                "class_name",
                "host_name",
                "training_images",
                "test_images",
                "frequency_bin",
            ],
            as_index=False,
        )["class_f1_pct"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "class_f1_pct_mean", "std": "class_f1_pct_sd"})
    )
    class_wide = class_summary.pivot(
        index=["class_index", "class_name", "host_name", "training_images", "test_images", "frequency_bin"],
        columns="configuration",
        values="class_f1_pct_mean",
    ).reset_index()
    class_wide["dual_minus_flat_f1_pp"] = class_wide["dual_none"] - class_wide["flat"]
    class_wide = class_wide.rename(
        columns={"flat": "flat_f1_pct_mean", "dual_none": "dual_f1_pct_mean"}
    ).sort_values("dual_minus_flat_f1_pp", ascending=False)

    bin_seed = pd.DataFrame(bin_rows)
    bin_summary_rows = []
    for label, _, _ in FREQUENCY_BINS:
        selected = bin_seed[bin_seed["frequency_bin"] == label]
        pivot = selected.pivot(index="seed", columns="configuration", values="macro_f1_pct")
        differences = (pivot["dual_none"] - pivot["flat"]).to_numpy(float)
        lower, upper = mean_ci(differences, 0.95)
        bin_summary_rows.append(
            {
                "frequency_bin": label,
                "classes": int(selected["classes"].iloc[0]),
                "training_images": int(selected["training_images"].iloc[0]),
                "flat_macro_f1_pct_mean": pivot["flat"].mean(),
                "flat_macro_f1_pct_sd": pivot["flat"].std(ddof=1),
                "dual_macro_f1_pct_mean": pivot["dual_none"].mean(),
                "dual_macro_f1_pct_sd": pivot["dual_none"].std(ddof=1),
                "dual_minus_flat_f1_pp": differences.mean(),
                "paired_95ci_lower_pp": lower,
                "paired_95ci_upper_pp": upper,
            }
        )
    bin_summary = pd.DataFrame(bin_summary_rows)

    host_seed = pd.DataFrame(host_rows)
    host_summary = (
        host_seed.groupby(["configuration", "host_index", "host_name"], as_index=False)
        .agg(
            test_images=("test_images", "first"),
            condition_accuracy_pct_mean=("condition_accuracy_pct", "mean"),
            condition_accuracy_pct_sd=("condition_accuracy_pct", "std"),
            cher_pct_mean=("cher_pct", "mean"),
            cher_pct_sd=("cher_pct", "std"),
        )
    )
    host_wide = host_summary.pivot(
        index=["host_index", "host_name", "test_images"],
        columns="configuration",
        values="cher_pct_mean",
    ).reset_index()
    host_wide["dual_minus_flat_cher_pp"] = host_wide["dual_none"] - host_wide["flat"]
    host_wide = host_wide.rename(
        columns={"flat": "flat_cher_pct_mean", "dual_none": "dual_cher_pct_mean"}
    ).sort_values("flat_cher_pct_mean", ascending=False)

    oracle_seed = pd.DataFrame(oracle_rows)
    oracle_summary = (
        oracle_seed.groupby("configuration", as_index=False)
        .agg(
            seeds=("seed", "size"),
            standard_accuracy_pct_mean=("standard_accuracy_pct", "mean"),
            standard_accuracy_pct_sd=("standard_accuracy_pct", "std"),
            standard_macro_f1_pct_mean=("standard_macro_f1_pct", "mean"),
            standard_macro_f1_pct_sd=("standard_macro_f1_pct", "std"),
            standard_cher_pct_mean=("standard_cher_pct", "mean"),
            standard_cher_pct_sd=("standard_cher_pct", "std"),
            oracle_accuracy_pct_mean=("oracle_accuracy_pct", "mean"),
            oracle_accuracy_pct_sd=("oracle_accuracy_pct", "std"),
            oracle_macro_f1_pct_mean=("oracle_macro_f1_pct", "mean"),
            oracle_macro_f1_pct_sd=("oracle_macro_f1_pct", "std"),
            oracle_accuracy_gain_pp_mean=("oracle_accuracy_gain_pp", "mean"),
            oracle_accuracy_gain_pp_sd=("oracle_accuracy_gain_pp", "std"),
            oracle_macro_f1_gain_pp_mean=("oracle_macro_f1_gain_pp", "mean"),
            oracle_macro_f1_gain_pp_sd=("oracle_macro_f1_gain_pp", "std"),
        )
    )
    oracle_summary["configuration_order"] = oracle_summary["configuration"].map(
        {"flat": 0, "dual_none": 1}
    )
    oracle_summary = oracle_summary.sort_values("configuration_order").drop(
        columns="configuration_order"
    )
    disagreement_seed = pd.DataFrame(disagreement_rows)
    disagreement_summary = (
        disagreement_seed.groupby("category", as_index=False)
        .agg(
            seeds=("seed", "size"),
            count_mean=("count", "mean"),
            count_sd=("count", "std"),
            pct_all_test_images_mean=("pct_all_test_images", "mean"),
            pct_of_disagreements_mean=("pct_of_disagreements", "mean"),
        )
        .sort_values("category")
    )
    return (
        class_wide,
        bin_summary,
        host_wide,
        oracle_seed,
        oracle_summary,
        disagreement_summary,
    )


def make_agri_figure(
    bin_summary: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    x = np.arange(len(bin_summary))
    width = 0.34
    axes[0].bar(
        x - width / 2,
        bin_summary["flat_macro_f1_pct_mean"],
        width,
        color="#315B7D",
        label="Flat",
        yerr=student_t.ppf(0.975, 9)
        * bin_summary["flat_macro_f1_pct_sd"]
        / np.sqrt(10),
        capsize=3,
    )
    axes[0].bar(
        x + width / 2,
        bin_summary["dual_macro_f1_pct_mean"],
        width,
        color="#2E7D5B",
        label="Dual",
        yerr=student_t.ppf(0.975, 9)
        * bin_summary["dual_macro_f1_pct_sd"]
        / np.sqrt(10),
        capsize=3,
    )
    axes[0].set_xticks(x, bin_summary["frequency_bin"].str.replace("--", "-"))
    axes[0].set_xlabel("Training images per class")
    axes[0].set_ylabel("Macro-F1 within bin (%)")
    axes[0].set_title("Long-tail performance", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)

    labels = [CONFIG_LABELS[value] for value in oracle_summary["configuration"]]
    standard = oracle_summary["standard_accuracy_pct_mean"].to_numpy(float)
    oracle = oracle_summary["oracle_accuracy_pct_mean"].to_numpy(float)
    standard_error = (
        student_t.ppf(0.975, 9)
        * oracle_summary["standard_accuracy_pct_sd"].to_numpy(float)
        / np.sqrt(10)
    )
    oracle_error = (
        student_t.ppf(0.975, 9)
        * oracle_summary["oracle_accuracy_pct_sd"].to_numpy(float)
        / np.sqrt(10)
    )
    axes[1].bar(
        x[:2] - width / 2,
        standard,
        width,
        color="#315B7D",
        label="Standard",
        yerr=standard_error,
        capsize=3,
    )
    axes[1].bar(
        x[:2] + width / 2,
        oracle,
        width,
        color="#C66A1B",
        label="Ground-truth-host oracle",
        yerr=oracle_error,
        capsize=3,
    )
    axes[1].set_xticks(x[:2], labels)
    axes[1].set_ylabel("Condition accuracy (%)")
    axes[1].set_ylim(min(standard.min(), oracle.min()) - 1.0, 100.0)
    axes[1].set_title("Ground-truth-host oracle", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8.5)
    for suffix, options in (("pdf", {}), ("png", {"dpi": 350})):
        fig.savefig(
            figure_dir / f"agri_long_tail_oracle.{suffix}",
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.15,
            **options,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    evidence_dir = args.evidence_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    sensitivity, event_counts = equivalence_sensitivity(evidence_dir)
    family_seed, family_summary = npd_family_balanced(package_root)
    (
        class_summary,
        bin_summary,
        host_summary,
        oracle_seed,
        oracle_summary,
        disagreement_summary,
    ) = agri_diagnostics(package_root)

    outputs = {
        "equivalence_margin_sensitivity.csv": sensitivity,
        "cher_event_count_context.csv": event_counts,
        "npd_family_balanced_seed_metrics.csv": family_seed,
        "npd_family_balanced_summary.csv": family_summary,
        "agri_class_level_effects.csv": class_summary,
        "agri_frequency_bin_summary.csv": bin_summary,
        "agri_host_level_cher.csv": host_summary,
        "agri_oracle_seed_metrics.csv": oracle_seed,
        "agri_oracle_summary.csv": oracle_summary,
        "agri_disagreement_categories.csv": disagreement_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(evidence_dir / filename, index=False)
    make_agri_figure(bin_summary, oracle_summary, figure_dir)
    print("PASS: reviewer-requested fixed-output analyses completed")
    print(bin_summary.to_string(index=False))
    print(oracle_summary.to_string(index=False))
    print(disagreement_summary.to_string(index=False))


if __name__ == "__main__":
    main()
