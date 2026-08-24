"""Audit head disagreement and predicted-host masking on final dual-head runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


DATASETS = {
    "PV": "NPD-Rice-COCO-43",
    "DS1": "MultiCrop-88",
    "DS2": "Agri-Foundation-145k",
}
CONFIGURATIONS = ("dual_none", "dual_mae", "dual_kl")
CONFIGURATION_LABELS = {
    "dual_none": "Dual, no penalty",
    "dual_mae": "Dual + MAE",
    "dual_kl": "Dual + KL(q || p^h)",
}


def final_run_directories(package_root: Path, dataset: str, configuration: str):
    if dataset in {"PV", "DS1"}:
        root = (
            package_root
            / "results/critical_reruns/training/clean_grouped_v1"
            / dataset
            / configuration
        )
        yield from sorted(root.glob("seed_*"))
        return

    base = package_root / "results/revised/e2e_ablation/DS2" / configuration
    yield from sorted(base.glob("seed_*"))
    if configuration == "dual_none":
        extension = (
            package_root
            / "results/revised/e2e_ablation/critical_extension_v1/DS2/dual_none"
        )
        yield from sorted(extension.glob("seed_*"))


def load_predictions(run_dir: Path, mapping: np.ndarray):
    logits_path = run_dir / "test_logits.npz"
    controlled_path = run_dir / "controlled_logits.npz"
    predictions_path = run_dir / "test_predictions.npz"

    if logits_path.exists() or controlled_path.exists():
        path = logits_path if logits_path.exists() else controlled_path
        archive = np.load(path)
        prefix = "" if "labels" in archive.files else "test_"
        labels = archive[f"{prefix}labels"].astype(int)
        condition_logits = archive[f"{prefix}condition_logits"]
        host_logits = archive[f"{prefix}host_logits"]
        standard = condition_logits.argmax(axis=1)
        predicted_hosts = host_logits.argmax(axis=1)
        masked = np.where(
            mapping[None, :] == predicted_hosts[:, None], condition_logits, -np.inf
        )
        tci = masked.argmax(axis=1)
        return labels, standard, tci, predicted_hosts

    if predictions_path.exists():
        archive = np.load(predictions_path)
        return (
            archive["labels"].astype(int),
            archive["standard_predictions"].astype(int),
            archive["tci_predictions"].astype(int),
            archive["predicted_hosts"].astype(int),
        )

    raise FileNotFoundError(f"No test output archive in {run_dir}")


def evaluate_run(labels, standard, tci, predicted_hosts, mapping):
    true_hosts = mapping[labels]
    standard_hosts = mapping[standard]
    tci_hosts = mapping[tci]
    standard_accuracy = 100.0 * np.mean(standard == labels)
    tci_accuracy = 100.0 * np.mean(tci == labels)
    standard_f1 = 100.0 * f1_score(labels, standard, average="macro", zero_division=0)
    tci_f1 = 100.0 * f1_score(labels, tci, average="macro", zero_division=0)
    standard_cher = 100.0 * np.mean(standard_hosts != true_hosts)
    tci_cher = 100.0 * np.mean(tci_hosts != true_hosts)
    return {
        "host_accuracy_pct": 100.0 * np.mean(predicted_hosts == true_hosts),
        "pre_tci_hdr_pct": 100.0 * np.mean(standard_hosts != predicted_hosts),
        "post_tci_hdr_pct": 100.0 * np.mean(tci_hosts != predicted_hosts),
        "standard_accuracy_pct": standard_accuracy,
        "tci_accuracy_pct": tci_accuracy,
        "delta_accuracy_pp": tci_accuracy - standard_accuracy,
        "standard_macro_f1_pct": standard_f1,
        "tci_macro_f1_pct": tci_f1,
        "delta_macro_f1_pp": tci_f1 - standard_f1,
        "standard_cher_pct": standard_cher,
        "tci_cher_pct": tci_cher,
        "delta_cher_pp": tci_cher - standard_cher,
        "changed_predictions": int(np.sum(standard != tci)),
    }


def summarize(seed_level: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "host_accuracy_pct",
        "pre_tci_hdr_pct",
        "post_tci_hdr_pct",
        "delta_accuracy_pp",
        "delta_macro_f1_pp",
        "delta_cher_pp",
        "changed_predictions",
    ]
    rows = []
    for dataset in DATASETS.values():
        for configuration in CONFIGURATIONS:
            group = seed_level[
                (seed_level["dataset"] == dataset)
                & (seed_level["configuration"] == configuration)
            ]
            row = {
                "dataset": dataset,
                "configuration": configuration,
                "configuration_label": group["configuration_label"].iloc[0],
                "seeds": len(group),
            }
            for metric in metrics:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_sd"] = group[metric].std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Head disagreement under standard inference and the effect of predicted-host masking (TCI). Values are mean $\pm$ standard deviation over matched final-protocol runs. Deltas are TCI minus standard inference in percentage points.}",
        r"\label{tab:tci-hdr}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Configuration & Seeds & Host accuracy & Pre-TCI HDR & $\Delta$ accuracy & $\Delta$ macro-F1 & $\Delta$ CHER \\",
        r"\midrule",
    ]
    for index, row in summary.iterrows():
        if index and summary.iloc[index - 1]["dataset"] != row["dataset"]:
            lines.append(r"\midrule")
        values = []
        for metric in (
            "host_accuracy_pct",
            "pre_tci_hdr_pct",
            "delta_accuracy_pp",
            "delta_macro_f1_pp",
            "delta_cher_pp",
        ):
            values.append(
                f"${row[f'{metric}_mean']:.3f} \\pm {row[f'{metric}_sd']:.3f}$"
            )
        lines.append(
            f"{row['dataset']} & {row['configuration_label']} & {int(row['seeds'])} & "
            + " & ".join(values)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_key, dataset_name in DATASETS.items():
        taxonomy = json.loads(
            (package_root / f"config/taxonomy_{dataset_key}.json").read_text(
                encoding="utf-8"
            )
        )
        mapping = np.asarray(taxonomy["condition_to_species"], dtype=int)
        for configuration in CONFIGURATIONS:
            for run_dir in final_run_directories(
                package_root, dataset_key, configuration
            ):
                labels, standard, tci, predicted_hosts = load_predictions(
                    run_dir, mapping
                )
                row = {
                    "dataset": dataset_name,
                    "configuration": configuration,
                    "configuration_label": CONFIGURATION_LABELS[configuration],
                    "seed": int(run_dir.name.removeprefix("seed_")),
                    "test_samples": len(labels),
                }
                row.update(
                    evaluate_run(labels, standard, tci, predicted_hosts, mapping)
                )
                rows.append(row)

    seed_level = pd.DataFrame(rows).sort_values(
        ["dataset", "configuration", "seed"]
    )
    expected = {
        (dataset, configuration): 10 if configuration == "dual_none" else 5
        for dataset in DATASETS.values()
        for configuration in CONFIGURATIONS
    }
    observed = seed_level.groupby(["dataset", "configuration"]).size().to_dict()
    for key, count in expected.items():
        if observed.get(key) != count:
            raise RuntimeError(f"Expected {count} runs for {key}, found {observed.get(key)}")
    if not np.allclose(seed_level["post_tci_hdr_pct"], 0.0):
        raise RuntimeError("TCI did not eliminate every head disagreement")

    summary = summarize(seed_level)
    seed_level.to_csv(output_dir / "tci_hdr_seed_level.csv", index=False)
    summary.to_csv(output_dir / "tci_hdr_summary.csv", index=False)
    write_latex(summary, output_dir / "table_tci_hdr.tex")
    print(summary.to_string(index=False))
    print(f"PASS: analyzed {len(seed_level)} final dual-head runs")


if __name__ == "__main__":
    main()
