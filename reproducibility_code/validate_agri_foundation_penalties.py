"""Validate Agri-Foundation-145k penalty runs and reconstruct test metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


CONFIGURATIONS = ("dual_none", "dual_mae", "dual_kl")
SEEDS = tuple(range(2026, 2031))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = json.loads(
        (package_root / "config/taxonomy_DS2.json").read_text(encoding="utf-8")
    )
    condition_to_host = np.asarray(taxonomy["condition_to_species"], dtype=np.int64)

    records: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    errors: list[str] = []
    root = package_root / "results/revised/e2e_ablation/DS2"

    for configuration in CONFIGURATIONS:
        for seed in SEEDS:
            run_dir = root / configuration / f"seed_{seed}"
            required = {
                "completion": run_dir / "COMPLETE.json",
                "history": run_dir / "history.csv",
                "predictions": run_dir / "test_predictions.npz",
                "checkpoint": run_dir / "best_checkpoint.pth",
            }
            missing = [name for name, path in required.items() if not path.is_file()]
            if missing:
                errors.append(f"{configuration}/seed_{seed}: missing {', '.join(missing)}")
                continue

            completion = json.loads(required["completion"].read_text(encoding="utf-8"))
            history = pd.read_csv(required["history"])
            if int(completion.get("epochs", -1)) != 30 or len(history) != 30:
                errors.append(f"{configuration}/seed_{seed}: incomplete 30-epoch history")
                continue
            if not np.isclose(
                float(completion["best_selection_macro_f1"]),
                float(history["selection_macro_f1"].max()),
            ):
                errors.append(f"{configuration}/seed_{seed}: selection metric mismatch")
                continue

            archive = np.load(required["predictions"])
            labels = archive["labels"].astype(np.int64)
            predictions = archive["standard_predictions"].astype(np.int64)
            if labels.shape != (29041,) or predictions.shape != labels.shape:
                errors.append(f"{configuration}/seed_{seed}: unexpected prediction shape")
                continue

            accuracy = 100.0 * accuracy_score(labels, predictions)
            macro_f1 = 100.0 * f1_score(
                labels, predictions, average="macro", labels=np.arange(215), zero_division=0
            )
            cher = 100.0 * np.mean(
                condition_to_host[predictions] != condition_to_host[labels]
            )
            records.append(
                {
                    "dataset": "Agri-Foundation-145k",
                    "configuration": configuration,
                    "seed": seed,
                    "accuracy_pct": accuracy,
                    "macro_f1_pct": macro_f1,
                    "cross_host_error_pct": cher,
                }
            )
            artifacts.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    **{f"{name}_sha256": sha256(path) for name, path in required.items()},
                }
            )

    if errors:
        raise SystemExit("\n".join(errors))

    frame = pd.DataFrame(records)
    if len(frame) != 15:
        raise SystemExit(f"Expected 15 valid runs, found {len(frame)}")
    frame.to_csv(output_dir / "agri_foundation_penalty_metrics.csv", index=False)

    metrics = ("accuracy_pct", "macro_f1_pct", "cross_host_error_pct")
    summary_rows = []
    for configuration in CONFIGURATIONS:
        subset = frame[frame["configuration"] == configuration]
        for metric in metrics:
            values = subset[metric].to_numpy(float)
            summary_rows.append(
                {
                    "dataset": "Agri-Foundation-145k",
                    "configuration": configuration,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "agri_foundation_penalty_summary.csv", index=False
    )
    (output_dir / "agri_foundation_penalty_validation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "valid_runs": len(frame),
                "expected_runs": 15,
                "epochs_per_run": 30,
                "test_cases_per_run": 29041,
                "artifact_hashes": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("PASS: 15 Agri-Foundation-145k runs validated and metrics reconstructed")


if __name__ == "__main__":
    main()
