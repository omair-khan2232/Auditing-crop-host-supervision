"""Freeze the original runtime development split as an explicit CSV manifest.

The first 60 controlled runs generated their 80/20 development partition inside
``train_e2e_ablation.py``.  Additional DS2 seeds must use exactly that same
partition to form a valid ten-seed paired comparison.  This utility serializes
the original deterministic indices without changing any image or assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from torchvision.datasets import ImageFolder

from linear_probe_ablation import stratified_development_indices


FIELDS = (
    "dataset",
    "new_split",
    "class_name",
    "class_index",
    "assignment_group",
    "family_id",
    "sha256",
    "size_bytes",
    "source_split",
    "source_relative_path",
    "materialized_relative_path",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--dataset", default="DS2", choices=["PV", "DS1", "DS2"])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    dataset_root = package_root / "datasets" / args.dataset
    development_root = dataset_root / "val"
    test_root = dataset_root / "test"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else package_root
        / "results"
        / "critical_reruns"
        / "legacy_controlled_splits"
        / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    development = ImageFolder(development_root)
    test = ImageFolder(test_root)
    if development.classes != test.classes:
        raise ValueError("Development and test class order differs")
    labels = np.asarray(development.targets, dtype=np.int64)
    training_indices, selection_indices = stratified_development_indices(
        labels, args.selection_fraction, args.split_seed
    )
    assignment = np.full(len(development), "", dtype=object)
    assignment[training_indices] = "train"
    assignment[selection_indices] = "val"
    if np.any(assignment == ""):
        raise AssertionError("Development assignment is incomplete")

    rows: list[dict[str, str | int]] = []
    for index, (path_string, class_index) in enumerate(development.samples):
        path = Path(path_string)
        relative = path.relative_to(dataset_root).as_posix()
        rows.append(
            {
                "dataset": args.dataset,
                "new_split": str(assignment[index]),
                "class_name": development.classes[class_index],
                "class_index": class_index,
                "assignment_group": f"legacy_path/{relative}",
                "family_id": f"legacy_path/{relative}",
                "sha256": "",
                "size_bytes": path.stat().st_size,
                "source_split": "val",
                "source_relative_path": relative,
                "materialized_relative_path": "",
            }
        )
    for path_string, class_index in test.samples:
        path = Path(path_string)
        relative = path.relative_to(dataset_root).as_posix()
        rows.append(
            {
                "dataset": args.dataset,
                "new_split": "test",
                "class_name": test.classes[class_index],
                "class_index": class_index,
                "assignment_group": f"legacy_path/{relative}",
                "family_id": f"legacy_path/{relative}",
                "sha256": "",
                "size_bytes": path.stat().st_size,
                "source_split": "test",
                "source_relative_path": relative,
                "materialized_relative_path": "",
            }
        )
    split_order = {"train": 0, "val": 1, "test": 2}
    rows.sort(
        key=lambda row: (
            split_order[str(row["new_split"])],
            int(row["class_index"]),
            str(row["source_relative_path"]),
        )
    )
    manifest_path = output_dir / "split_manifest.csv"
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest_path)

    split_counts = Counter(str(row["new_split"]) for row in rows)
    class_coverage = {
        split: len(
            {
                str(row["class_name"])
                for row in rows
                if row["new_split"] == split
            }
        )
        for split in ("train", "val", "test")
    }
    audit = {
        "protocol": "legacy_development_pool_80_20_frozen_v1",
        "purpose": (
            "Exact explicit representation of train_e2e_ablation.py's original "
            "runtime split; intended only for compatible additional seeds."
        ),
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "split_seed": args.split_seed,
        "selection_fraction": args.selection_fraction,
        "split_counts": dict(split_counts),
        "class_coverage": class_coverage,
        "classes": development.classes,
        "manifest_sha256": sha256(manifest_path),
        "assertions": {
            "complete_development_assignment": int(
                split_counts["train"] + split_counts["val"]
            )
            == len(development),
            "untouched_test_count": int(split_counts["test"]) == len(test),
            "all_classes_in_train": class_coverage["train"]
            == len(development.classes),
            # The original splitter deliberately keeps one-observation
            # development classes in training only.  Compatibility requires
            # preserving that behavior rather than manufacturing validation
            # examples.
            "validation_classes_are_known": class_coverage["val"]
            <= len(development.classes),
            "test_classes_are_known": class_coverage["test"]
            <= len(development.classes),
        },
    }
    failed = [key for key, value in audit["assertions"].items() if not value]
    if failed:
        raise RuntimeError(f"Manifest assertions failed: {failed}")
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.sha256").write_text(
        f"{audit['manifest_sha256']}  split_manifest.csv\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
