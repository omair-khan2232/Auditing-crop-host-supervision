"""Independently validate generated clean-split manifests and audit records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(directory: Path, verify_paths: bool) -> dict[str, object]:
    manifest = directory / "split_manifest.csv"
    audit_path = directory / "audit.json"
    checksum_path = directory / "manifest.sha256"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest}")
    dataset_names = {row["dataset"] for row in rows}
    if len(dataset_names) != 1:
        raise RuntimeError(f"Multiple datasets in {manifest}: {dataset_names}")
    dataset = next(iter(dataset_names))
    digest = sha256_file(manifest)
    checksum_digest = checksum_path.read_text(encoding="ascii").split()[0]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if digest != checksum_digest or digest != audit["manifest_sha256"]:
        raise RuntimeError(f"Manifest checksum disagreement for {dataset}")

    source_paths = [row["source_relative_path"] for row in rows]
    if len(source_paths) != len(set(source_paths)):
        raise RuntimeError(f"Source path assigned more than once for {dataset}")
    materialized = [row["materialized_relative_path"] for row in rows]
    if len(materialized) != len(set(materialized)):
        raise RuntimeError(f"Materialized path collision for {dataset}")

    classes = sorted({row["class_name"] for row in rows})
    expected_indices = {name: index for index, name in enumerate(classes)}
    for row in rows:
        if row["new_split"] not in SPLITS:
            raise RuntimeError(f"Invalid split in {dataset}: {row['new_split']}")
        if int(row["class_index"]) != expected_indices[row["class_name"]]:
            raise RuntimeError(f"Class-index mismatch in {dataset}")
        expected_prefix = f"{row['new_split']}/{row['class_name']}/"
        if not row["materialized_relative_path"].startswith(expected_prefix):
            raise RuntimeError(f"Materialized path/split mismatch in {dataset}")

    split_sets: dict[str, dict[str, set[str]]] = {
        field: {
            split: {
                row[field] for row in rows if row["new_split"] == split
            }
            for split in SPLITS
        }
        for field in ("assignment_group", "family_id", "sha256")
    }
    overlap: dict[str, dict[str, int]] = {}
    for field, values in split_sets.items():
        overlap[field] = {
            f"{left}_{right}": len(values[left] & values[right])
            for left_index, left in enumerate(SPLITS)
            for right in SPLITS[left_index + 1 :]
        }
        if any(overlap[field].values()):
            raise RuntimeError(f"{dataset} has cross-split {field} overlap")

    class_coverage = {
        split: len({row["class_name"] for row in rows if row["new_split"] == split})
        for split in SPLITS
    }
    if class_coverage["train"] != len(classes):
        raise RuntimeError(f"{dataset} does not contain every class in train")
    if audit["class_coverage"]["validation"] != class_coverage["val"]:
        raise RuntimeError(f"{dataset} validation coverage disagrees with audit")
    if audit["class_coverage"]["test"] != class_coverage["test"]:
        raise RuntimeError(f"{dataset} test coverage disagrees with audit")

    split_counts = Counter(row["new_split"] for row in rows)
    for split in SPLITS:
        if split_counts[split] != audit["split_summary"][split]["paths"]:
            raise RuntimeError(f"{dataset}/{split} count disagrees with audit")
    if not all(audit["assertions"].values()):
        raise RuntimeError(f"{dataset} audit contains a failed assertion")

    if verify_paths:
        source_root = Path(audit["source_root"])
        missing = [
            row["source_relative_path"]
            for row in rows
            if not (source_root / row["source_relative_path"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{dataset}: {len(missing)} source paths are missing"
            )

    return {
        "dataset": dataset,
        "manifest_sha256": digest,
        "paths": len(rows),
        "classes": len(classes),
        "split_paths": dict(split_counts),
        "class_coverage": class_coverage,
        "overlap": overlap,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "critical_reruns"
        / "clean_splits",
    )
    parser.add_argument("--datasets", nargs="+", default=["PV", "DS1"])
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reports = [
        validate_dataset(args.split_root / dataset, args.verify_paths)
        for dataset in args.datasets
    ]
    report = {"status": "PASS", "datasets": reports}
    output = args.output or args.split_root / "validation.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for dataset in reports:
        print(
            f"{dataset['dataset']}: PASS, {dataset['paths']:,} paths, "
            f"manifest={dataset['manifest_sha256']}",
            flush=True,
        )
    print(output)


if __name__ == "__main__":
    main()
