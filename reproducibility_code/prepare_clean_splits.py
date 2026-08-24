"""Build deterministic leakage-resistant train/validation/test manifests.

This script does not move, rename, or delete source images.  It pools the
existing packaged splits and assigns every retained image to a new 80/10/10
partition:

* PV (NPD-Rice-COCO-43): all offline transformations of one source-image
  family are assigned together.  UUIDs identify New Plant Diseases families;
  the non-UUID PlantVillage common-rust files are grouped after removing the
  documented transformation suffixes.  Exact byte matches also join groups.
* DS1 (MultiCrop-88): exact byte duplicates are collapsed to one deterministic
  canonical path before stratified partitioning.

The generated CSV is intended to be consumed directly by
``ManifestImageDataset`` in ``clean_manifest_dataset.py``.  An optional,
separate materializer is supplied for tools that require ImageFolder
directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = "clean_grouped_split_manifest_v1"
PROTOCOL_NAME = "clean_grouped_v1"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
COCO_PATTERN = re.compile(r"COCO_(?:train|val)2014_(\d{12})", re.IGNORECASE)
PV_TRANSFORM_SUFFIX = re.compile(
    r"(?:"
    r"_(?:flip(?:lr|tb|ud)|(?:90|180|270)deg)"
    r"|_new(?:\d+)?deg(?:flip(?:lr|tb|ud))?"
    r")$",
    re.IGNORECASE,
)
MANIFEST_FIELDS = (
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
EXCLUDED_FIELDS = (
    "dataset",
    "class_name",
    "sha256",
    "size_bytes",
    "source_split",
    "source_relative_path",
    "canonical_source_relative_path",
    "reason",
)


class SplitPreparationError(RuntimeError):
    """Raised when the source inventory cannot support an auditable split."""


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    source_split: str
    source_relative_path: str
    class_name: str
    size_bytes: int
    mtime_ns: int
    sha256: str = ""
    family_id: str = ""
    assignment_group: str = ""
    new_split: str = ""


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # A lexical parent makes the result independent of traversal order.
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_key(seed: int, *parts: str) -> str:
    joined = "\0".join((str(seed), *parts))
    return sha256_bytes(joined.encode("utf-8"))


def scan_source(root: Path, dataset: str) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    available_splits = [split for split in SPLITS if (root / split).is_dir()]
    if not available_splits:
        raise SplitPreparationError(
            f"No train/val/test directories found below dataset root: {root}"
        )
    for source_split in available_splits:
        split_root = root / source_split
        class_directories = sorted(
            (path for path in split_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        if not class_directories:
            raise SplitPreparationError(f"No class directories in {split_root}")
        for class_directory in class_directories:
            for path in sorted(
                (
                    candidate
                    for candidate in class_directory.rglob("*")
                    if candidate.is_file()
                    and candidate.suffix.casefold() in IMAGE_EXTENSIONS
                ),
                key=lambda candidate: candidate.as_posix(),
            ):
                stat = path.stat()
                records.append(
                    ImageRecord(
                        dataset=dataset,
                        source_split=source_split,
                        source_relative_path=path.relative_to(root).as_posix(),
                        class_name=class_directory.name,
                        size_bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                    )
                )
    if not records:
        raise SplitPreparationError(f"No supported image files found below {root}")
    return records


def load_hash_cache(path: Path) -> dict[str, tuple[int, int, str]]:
    if not path.is_file():
        return {}
    cache: dict[str, tuple[int, int, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cache[row["source_relative_path"]] = (
                int(row["size_bytes"]),
                int(row["mtime_ns"]),
                row["sha256"],
            )
    return cache


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def attach_hashes(
    records: Sequence[ImageRecord],
    root: Path,
    cache_path: Path,
    workers: int,
) -> tuple[list[ImageRecord], int]:
    cache = load_hash_cache(cache_path)
    digests: dict[str, str] = {}
    pending: list[ImageRecord] = []
    reused = 0
    for record in records:
        cached = cache.get(record.source_relative_path)
        if (
            cached is not None
            and cached[0] == record.size_bytes
            and cached[1] == record.mtime_ns
            and re.fullmatch(r"[0-9a-f]{64}", cached[2])
        ):
            digests[record.source_relative_path] = cached[2]
            reused += 1
        else:
            pending.append(record)

    if pending:
        print(
            f"  hashing {len(pending):,} files "
            f"({reused:,} unchanged hashes reused)",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    sha256_file, root / record.source_relative_path
                ): record.source_relative_path
                for record in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                relative_path = futures[future]
                digests[relative_path] = future.result()
                if completed % 5000 == 0 or completed == len(pending):
                    print(
                        f"    hashed {completed:,}/{len(pending):,}",
                        flush=True,
                    )

    hashed = [
        replace(record, sha256=digests[record.source_relative_path])
        for record in records
    ]
    write_csv(
        cache_path,
        ("source_relative_path", "size_bytes", "mtime_ns", "sha256"),
        (
            {
                "source_relative_path": record.source_relative_path,
                "size_bytes": record.size_bytes,
                "mtime_ns": record.mtime_ns,
                "sha256": record.sha256,
            }
            for record in sorted(hashed, key=lambda item: item.source_relative_path)
        ),
    )
    return hashed, reused


def pv_family_id(record: ImageRecord) -> str:
    filename = Path(record.source_relative_path).name
    uuid_match = UUID_PATTERN.search(filename)
    if uuid_match:
        return f"{record.class_name}/npd_uuid/{uuid_match.group(0).casefold()}"
    coco_match = COCO_PATTERN.search(filename)
    if coco_match:
        return f"{record.class_name}/coco2014/{coco_match.group(1)}"
    stem = Path(filename).stem
    previous = None
    while previous != stem:
        previous = stem
        stem = PV_TRANSFORM_SUFFIX.sub("", stem)
    return f"{record.class_name}/source_stem/{stem.casefold()}"


def find_cross_label_hashes(
    records: Sequence[ImageRecord],
) -> list[dict[str, object]]:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    conflicts = []
    for digest, members in sorted(by_hash.items()):
        labels = sorted({member.class_name for member in members})
        if len(labels) > 1:
            conflicts.append(
                {
                    "sha256": digest,
                    "class_names": labels,
                    "paths": sorted(member.source_relative_path for member in members),
                }
            )
    return conflicts


def deduplicate_ds1(
    records: Sequence[ImageRecord],
) -> tuple[list[ImageRecord], list[dict[str, object]], dict[str, int]]:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    retained: list[ImageRecord] = []
    excluded: list[dict[str, object]] = []
    duplicate_clusters = 0
    for digest, members in sorted(by_hash.items()):
        ordered = sorted(members, key=lambda item: item.source_relative_path)
        canonical = ordered[0]
        retained.append(
            replace(
                canonical,
                family_id=f"{canonical.class_name}/sha256/{digest}",
                assignment_group=f"{canonical.class_name}/sha256/{digest}",
            )
        )
        if len(ordered) > 1:
            duplicate_clusters += 1
        for duplicate in ordered[1:]:
            excluded.append(
                {
                    "dataset": duplicate.dataset,
                    "class_name": duplicate.class_name,
                    "sha256": duplicate.sha256,
                    "size_bytes": duplicate.size_bytes,
                    "source_split": duplicate.source_split,
                    "source_relative_path": duplicate.source_relative_path,
                    "canonical_source_relative_path": canonical.source_relative_path,
                    "reason": "exact_byte_duplicate",
                }
            )
    return (
        retained,
        excluded,
        {
            "duplicate_hash_clusters": duplicate_clusters,
            "duplicate_paths_removed": len(excluded),
        },
    )


def group_pv(records: Sequence[ImageRecord]) -> list[ImageRecord]:
    with_families = [
        replace(record, family_id=pv_family_id(record)) for record in records
    ]
    family_classes: dict[str, set[str]] = defaultdict(set)
    hash_families: dict[str, set[str]] = defaultdict(set)
    for record in with_families:
        family_classes[record.family_id].add(record.class_name)
        hash_families[record.sha256].add(record.family_id)
    impure = {
        family: sorted(labels)
        for family, labels in family_classes.items()
        if len(labels) > 1
    }
    if impure:
        raise SplitPreparationError(f"PV family spans labels: {impure}")

    disjoint = DisjointSet(family_classes)
    for families in hash_families.values():
        ordered = sorted(families)
        for family in ordered[1:]:
            disjoint.union(ordered[0], family)
    components: dict[str, list[str]] = defaultdict(list)
    for family in sorted(family_classes):
        components[disjoint.find(family)].append(family)
    component_names: dict[str, str] = {}
    for root, families in components.items():
        if len(families) == 1:
            component_names[root] = families[0]
        else:
            class_name = next(iter(family_classes[families[0]]))
            identity = sha256_bytes("\n".join(families).encode("utf-8"))[:20]
            component_names[root] = f"{class_name}/family_union/{identity}"
    return [
        replace(
            record,
            assignment_group=component_names[disjoint.find(record.family_id)],
        )
        for record in with_families
    ]


def largest_remainder(total: int, fractions: Sequence[float]) -> list[int]:
    raw = [total * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(len(fractions)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def group_count_targets(number_groups: int, fractions: Sequence[float]) -> list[int]:
    if number_groups < 1:
        raise ValueError("number_groups must be positive")
    targets = largest_remainder(number_groups, fractions)
    required: list[int] = [0] * len(fractions)
    required[0] = 1  # every class must occur in training
    if number_groups >= sum(fraction > 0 for fraction in fractions):
        required = [int(fraction > 0) for fraction in fractions]
    elif number_groups == 2 and len(fractions) >= 2:
        non_train = max(
            range(1, len(fractions)), key=lambda index: (fractions[index], -index)
        )
        required[non_train] = 1
    for recipient, minimum in enumerate(required):
        while targets[recipient] < minimum:
            donors = [
                index
                for index, count in enumerate(targets)
                if count > required[index]
            ]
            if not donors:
                raise SplitPreparationError(
                    f"Cannot allocate {number_groups} groups across {fractions}"
                )
            donor = max(
                donors,
                key=lambda index: (targets[index] - required[index], fractions[index]),
            )
            targets[donor] -= 1
            targets[recipient] += 1
    return targets


def assign_groups_for_class(
    class_name: str,
    groups: dict[str, list[ImageRecord]],
    fractions: Sequence[float],
    seed: int,
) -> dict[str, str]:
    group_items = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            stable_key(seed, class_name, item[0]),
            item[0],
        ),
    )
    quota = group_count_targets(len(group_items), fractions)
    target_images = [
        sum(len(records) for _, records in group_items) * fraction
        for fraction in fractions
    ]
    current_images = [0] * len(fractions)
    assignments: dict[str, str] = {}
    for group_id, members in group_items:
        candidates = [index for index, remaining in enumerate(quota) if remaining > 0]
        if not candidates:
            raise AssertionError("No split quota remains")

        # Groups are processed largest first.  Give the next group to the split
        # whose remaining quota currently requires the largest mean group.  This
        # preserves the exact group-count quotas without preferentially placing
        # large augmentation families in the smaller validation/test partitions.
        def candidate_score(index: int) -> tuple[float, float, int]:
            remaining_mean = (
                target_images[index] - current_images[index]
            ) / quota[index]
            deficit = target_images[index] - current_images[index]
            return (-remaining_mean, -deficit, index)

        chosen = min(candidates, key=candidate_score)
        assignments[group_id] = SPLITS[chosen]
        current_images[chosen] += len(members)
        quota[chosen] -= 1
    return assignments


def assign_stratified_groups(
    records: Sequence[ImageRecord],
    fractions: Sequence[float],
    seed: int,
) -> list[ImageRecord]:
    by_class_group: dict[str, dict[str, list[ImageRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_class_group[record.class_name][record.assignment_group].append(record)
    assignments: dict[str, str] = {}
    for class_name in sorted(by_class_group):
        class_assignments = assign_groups_for_class(
            class_name, by_class_group[class_name], fractions, seed
        )
        overlap = set(assignments).intersection(class_assignments)
        if overlap:
            raise SplitPreparationError(
                f"Assignment groups cross class labels: {sorted(overlap)[:5]}"
            )
        assignments.update(class_assignments)
    return [
        replace(record, new_split=assignments[record.assignment_group])
        for record in records
    ]


def materialized_relative_path(record: ImageRecord) -> str:
    identity = sha256_bytes(record.source_relative_path.encode("utf-8"))[:20]
    suffix = Path(record.source_relative_path).suffix.casefold()
    return f"{record.new_split}/{record.class_name}/{identity}{suffix}"


def manifest_rows(
    records: Sequence[ImageRecord], class_to_index: dict[str, int]
) -> list[dict[str, object]]:
    split_order = {name: index for index, name in enumerate(SPLITS)}
    ordered = sorted(
        records,
        key=lambda record: (
            split_order[record.new_split],
            class_to_index[record.class_name],
            record.assignment_group,
            record.source_relative_path,
        ),
    )
    rows = []
    destinations: set[str] = set()
    for record in ordered:
        destination = materialized_relative_path(record)
        if destination in destinations:
            raise SplitPreparationError(f"Materialized path collision: {destination}")
        destinations.add(destination)
        rows.append(
            {
                "dataset": record.dataset,
                "new_split": record.new_split,
                "class_name": record.class_name,
                "class_index": class_to_index[record.class_name],
                "assignment_group": record.assignment_group,
                "family_id": record.family_id,
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "source_split": record.source_split,
                "source_relative_path": record.source_relative_path,
                "materialized_relative_path": destination,
            }
        )
    return rows


def pairwise_overlap(
    records: Sequence[ImageRecord], attribute: str
) -> dict[str, int]:
    values = {
        split: {
            getattr(record, attribute)
            for record in records
            if record.new_split == split
        }
        for split in SPLITS
    }
    return {
        f"{left}_{right}": len(values[left].intersection(values[right]))
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    }


def build_audit(
    dataset: str,
    source_root: Path,
    seed: int,
    fractions: Sequence[float],
    original: Sequence[ImageRecord],
    retained: Sequence[ImageRecord],
    excluded: Sequence[dict[str, object]],
    duplicate_summary: dict[str, int],
    manifest_digest: str,
    cross_label_policy: str,
) -> dict[str, object]:
    classes = sorted({record.class_name for record in retained})
    split_records = {
        split: [record for record in retained if record.new_split == split]
        for split in SPLITS
    }
    class_counts = {
        split: dict(
            sorted(Counter(record.class_name for record in records).items())
        )
        for split, records in split_records.items()
    }
    group_overlap = pairwise_overlap(retained, "assignment_group")
    family_overlap = pairwise_overlap(retained, "family_id")
    hash_overlap = pairwise_overlap(retained, "sha256")
    train_classes = set(class_counts["train"])
    validation_classes = set(class_counts["val"])
    test_classes = set(class_counts["test"])
    all_classes = set(classes)
    assertions = {
        "all_retained_paths_assigned_once": len(retained)
        == sum(len(records) for records in split_records.values()),
        "all_classes_present_in_train": train_classes == all_classes,
        "zero_assignment_group_overlap": not any(group_overlap.values()),
        "zero_family_overlap": not any(family_overlap.values()),
        "zero_exact_hash_overlap": not any(hash_overlap.values()),
        "no_cross_label_exact_hashes": not find_cross_label_hashes(retained),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "dataset": dataset,
        "source_root": str(source_root),
        "source_splits_pooled": [
            split for split in SPLITS if any(r.source_split == split for r in original)
        ],
        "split_seed": seed,
        "requested_fractions": dict(zip(SPLITS, fractions)),
        "inventory": {
            "paths_before_deduplication": len(original),
            "paths_retained": len(retained),
            "paths_excluded": len(excluded),
            "classes": len(classes),
            "assignment_groups": len(
                {record.assignment_group for record in retained}
            ),
            "families": len({record.family_id for record in retained}),
            "unique_sha256": len({record.sha256 for record in retained}),
            **duplicate_summary,
        },
        "split_summary": {
            split: {
                "paths": len(records),
                "groups": len(
                    {record.assignment_group for record in records}
                ),
                "classes": len(class_counts[split]),
                "fraction": len(records) / len(retained),
                "fraction_deviation_percentage_points": 100
                * (len(records) / len(retained) - fractions[index]),
                "class_counts": class_counts[split],
                "source_split_counts": dict(
                    sorted(
                        Counter(record.source_split for record in records).items()
                    )
                ),
            }
            for index, (split, records) in enumerate(split_records.items())
        },
        "class_coverage": {
            "train": len(train_classes),
            "validation": len(validation_classes),
            "test": len(test_classes),
            "missing_from_validation": sorted(all_classes - validation_classes),
            "missing_from_test": sorted(all_classes - test_classes),
        },
        "overlap_checks": {
            "assignment_group": group_overlap,
            "family_id": family_overlap,
            "sha256": hash_overlap,
        },
        "assertions": assertions,
        "manifest_sha256": manifest_digest,
        "family_definition": (
            "PV: class plus New Plant Diseases UUID; class plus normalized "
            "pre-augmentation stem for non-UUID PlantVillage/Rice images; "
            "class plus COCO image id. Exact byte matches union families."
            if dataset == "PV"
            else "DS1: exact SHA-256 cluster; one canonical path retained per hash."
        ),
        "canonical_duplicate_policy": (
            "retain all PV family members"
            if dataset == "PV"
            else "lexicographically first source-relative path retained"
        ),
        "cross_label_hash_policy": cross_label_policy,
    }


def prepare_dataset(
    dataset: str,
    source_root: Path,
    output_directory: Path,
    fractions: Sequence[float],
    seed: int,
    hash_workers: int,
    cross_label_policy: str = "error",
) -> dict[str, object]:
    dataset = dataset.upper()
    if dataset not in {"PV", "DS1"}:
        raise SplitPreparationError("Only PV and DS1 clean protocols are defined")
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise SplitPreparationError(f"Dataset root does not exist: {source_root}")
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"{dataset}: scanning {source_root}", flush=True)
    original = scan_source(source_root, dataset)
    print(f"  found {len(original):,} source paths", flush=True)
    hashed, reused = attach_hashes(
        original, source_root, output_directory / "hash_cache.csv", hash_workers
    )
    print(f"  hash cache reused {reused:,}/{len(original):,}", flush=True)

    conflicts = find_cross_label_hashes(hashed)
    write_json(
        output_directory / "cross_label_hash_conflicts.json",
        {
            "dataset": dataset,
            "conflict_count": len(conflicts),
            "conflicts": conflicts,
        },
    )
    if conflicts and cross_label_policy == "error":
        raise SplitPreparationError(
            f"{dataset} contains {len(conflicts)} exact hashes assigned to "
            f"multiple classes; see {output_directory / 'cross_label_hash_conflicts.json'}"
        )
    if cross_label_policy not in {"error", "exclude"}:
        raise ValueError(f"Unknown cross-label policy: {cross_label_policy}")
    conflict_hashes = {str(conflict["sha256"]) for conflict in conflicts}
    conflict_excluded = [
        {
            "dataset": record.dataset,
            "class_name": record.class_name,
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
            "source_split": record.source_split,
            "source_relative_path": record.source_relative_path,
            "canonical_source_relative_path": "",
            "reason": "cross_label_hash_conflict",
        }
        for record in hashed
        if record.sha256 in conflict_hashes
    ]
    unambiguous = [
        record for record in hashed if record.sha256 not in conflict_hashes
    ]

    if dataset == "DS1":
        retained, duplicate_excluded, duplicate_summary = deduplicate_ds1(unambiguous)
        excluded = conflict_excluded + duplicate_excluded
    else:
        retained = group_pv(unambiguous)
        excluded = conflict_excluded
        by_hash = Counter(record.sha256 for record in unambiguous)
        duplicate_summary = {
            "duplicate_hash_clusters": sum(value > 1 for value in by_hash.values()),
            "duplicate_paths_removed": 0,
        }
    duplicate_summary.update(
        {
            "cross_label_hash_clusters_excluded": len(conflicts)
            if cross_label_policy == "exclude"
            else 0,
            "cross_label_paths_excluded": len(conflict_excluded)
            if cross_label_policy == "exclude"
            else 0,
        }
    )
    if dataset == "DS1":
        # deduplicate_ds1 assigns both grouping fields.
        pass
    assigned = assign_stratified_groups(retained, fractions, seed)
    classes = sorted({record.class_name for record in assigned})
    class_to_index = {name: index for index, name in enumerate(classes)}
    rows = manifest_rows(assigned, class_to_index)
    manifest_path = output_directory / "split_manifest.csv"
    write_csv(manifest_path, MANIFEST_FIELDS, rows)
    write_csv(
        output_directory / "excluded_duplicates.csv",
        EXCLUDED_FIELDS,
        sorted(
            excluded,
            key=lambda row: (
                str(row["class_name"]),
                str(row["sha256"]),
                str(row["source_relative_path"]),
            ),
        ),
    )
    manifest_digest = sha256_file(manifest_path)
    audit = build_audit(
        dataset,
        source_root,
        seed,
        fractions,
        hashed,
        assigned,
        excluded,
        duplicate_summary,
        manifest_digest,
        cross_label_policy,
    )
    failed = [name for name, passed in audit["assertions"].items() if not passed]
    write_json(output_directory / "audit.json", audit)
    (output_directory / "manifest.sha256").write_text(
        f"{manifest_digest}  split_manifest.csv\n", encoding="ascii"
    )
    if failed:
        raise SplitPreparationError(
            f"{dataset} split failed invariants: {', '.join(failed)}"
        )
    summary = audit["split_summary"]
    print(
        f"  retained {len(assigned):,}; "
        + ", ".join(f"{split}={summary[split]['paths']:,}" for split in SPLITS),
        flush=True,
    )
    print(f"  manifest {manifest_digest}", flush=True)
    return audit


def parse_dataset_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use DATASET=PATH, for example PV=datasets/PV")
    dataset, path = value.split("=", 1)
    dataset = dataset.upper()
    if dataset not in {"PV", "DS1"}:
        raise argparse.ArgumentTypeError("dataset must be PV or DS1")
    return dataset, Path(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic leakage-resistant split manifests."
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        type=parse_dataset_root,
        help="DATASET=PATH; may be repeated. Defaults to datasets/PV and datasets/DS1.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: results/critical_reruns/clean_splits below package root.",
    )
    parser.add_argument(
        "--fractions",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--cross-label-policy",
        choices=["error", "exclude"],
        default="error",
        help=(
            "Default error stops on identical bytes with conflicting labels. "
            "Use exclude to remove every member of each ambiguous hash cluster "
            "and record the exclusions."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fractions = tuple(args.fractions)
    if any(fraction <= 0 for fraction in fractions):
        raise SplitPreparationError("All three split fractions must be positive")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise SplitPreparationError("Split fractions must sum to 1.0")
    package_root = args.package_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else package_root / "results" / "critical_reruns" / "clean_splits"
    )
    dataset_roots = dict(args.dataset_root or [])
    if not dataset_roots:
        dataset_roots = {
            "PV": package_root / "datasets" / "PV",
            "DS1": package_root / "datasets" / "DS1",
        }
    for dataset, root in dataset_roots.items():
        prepare_dataset(
            dataset,
            root,
            output_root / dataset,
            fractions,
            args.seed,
            args.hash_workers,
            args.cross_label_policy,
        )
    write_json(
        output_root / "protocol.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "datasets": sorted(dataset_roots),
            "split_seed": args.seed,
            "fractions": dict(zip(SPLITS, fractions)),
            "dataset_manifests": {
                dataset: f"{dataset}/split_manifest.csv"
                for dataset in sorted(dataset_roots)
            },
        },
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SplitPreparationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
