"""Portable training entry point for the taxonomy-aware dual-head classifier.

Dataset directories are supplied through the command-line arguments. This script
fully specifies the optimisation used by the revised manuscript; taxonomy files
can be exported from the supplied checkpoints with ``export_taxonomy.py``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from revised_models import HierarchicalClassifier


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return training, evaluation


def aggregate_species_probability(
    disease_logits: torch.Tensor,
    condition_to_species: torch.Tensor,
    num_species: int,
) -> torch.Tensor:
    disease_probability = torch.softmax(disease_logits, dim=1)
    aggregation = disease_probability.new_zeros((disease_probability.shape[0], num_species))
    aggregation.scatter_add_(
        1,
        condition_to_species.unsqueeze(0).expand(disease_probability.shape[0], -1),
        disease_probability,
    )
    return aggregation


def taxonomic_consistency_loss(
    species_logits: torch.Tensor,
    disease_logits: torch.Tensor,
    condition_to_species: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    species_probability = torch.softmax(species_logits, dim=1)
    aggregated = aggregate_species_probability(
        disease_logits,
        condition_to_species,
        species_logits.shape[1],
    )
    if variant == "mae":
        return torch.mean(torch.abs(aggregated - species_probability))
    if variant == "kl":
        epsilon = torch.finfo(species_probability.dtype).eps
        return torch.mean(
            torch.sum(
                aggregated
                * (
                    torch.log(aggregated.clamp_min(epsilon))
                    - torch.log(species_probability.clamp_min(epsilon))
                ),
                dim=1,
            )
        )
    if variant == "none":
        return species_logits.new_zeros(())
    raise ValueError(f"Unknown consistency variant: {variant}")


def load_taxonomy(path: Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    taxonomy_classes = list(taxonomy["condition_classes"])
    if taxonomy_classes != class_names:
        raise ValueError("Taxonomy class order does not match ImageFolder class order")
    mapping = [int(value) for value in taxonomy["condition_to_species"]]
    species_names = list(taxonomy["species_names"])
    if len(mapping) != len(class_names):
        raise ValueError("Taxonomy mapping length does not match class count")
    if min(mapping) < 0 or max(mapping) >= len(species_names):
        raise ValueError("Taxonomy contains an out-of-range species index")
    return species_names, mapping


def run_epoch(
    model: HierarchicalClassifier,
    loader: DataLoader,
    device: torch.device,
    condition_to_species: torch.Tensor,
    consistency_variant: str,
    consistency_weight: float,
    species_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    species_correct = 0
    samples = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, disease_labels in loader:
            images = images.to(device, non_blocking=True)
            disease_labels = disease_labels.to(device, non_blocking=True)
            species_labels = condition_to_species[disease_labels]
            if training:
                optimizer.zero_grad(set_to_none=True)
            species_logits, disease_logits = model(images)
            disease_loss = F.cross_entropy(disease_logits, disease_labels)
            species_loss = F.cross_entropy(species_logits, species_labels)
            consistency_loss = taxonomic_consistency_loss(
                species_logits,
                disease_logits,
                condition_to_species,
                consistency_variant,
            )
            loss = disease_loss + species_weight * species_loss + consistency_weight * consistency_loss
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.shape[0]
            total_loss += float(loss.detach()) * batch_size
            labels_all.append(disease_labels.cpu().numpy())
            predictions_all.append(disease_logits.argmax(dim=1).cpu().numpy())
            species_correct += int((species_logits.argmax(dim=1) == species_labels).sum())
            samples += batch_size
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    return {
        "loss": total_loss / samples,
        "disease_accuracy": float(np.mean(labels == predictions)),
        "disease_macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "species_accuracy": species_correct / samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="swin_small_patch4_window7_224")
    parser.add_argument("--consistency", choices=["none", "mae", "kl"], default="kl")
    parser.add_argument("--lambda-tc", type=float, default=1.0)
    parser.add_argument("--species-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_transform, evaluation_transform = build_transforms()
    train_dataset = datasets.ImageFolder(args.data_dir / "train", transform=train_transform)
    validation_dataset = datasets.ImageFolder(args.data_dir / "val", transform=evaluation_transform)
    if train_dataset.classes != validation_dataset.classes:
        raise ValueError("Training and validation class orders differ")
    species_names, mapping = load_taxonomy(args.taxonomy, train_dataset.classes)
    mapping_tensor = torch.as_tensor(mapping, dtype=torch.long, device=device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = HierarchicalClassifier(
        len(species_names),
        len(train_dataset.classes),
        model_name=args.model,
        pretrained=not args.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, float | int]] = []
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            mapping_tensor,
            args.consistency,
            args.lambda_tc,
            args.species_weight,
            optimizer,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            mapping_tensor,
            args.consistency,
            args.lambda_tc,
            args.species_weight,
            None,
        )
        scheduler.step()
        row: dict[str, float | int] = {"epoch": epoch, "learning_rate": scheduler.get_last_lr()[0]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if validation_metrics["disease_macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["disease_macro_f1"]
            torch.save(
                {
                    "epoch": epoch - 1,
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "disease_classes": train_dataset.classes,
                    "species_names": species_names,
                    "disease_to_species_map": mapping,
                    "ablation_type": args.consistency,
                    "seed": args.seed,
                    "lambda_tc": args.lambda_tc,
                    "species_weight": args.species_weight,
                    "best_validation_macro_f1": best_macro_f1,
                },
                args.output_dir / "best_checkpoint.pth",
            )


if __name__ == "__main__":
    main()
