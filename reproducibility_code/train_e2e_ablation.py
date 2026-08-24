"""Resumable end-to-end Swin ablation using a packaged development pool.

To apply one protocol across all three datasets, this script deliberately uses
the packaged ``val`` directory—even where a separate ``train`` directory is
available—and creates a fixed 80/20 train/selection development partition. The
packaged ``test`` directory is never opened during training or model selection.
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from linear_probe_ablation import stratified_development_indices
from revised_models import HierarchicalClassifier
from train_hierarchical_revised import (
    build_transforms,
    load_taxonomy,
    taxonomic_consistency_loss,
)


CONFIGS = {
    "flat": {"species_weight": 0.0, "consistency": "none"},
    "dual_none": {"species_weight": 1.0, "consistency": "none"},
    "dual_mae": {"species_weight": 1.0, "consistency": "mae"},
    "dual_kl": {"species_weight": 1.0, "consistency": "kl"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def epoch_pass(model, loader, device, mapping, config, lambda_tc, optimizer, scaler):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    labels_all, predictions_all = [], []
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                host_logits, condition_logits = model(images)
                loss = F.cross_entropy(condition_logits, labels)
                if config["species_weight"]:
                    host_labels = mapping[labels]
                    loss = loss + F.cross_entropy(host_logits, host_labels)
                    loss = loss + lambda_tc * taxonomic_consistency_loss(
                        host_logits, condition_logits, mapping, config["consistency"]
                    )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            sample_count += len(labels)
            labels_all.append(labels.cpu().numpy())
            predictions_all.append(condition_logits.argmax(1).cpu().numpy())
    labels_np = np.concatenate(labels_all)
    predictions_np = np.concatenate(predictions_all)
    return {
        "loss": loss_sum / sample_count,
        "accuracy": float(np.mean(labels_np == predictions_np)),
        "macro_f1": float(f1_score(labels_np, predictions_np, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", required=True, choices=["PV", "DS1", "DS2"])
    parser.add_argument("--configuration", required=True, choices=list(CONFIGS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-tc", type=float, default=1.0)
    parser.add_argument("--model", default="swin_small_patch4_window7_224")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    root = args.package_root.resolve()
    output_root = args.output_root or root / "results/revised/e2e_ablation"
    run_dir = output_root / args.dataset / args.configuration / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completion = run_dir / "COMPLETE.json"
    if completion.exists():
        completed_epochs = int(json.loads(completion.read_text()).get("epochs", 0))
        if completed_epochs >= args.epochs:
            print(f"Already complete through epoch {completed_epochs}: {run_dir}")
            return
        # A smoke/evidence profile can be extended by the paper profile. The latest
        # checkpoint remains intact; only the derived completion marker is reopened.
        completion.unlink()
        print(f"Extending {run_dir} from {completed_epochs} to {args.epochs} epochs")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform = build_transforms()
    pool_train = datasets.ImageFolder(root / "datasets" / args.dataset / "val", transform=train_transform)
    pool_eval = datasets.ImageFolder(root / "datasets" / args.dataset / "val", transform=eval_transform)
    taxonomy_path = root / "config" / f"taxonomy_{args.dataset}.json"
    species_names, mapping_list = load_taxonomy(taxonomy_path, pool_train.classes)
    labels = np.asarray(pool_train.targets)
    train_indices, selection_indices = stratified_development_indices(labels, 0.2, args.split_seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(pool_train, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    selection_loader = DataLoader(
        Subset(pool_eval, selection_indices), batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    mapping = torch.as_tensor(mapping_list, dtype=torch.long, device=device)
    model = HierarchicalClassifier(
        len(species_names), len(pool_train.classes), args.model, pretrained=not args.no_pretrained
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_f1, history = 1, -1.0, []
    latest = run_dir / "latest_checkpoint.pth"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_f1 = float(state["best_selection_macro_f1"])
        history = state.get("history", [])
        print(f"Resuming {run_dir} at epoch {start_epoch}")

    config = CONFIGS[args.configuration]
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = epoch_pass(model, train_loader, device, mapping, config, args.lambda_tc, optimizer, scaler)
        selection_metrics = epoch_pass(model, selection_loader, device, mapping, config, args.lambda_tc, None, scaler)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"selection_{k}": v for k, v in selection_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        common = {
            "epoch": epoch, "model_name": args.model, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "disease_classes": pool_train.classes,
            "species_names": species_names, "disease_to_species_map": mapping_list,
            "configuration": args.configuration, "ablation_type": config["consistency"],
            "seed": args.seed, "split_seed": args.split_seed, "lambda_tc": args.lambda_tc,
            "best_selection_macro_f1": max(best_f1, selection_metrics["macro_f1"]), "history": history,
        }
        torch.save(common, latest)
        if selection_metrics["macro_f1"] > best_f1:
            best_f1 = selection_metrics["macro_f1"]
            torch.save(common, run_dir / "best_checkpoint.pth")
        print(json.dumps({"dataset": args.dataset, "configuration": args.configuration,
                          "seed": args.seed, **row}), flush=True)
    completion.write_text(json.dumps({"best_selection_macro_f1": best_f1, "epochs": args.epochs}, indent=2))


if __name__ == "__main__":
    main()
