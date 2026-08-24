"""Regenerate deterministic schematic and statistical figures for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures"
EVIDENCE_DIR = ROOT / "supplementary_evidence"
if not EVIDENCE_DIR.exists():
    EVIDENCE_DIR = ROOT / "IPA_Upload_Ready" / "supplementary_evidence"

DATASET_ALIASES = {
    "PV": "NPD-Rice-COCO-43",
    "DS1": "MultiCrop-88",
    "DS2": "Agri-Foundation-145k",
}
DATASET_ORDER = tuple(DATASET_ALIASES.values())
METRICS = (
    ("accuracy_pct", "Accuracy"),
    ("macro_f1_pct", "Macro-F1"),
    ("cross_host_error_pct", "CHER"),
)


def save_figure(fig: plt.Figure, stem: str, *, tight: bool = True) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("pdf", {}), ("png", {"dpi": 350})):
        save_options = {"facecolor": "white", **options}
        if tight:
            save_options.update({"bbox_inches": "tight", "pad_inches": 0.20})
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            **save_options,
        )
    plt.close(fig)


def normalize_dataset_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return evidence with manuscript-facing dataset names."""

    normalized = frame.copy()
    normalized["dataset"] = normalized["dataset"].replace(DATASET_ALIASES)
    return normalized


def box(
    ax,
    x,
    y,
    width,
    height,
    title,
    lines,
    face,
    edge,
    title_size=12.0,
    line_size=10.5,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#17222b",
    )
    ax.text(
        x + width / 2,
        y + height * 0.33,
        lines,
        ha="center",
        va="center",
        fontsize=line_size,
        color="#26343e",
        linespacing=1.22,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color="#586973",
        )
    )


def make_workflow() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.16,
        0.95,
        "1  DATA INTEGRITY",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#244f66",
    )
    ax.text(
        0.50,
        0.95,
        "2  CONTROLLED TRAINING",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#2c6a4f",
    )
    ax.text(
        0.85,
        0.95,
        "3  LOCKED-TEST EVIDENCE",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#8a5a20",
    )

    box(
        ax,
        0.015,
        0.65,
        0.285,
        0.21,
        "NPD-Rice-COCO-43",
        "Augmentation families grouped\n74,700 / 9,355 / 9,333",
        "#e7f1f5",
        "#39708a",
    )
    box(
        ax,
        0.015,
        0.38,
        0.285,
        0.21,
        "MultiCrop-88",
        "Duplicates and conflicts excluded\n62,214 / 7,787 / 7,769",
        "#e7f1f5",
        "#39708a",
    )
    box(
        ax,
        0.015,
        0.11,
        0.285,
        0.21,
        "Agri-Foundation-145k",
        "Fixed development/test partitions\n92,571 / 23,140 / 29,041",
        "#e7f1f5",
        "#39708a",
    )

    box(
        ax,
        0.38,
        0.61,
        0.24,
        0.24,
        "Pretrained Swin-S",
        "Shared representation\n768 dimensions\n30 epochs; validation macro-F1",
        "#e8f3ec",
        "#36815e",
    )
    box(
        ax,
        0.35,
        0.33,
        0.14,
        0.18,
        "Flat",
        "Condition\ncross-entropy",
        "#f2f5f6",
        "#60727d",
    )
    box(
        ax,
        0.51,
        0.33,
        0.14,
        0.18,
        "Dual head",
        "Condition + host\ncross-entropy",
        "#edf5ef",
        "#36815e",
    )
    box(
        ax,
        0.38,
        0.10,
        0.24,
        0.14,
        "Secondary penalties",
        r"MAE or KL($q\,\Vert\,p^h$), $\lambda=1$",
        "#f4edf6",
        "#7b4c86",
            title_size=11.2,
            line_size=9.5,
    )
    arrow(ax, (0.50, 0.61), (0.42, 0.51))
    arrow(ax, (0.50, 0.61), (0.58, 0.51))
    arrow(ax, (0.58, 0.33), (0.50, 0.24))

    evidence_boxes = (
        (0.71, "Paired performance", "10 seeds per dataset\nAccuracy, macro-F1, CHER"),
        (0.52, "Practical equivalence", "$\\pm0.25$ points\nTOST + Holm adjustment"),
        (0.33, "Hierarchy audit", "60 dual-head runs\nHost accuracy, HDR, TCI"),
        (0.14, "Confidence audit", "Validation-only scaling\nECE, NLL, Brier, AURC"),
    )
    for y, title, lines in evidence_boxes:
        box(
            ax,
            0.71,
            y,
            0.275,
            0.14,
            title,
            lines,
            "#fbf2df",
            "#a76b27",
            title_size=11.2,
            line_size=9.3,
        )

    arrow(ax, (0.30, 0.50), (0.38, 0.73))
    arrow(ax, (0.65, 0.42), (0.71, 0.76))
    arrow(ax, (0.85, 0.71), (0.85, 0.66))
    arrow(ax, (0.85, 0.52), (0.85, 0.47))
    arrow(ax, (0.85, 0.33), (0.85, 0.28))

    ax.text(
        0.50,
        0.025,
        "90 completed in-domain training runs; manifests fixed before training",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#26343e",
    )
    fig.tight_layout(pad=0.4)
    save_figure(fig, "audit_workflow")


def make_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.015, 0.36, 0.10, 0.28, "Image", "$224\\times224$\nRGB", "#f2f5f6", "#60727d")
    box(
        ax,
        0.16,
        0.30,
        0.20,
        0.40,
        "Shared Swin-S",
        "Shifted-window\nstages\nGlobal pooling\n768 features",
        "#e7f1f5",
        "#39708a",
    )
    box(ax, 0.42, 0.56, 0.16, 0.28, "Host head", "Linear $768\\to S$\nHost CE", "#e8f3ec", "#36815e")
    box(ax, 0.42, 0.16, 0.16, 0.28, "Condition\nhead", "Linear $768\\to C$\nCondition CE", "#fbf2df", "#a76b27", title_size=11.0)
    box(ax, 0.64, 0.56, 0.16, 0.28, "Host\nprobability", "Softmax $p^h$\nHost $\\hat{s}$", "#e8f3ec", "#36815e", title_size=11.0)
    box(ax, 0.64, 0.16, 0.16, 0.28, "Condition\nprobability", "Softmax $p^c$\nHost aggregate $q$", "#fbf2df", "#a76b27", title_size=11.0)
    box(ax, 0.85, 0.16, 0.135, 0.28, "Optional\nTCI", "Mask $p^c$ to $\\hat{s}$\nCompatible output", "#f4edf6", "#7b4c86", title_size=11.0, line_size=9.5)

    arrow(ax, (0.115, 0.50), (0.16, 0.50))
    arrow(ax, (0.36, 0.50), (0.42, 0.70))
    arrow(ax, (0.36, 0.50), (0.42, 0.30))
    arrow(ax, (0.58, 0.70), (0.64, 0.70))
    arrow(ax, (0.58, 0.30), (0.64, 0.30))
    arrow(ax, (0.80, 0.30), (0.85, 0.30))
    arrow(ax, (0.80, 0.70), (0.90, 0.44))
    ax.add_patch(
        FancyArrowPatch(
            (0.72, 0.45),
            (0.72, 0.55),
            arrowstyle="<->",
            mutation_scale=12,
            linewidth=1.5,
            color="#7b4c86",
        )
    )
    ax.text(
        0.62,
        0.50,
        "Optional consistency\n" + r"MAE or KL($q\,\Vert\,p^h$), $\lambda=1$",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#6b3f76",
    )
    ax.text(
        0.50,
        0.045,
        r"Standard: highest-probability condition     |     TCI: restrict classes to predicted host $\hat{s}$",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#26343e",
    )
    fig.tight_layout(pad=0.4)
    save_figure(fig, "audit_model_architecture")


def make_equivalence() -> None:
    summary = normalize_dataset_names(
        pd.read_csv(EVIDENCE_DIR / "core_equivalence_summary.csv")
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.0, 3.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    colours = {True: "#2E7D5B", False: "#C66A1B"}
    for column, dataset in enumerate(DATASET_ORDER):
        axis = axes[column]
        selected = summary[summary["dataset"] == dataset].set_index("metric")
        axis.axvspan(-0.25, 0.25, color="#DDEBDD", alpha=0.75, zorder=0)
        axis.axvline(0, color="#26343e", linewidth=0.9, zorder=1)
        for row_index, (metric, _) in enumerate(METRICS):
            row = selected.loc[metric]
            mean = float(row["mean_difference_pp"])
            lower = float(row["t_95ci_lower_pp"])
            upper = float(row["t_95ci_upper_pp"])
            equivalent = str(row["equivalent_holm_0_05"]).lower() == "true"
            axis.errorbar(
                mean,
                row_index,
                xerr=[[mean - lower], [upper - mean]],
                fmt="o" if equivalent else "D",
                color=colours[equivalent],
                ecolor=colours[equivalent],
                capsize=3,
                markersize=5.5,
                linewidth=1.35,
                zorder=2,
            )
        axis.set_title(dataset, fontweight="bold", fontsize=11.0)
        axis.set_xlabel("Dual - flat\n(percentage points)", fontsize=10.0)
        axis.tick_params(axis="both", labelsize=11.0)
        axis.grid(axis="x", alpha=0.18)
        axis.set_xlim(-1.4, 1.4)
        axis.text(
            0.98,
            0.03,
            "n=10 paired seeds",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color="#52616B",
        )
    axes[0].set_yticks(np.arange(len(METRICS)), [label for _, label in METRICS])
    axes[0].set_ylim(len(METRICS) - 0.55, -0.45)
    axes[0].tick_params(axis="y", labelleft=True)
    save_figure(fig, "core_equivalence_forest", tight=False)


def build_penalty_effects() -> pd.DataFrame:
    effects = normalize_dataset_names(
        pd.read_csv(EVIDENCE_DIR / "clean_penalty_effects.csv")
    )
    ds2 = normalize_dataset_names(
        pd.read_csv(EVIDENCE_DIR / "agri_foundation_penalty_metrics.csv")
    )
    rows = []
    for candidate, comparison in (
        ("dual_mae", "MAE - no penalty"),
        ("dual_kl", "KL - no penalty"),
    ):
        candidate_rows = ds2[ds2["configuration"] == candidate]
        reference_rows = ds2[ds2["configuration"] == "dual_none"]
        paired = candidate_rows.merge(reference_rows, on="seed", suffixes=("_candidate", "_reference"))
        for metric, _ in METRICS:
            differences = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
            ).to_numpy(dtype=float)
            mean = float(np.mean(differences))
            std = float(np.std(differences, ddof=1))
            half_width = float(student_t.ppf(0.975, len(differences) - 1) * std / np.sqrt(len(differences)))
            rows.append(
                {
                    "dataset": "Agri-Foundation-145k",
                    "comparison": comparison,
                    "candidate": candidate,
                    "reference": "dual_none",
                    "metric": metric,
                    "n_pairs": len(differences),
                    "paired_seeds": " ".join(str(seed) for seed in paired["seed"]),
                    "mean_difference_pp": mean,
                    "std_difference_pp": std,
                    "t_95ci_lower_pp": mean - half_width,
                    "t_95ci_upper_pp": mean + half_width,
                }
            )
    combined = pd.concat([effects, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(EVIDENCE_DIR / "penalty_effects_all_datasets.csv", index=False)
    return combined


def make_penalty_figure() -> None:
    effects = build_penalty_effects()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.0, 4.1),
        sharey=True,
        constrained_layout=True,
    )
    comparisons = (
        ("MAE - no penalty", "MAE", "#315B7D", -0.10),
        ("KL - no penalty", r"KL($q\,\Vert\,p^h$)", "#8A5A20", 0.10),
    )
    for column, dataset in enumerate(DATASET_ORDER):
        axis = axes[column]
        selected = effects[effects["dataset"] == dataset]
        axis.axvline(0, color="#26343e", linewidth=0.9, zorder=1)
        extent = 0.0
        for comparison, _, colour, offset in comparisons:
            comparison_rows = selected[selected["comparison"] == comparison].set_index("metric")
            for row_index, (metric, _) in enumerate(METRICS):
                row = comparison_rows.loc[metric]
                mean = float(row["mean_difference_pp"])
                lower = float(row["t_95ci_lower_pp"])
                upper = float(row["t_95ci_upper_pp"])
                extent = max(extent, abs(lower), abs(upper))
                axis.errorbar(
                    mean,
                    row_index + offset,
                    xerr=[[mean - lower], [upper - mean]],
                    fmt="o",
                    color=colour,
                    ecolor=colour,
                    capsize=3,
                    markersize=5.2,
                    linewidth=1.3,
                    zorder=2,
                )
        limit = max(0.08, extent * 1.18)
        axis.set_xlim(-limit, limit)
        axis.set_title(dataset, fontweight="bold", fontsize=11.0)
        axis.set_xlabel("Penalty - no penalty\n(percentage points)", fontsize=10.0)
        axis.tick_params(axis="both", labelsize=11.0)
        axis.grid(axis="x", alpha=0.18)
        axis.text(
            0.98,
            0.03,
            "n=5 paired seeds",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color="#52616B",
        )
    handles = [
        Line2D([0], [0], marker="o", color="#315B7D", linestyle="none", label="MAE"),
        Line2D([0], [0], marker="o", color="#8A5A20", linestyle="none", label=r"KL($q\,\Vert\,p^h$)"),
    ]
    axes[0].set_yticks(np.arange(len(METRICS)), [label for _, label in METRICS])
    axes[0].set_ylim(len(METRICS) - 0.55, -0.45)
    axes[0].tick_params(axis="y", labelleft=True)
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=9.3)
    save_figure(fig, "clean_penalty_effects", tight=False)


def main() -> None:
    make_workflow()
    make_architecture()
    make_equivalence()
    make_penalty_figure()


if __name__ == "__main__":
    main()
