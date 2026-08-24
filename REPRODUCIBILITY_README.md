# Reproducibility guide

This archive supports the paper, "Does crop-host supervision improve
Swin-based multi-crop condition recognition? A controlled 90-run in-domain
audit of performance, consistency, and confidence."

## Archive contents

- `paper/`: editable LaTeX, compiled proofs, highlights, captions, and cover letter.
- `figures/`: the seven cited figures in PDF and PNG formats.
- `supplementary_evidence/`: per-seed and aggregate evidence used by the paper.
- `reproducibility_code/`: training, split-audit, statistical, calibration,
  hierarchy-audit, and figure-generation scripts.
- `reproducibility_metadata/manifests/`: the three exact split manifests loaded
  by the final runs.
- `reproducibility_metadata/taxonomies/`: ordered condition classes, host names,
  and condition-to-host mappings.
- `reproducibility_metadata/requirements.txt`: pinned analysis/training packages.
- `submission_assets/`: editable highlights and deterministic graphical-abstract
  exports.
- `FILE_MANIFEST.sha256` and `OUTPUT_MANIFEST.sha256`: identical SHA-256
  inventories for every submitted file other than the two checksum manifests.

The image datasets are not redistributed. Their source-specific licences and
terms still apply. Model checkpoints and full logit archives are also too large
for the journal-facing bundle; they should be deposited as separate repository
assets when the permanent DOI is created. The submitted CSV files are the
machine-readable inputs behind the reported tables, and the supplied scripts
reconstruct them from the locked outputs when those large assets are present.

## Locked identities

| Dataset | Split-manifest SHA-256 |
|---|---|
| NPD-Rice-COCO-43 | `de4fd19299c735a4ab939d67288f01757822a671cfb8d6c41b95f71d5fdd34ae` |
| MultiCrop-88 | `b881f3e7feff8006f03a4d1e0066abf88cddda79dbee9295be3157cb11842231` |
| Agri-Foundation-145k | `ae87397218c8124721117e401291f10a3426ec0f94791b155a24fe3245c10732` |

The taxonomy JSON files preserve the internal run identifiers so their hashes
and class ordering remain traceable. Manuscript-facing text uses the full
dataset names shown above.

The exact pretrained artifact was:

- `swin_small_patch4_window7_224.ms_in22k_ft_in1k`
- Hugging Face revision `221ee757828e4772df860bd42ef476de78dd5fc6`
- model SHA-256 `e17adf843764761f138cb8cab081d9a37824db4a948c0082c7af7f0092d19581`

The recorded environment used Python 3.11.14, PyTorch 2.9.0 with CUDA 12.9,
and an NVIDIA GeForce RTX 5070. Package versions are pinned in
`reproducibility_metadata/requirements.txt`.

## Regenerating the added audits

From the archive root, with `<LOCKED_OUTPUT_ROOT>` replaced by the directory
containing the final run folders and locked logits:

```powershell
python reproducibility_code/audit_revision_analyses.py `
  --package-root "<LOCKED_OUTPUT_ROOT>" `
  --evidence-dir "supplementary_evidence" `
  --figure-dir "figures"

python reproducibility_code/analyze_tci_hdr.py `
  --package-root "<LOCKED_OUTPUT_ROOT>" `
  --output-dir "supplementary_evidence"

python reproducibility_code/controlled_calibration_analysis.py `
  --package-root "<LOCKED_OUTPUT_ROOT>" `
  --output-dir "<CALIBRATION_OUTPUT_DIR>" `
  --strict-target

python reproducibility_code/make_paper_figures.py
python reproducibility_code/make_submission_assets.py
```

The first command regenerates equivalence-margin sensitivity, CHER event
counts, family-balanced NPD metrics, Agri-Foundation-145k class and host
analyses, the disagreement decomposition, the ground-truth-host oracle, and
Figure 5. The second reconstructs HDR and TCI results for all 60 dual-head
runs. The third regenerates calibration evidence and Figures 6--7. The fourth
redraws Figures 1--4 from submitted evidence. The fifth regenerates the
graphical abstract and editable highlights.

The remaining scripts expose their required locations through `--help`:

```powershell
python reproducibility_code/paired_equivalence_analysis.py --help
python reproducibility_code/controlled_calibration_analysis.py --help
python reproducibility_code/validate_agri_foundation_penalties.py --help
python reproducibility_code/train_e2e_ablation.py --help
```

## Building the documents

Run twice from `paper/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

Before public deposit, attach the large locked-output assets if permitted,
replace the repository placeholder in the manuscript with the permanent DOI,
and verify the resulting archive against either checksum manifest.
