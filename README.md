# Auditing Crop-Host Supervision

This repository contains the reproducibility package for the study:

**Does crop-host supervision improve Swin-based multi-crop condition recognition? A controlled 90-run in-domain audit of performance, consistency, and confidence**

The package includes the manuscript source, reproducibility scripts, split/taxonomy metadata, derived evidence tables, and publication figures. Original image datasets and model checkpoints are not redistributed; dataset source links and split manifests are provided for reproducibility.

## Repository Layout

- `reproducibility_code/`: training, validation, audit, calibration, and figure-generation scripts.
- `reproducibility_metadata/`: requirements, taxonomies, and locked split manifests.
- `supplementary_evidence/`: derived CSV evidence used for reported analyses.
- `figures/`: publication figures in image/PDF formats.
- `paper/`: LaTeX source, compiled manuscript PDF, supplementary material, cover letter, highlights, and table files.
- `submission_assets/`: journal-upload helper files such as graphical abstract and declarations.

## Notes

The work is an in-domain audit of hierarchical crop-condition classification. It does not claim out-of-domain or cross-source generalizability.
