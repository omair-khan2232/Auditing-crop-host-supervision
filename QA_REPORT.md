# Final quality-assurance report

Date: 13 August 2026

## Evidence

- Completed experiment matrix: 90 runs.
- NPD--Rice--COCO--43 runs validated: 30.
- MultiCrop--88 runs validated: 30.
- Agri--Foundation--145k runs validated: 30.
- All 90 runs passed the evidence gates.
- Primary pairing: 10 seeds per dataset and model.
- Secondary penalties: 5 matched seeds per dataset and configuration.
- Large checkpoint/logit hashes were verified for the controlled runs.
- Agri--Foundation--145k penalty metrics were independently reconstructed from saved
  29,041-case predictions and complete 30-epoch histories.
- Equivalence decisions were repeated at +/-0.10, +/-0.25, and +/-0.50 points.
- NPD--Rice--COCO--43 family-balanced metrics were reconstructed over 5,171
  test families.
- Agri--Foundation--145k class-frequency, host-level, disagreement, and
  ground-truth-host oracle diagnostics were reconstructed from locked logits.

## Journal format

- Article type: Original Research Article.
- Abstract: 238 words (journal maximum: 250).
- Keywords: 6 (journal maximum: 6).
- Highlights: 5; longest bullet: 68 characters (maximum: 85).
- Main text: approximately 5,800 words in the LaTeX source.
- Main tables: 10.
- Main figures: 7.
- Separate figure-caption file: present.
- Numbered paper sections: present.
- CRediT, funding, competing-interest, data, and AI-use sections: present.

## Build and proof

- Main PDF: 29 pages.
- Supplement PDF: 7 pages.
- Undefined citations: 0.
- Citation/bibliography mismatches: 0.
- Undefined cross-references: 0.
- Overfull boxes: 0.
- Fatal LaTeX errors: 0.
- Full evidence-gate recheck: PASS, including large artifact hashes and all
  10-seed core/calibration cells.
- Independent HDR/TCI reconstruction: PASS for all 60 dual-head runs.
- Independent Agri--Foundation--145k penalty reconstruction: PASS for all 15
  matched penalty-block runs.
- Exact split manifests, taxonomies, pinned requirements, training/split code,
  and regeneration instructions: present.
- Main/supplement synchronization: complete for margin, family-balanced,
  long-tail, host-level, oracle-host, and disagreement analyses.
- Supplementary tables are numbered exactly S1--S3; the full HDR/TCI matrix is
  Table S2 and the consolidated reproducibility inventory is Table S3.
- Editable Highlights DOCX and 2400 x 960 graphical abstract in PNG, PDF, and
  TIFF formats: present.
- Reproducibility archive: 96 hashed artifacts plus two identical checksum
  inventories; only the seven cited figures are included (PDF and PNG).
- Portal folder: clearly named manuscript, supplement, source ZIP, highlights,
  graphical abstract, cover letter, captions, declarations, and action gate.
- Title page, every table/figure page, limitations, declarations, references,
  and all supplementary pages were visually inspected.

## Scope and separation

- The approved in-domain scope statement appears in the paper.
- Separate diagnostic material and source-held-out claims are absent from the
  submission package.
- No submitted figure uses generative AI artwork.

## Author-owned fields still required

- Author names and order.
- Full affiliation and postal address.
- Corresponding-author email.
- CRediT contribution roles.
- Verified funding statement and grant identifiers.
- Acknowledgements, or removal of that section.
- Permanent public code and derived-evidence DOI/URL.
