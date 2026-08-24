# IPA submission package

Target journal: **Information Processing in Agriculture**

Paper title: **Does crop-host supervision improve Swin-based multi-crop
condition recognition? A controlled 90-run in-domain audit of performance,
consistency, and confidence**

## Main files

- paper/main.tex: editable paper source.
- paper/main.pdf: compiled 29-page proof.
- paper/supplementary_material.tex: editable supplementary source.
- paper/supplementary_material.pdf: compiled 7-page supplement.
- paper/highlights.txt: five journal-compliant highlights.
- paper/cover_letter_IPA.md: updated cover letter.
- paper/figure_captions.txt: separate figure-caption file.
- paper/SUBMISSION_READINESS.md: author-only completion checklist.
- figures/: seven referenced figures in PDF and PNG.
- supplementary_evidence/: manuscript-only machine-readable evidence.
- reproducibility_code/: relevant analysis and figure scripts.
- reproducibility_metadata/: exact manifests, host mappings, and requirements.
- REPRODUCIBILITY_README.md: artifact map and regeneration commands.
- evidence_status.json: submission-specific 90-run evidence gate.
- submission_assets/Graphical_Abstract.{png,pdf,tiff}: deterministic graphical
  abstract at 2400 x 960 pixels.
- submission_assets/Highlights.docx: editable journal highlights file.
- paper/cover_letter_IPA.pdf: compiled one-page cover-letter draft.
- Journal_Upload_Ready/: clearly named portal files, including Manuscript.pdf,
  Supplementary_Material.pdf, editable LaTeX source, highlights, graphical
  abstract, cover letter, declarations, and the author-action gate.

## Build

From the paper directory, run pdflatex twice on main.tex and twice on
supplementary_material.tex. The source uses the Elsevier elsarticle class and
only locally available packages. The final tested logs contain no overfull
boxes, unresolved references, undefined citations, or fatal LaTeX errors.

## Before upload

Complete every bracketed author field listed in
paper/SUBMISSION_READINESS.md, rebuild both PDFs, and verify that the author
order matches the journal submission system. Deposit any permitted large
locked-logit/checkpoint assets separately and insert the permanent DOI.
Do not upload the current portal aliases until AUTHOR_ACTION_REQUIRED.md is
complete.
