# Information Processing in Agriculture: final journal-fit review

## Overall assessment

**Scientific fit: 8.6/10.**

**Technical package readiness: 8.6/10.** The package must not be uploaded until
the author metadata, CRediT roles, funding statement, acknowledgements decision,
and permanent repository link are completed. Expected readiness after those
author-owned items is approximately **8.8/10**.

The no-retraining revision addresses the review's main presentation,
implementation, equivalence-margin, family-weighting, long-tail, literature,
supplement-synchronization, and repository-metadata concerns. The score remains
below 9 because host-loss weighting, another backbone, and external-domain
evidence were not added; those experiments would broaden the tested claim and
are not necessary for the paper's deliberately in-domain audit scope.

The paper is now defensible as an Original Research Article whose novelty lies
in evaluation design and actionable evidence, not in a new neural architecture.
Its strongest editorial argument is that a multi-crop model can be accurate,
taxonomically inconsistent, and miscalibrated at the same time; those properties
must be audited separately before a host-aware design is treated as useful.

## Why the audit is needed

- Multi-crop screening and referral require crop and condition outputs that are
  compatible, not merely a high pooled accuracy.
- An auxiliary host label is deterministic in these datasets, so adding a host
  head can create an appearance of confirmation without adding new annotation.
- Predicted-host masking guarantees valid output pairs but can propagate a wrong
  host decision. Pre-TCI HDR and post-TCI performance changes are therefore
  needed to interpret it.
- Near-saturated benchmarks make fractions of a percentage point easy to
  overinterpret. Fixed identities, paired seeds, and equivalence testing address
  that problem directly.
- Confidence can govern in-domain acceptance or referral. Calibration and
  risk-coverage evidence are therefore more operationally useful than accuracy
  alone.

## Section-by-section result

| Element | Assessment |
|---|---|
| Title | Specific, searchable, and explicit that the 90-run evidence concerns Swin-based recognition. |
| Abstract | Within the 250-word limit; states the need, protocol, numerical results, and practical conclusion. |
| Introduction | Leads with screening, triage, contradictory outputs, and the decision faced by model developers. |
| Related work | Adds a structured comparison of crop-conditional, multi-output, and rule-audit studies to make the evaluation novelty explicit. |
| Methods | Reproducible split controls, exact pretrained identifier, deterministic training, loss implementation details, experiment matrix, metrics, and statistical decision rules. |
| Results | Separates descriptive means, margin sensitivity, family-balanced effects, long-tail and oracle diagnostics, penalties, HDR/TCI, calibration, and selective risk without claiming unsupported superiority. |
| Discussion | Converts the findings into design decisions and distinguishes validity constraints from correctness. |
| Figures | Seven readable, data-linked figures; the new long-tail/oracle panel exposes where host information can and cannot help. |
| Tables | Dataset identities, run counts, units, seed counts, uncertainty, and direction of deltas are explicit. |
| Limitations | Correctly excludes field, cross-source, and out-of-domain generalizability claims. |

## Metric assessment

The metric set is appropriate for the stated questions:

- Accuracy measures pooled top-1 recognition.
- Macro-F1 exposes class-balanced performance and the large tail-class weakness
  on Agri-Foundation-145k.
- CHER measures whether the condition prediction crosses the labelled host.
- Host accuracy and HDR separate host correctness from agreement between heads.
- TCI deltas show whether forced validity changes recognition.
- ECE, NLL, and Brier score evaluate probability quality.
- AURC and risk-coverage curves evaluate confidence-ranked retention.

ROC/AUC and a large multiclass confusion matrix are not necessary for the
current research questions and would add volume without resolving the
hierarchy-audit claim. They should be added only if a reviewer requests a
specific one-vs-rest or class-level diagnostic.

## Comparison with recent IPA papers

Recent IPA articles commonly begin with an agricultural workflow need, state an
explicit computational gap, provide multi-metric evidence, and close with
practical limitations. Audit-oriented work in the journal also emphasizes
deduplication, calibrated probabilities, referral/risk evidence, rule
consistency, and traceable artifacts. This paper now follows that pattern while
remaining distinct: its central contribution is a paired, multi-seed,
hierarchy-specific equivalence and consistency audit across three taxonomy
sizes.

## Main reviewer risks

- Algorithmic novelty is modest because Swin-S and the losses are established.
- Only one backbone is evaluated; the claim must remain component-specific.
- The study intentionally does not test field or cross-source generalization.
- MultiCrop--88 and Agri--Foundation--145k have exact-hash but no perceptual or
  embedding-similarity near-duplicate audit.
- The host-loss and consistency-loss weights are fixed at one; penalty
  comparisons use five seeds.
- The rarest 215-class strata remain highly uncertain despite the added
  frequency, class, host, disagreement, and oracle diagnostics.

These are limitations, not fatal contradictions, because the title, questions,
claims, and conclusion are restricted to the controlled in-domain audit.

## Required author actions

- Replace every bracketed author, affiliation, and email field.
- Insert verified CRediT roles and funding information.
- Add acknowledgements or remove that section.
- Deposit the code and derived evidence and insert the permanent DOI or URL.
- Confirm dataset citation and licence details.
- Have every author manually verify the plots and approve the AI-use disclosure.

## Submission recommendation

Submit as an **Original Research Article** after the author-only fields are
completed. Keep the earlier unmatched CNN/ResNet, lambda-sweep, and separate
cross-source experiments outside the main paper; adding them would weaken the
controlled comparison unless they are rerun under the same manifests, seeds,
and validation rules.

## Sources reviewed

- Information Processing in Agriculture, Guide for Authors:
  https://www.sciencedirect.com/journal/information-processing-in-agriculture/publish/guide-for-authors
- Fu et al., "Crop pest image recognition based on the improved ViT method":
  https://doi.org/10.1016/j.inpa.2023.02.007
- Ergun and Okumus, "Robust eggplant disease recognition using a learnable
  weighted deep ensemble with test-time augmentation":
  https://doi.org/10.1016/j.inpa.2026.03.010
- Singh et al., "Neuro-symbolic AI for rice disease diagnosis with calibrated
  attention and rule-aware explanations":
  https://doi.org/10.1016/j.inpa.2026.02.006
- Ngugi et al., "Recent advances in image processing techniques for automated
  leaf pest and disease recognition---A review":
  https://doi.org/10.1016/j.inpa.2020.04.004
- El Hanafy et al., "Exploring attention mechanisms in deep learning for plant
  health: A scoping review on disease and pest detection":
  https://doi.org/10.1016/j.inpa.2026.05.006
