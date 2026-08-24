# Manuscript evidence ledger

## Completed experiment count

| Dataset | Flat | Dual | Dual + MAE | Dual + KL(q \|\| p^h) | Total |
|---|---:|---:|---:|---:|---:|
| NPD--Rice--COCO--43 | 10 | 10 | 5 | 5 | 30 |
| MultiCrop--88 | 10 | 10 | 5 | 5 | 30 |
| Agri--Foundation--145k | 10 | 10 | 5 | 5 | 30 |
| **Total** | **30** | **30** | **15** | **15** | **90** |

Every run completed 30 epochs. Flat and dual primary comparisons use
paired seeds 2026--2035. MAE and KL(q || p^h) comparisons use matched seeds
2026--2030.

## Primary means

| Dataset | Configuration | Accuracy | Macro-F1 | CHER |
|---|---|---:|---:|---:|
| NPD--Rice--COCO--43 | Flat | 99.409 | 98.538 | 0.038 |
| NPD--Rice--COCO--43 | Dual | 99.398 | 98.483 | 0.024 |
| MultiCrop--88 | Flat | 97.657 | 95.138 | 0.336 |
| MultiCrop--88 | Dual | 97.708 | 95.333 | 0.306 |
| Agri--Foundation--145k | Flat | 92.151 | 62.645 | 3.321 |
| Agri--Foundation--145k | Dual | 92.271 | 63.216 | 3.138 |

## Equivalence result

The predeclared margin is +/-0.25 percentage points. Holm-adjusted equivalence
is established for:

- All three NPD--Rice--COCO--43 outcomes.
- MultiCrop--88 accuracy.
- MultiCrop--88 CHER.

Equivalence is not established for MultiCrop--88 macro-F1 or any
Agri--Foundation--145k outcome. This is an inconclusive result at the stated
margin, not proof of a difference.

At the stricter +/-0.10-point margin, NPD--Rice--COCO--43 accuracy and CHER and
MultiCrop--88 CHER remain equivalent. At +/-0.50 points, accuracy and CHER also
meet equivalence on Agri--Foundation--145k; MultiCrop--88 and
Agri--Foundation--145k macro-F1 remain unresolved. The full nine-decision Holm
procedure is repeated independently at every margin.

## Locked-output diagnostic results

- Family-balanced NPD--Rice--COCO--43 dual-minus-flat effects are -0.0129
  accuracy points, -0.0307 macro-F1 points, and -0.0110 CHER points.
- On Agri--Foundation--145k, ground-truth-host masking raises accuracy by 2.294
  points for flat and 2.184 points for dual, and macro-F1 by 10.601 and 10.232
  points, respectively.
- The no-penalty dual model changes 332.3 predictions per seed under TCI. It
  gains 79.1 and loses 74.4 correct top-1 decisions, leaving a net 4.7 cases.
- The most stable descriptive dual-head macro-F1 gain occurs in classes with
  101--500 training images: +0.854 points, 95% CI [0.533, 1.176].

## Evidence gate

- NPD--Rice--COCO--43 runs validated: 30.
- MultiCrop--88 runs validated: 30.
- Agri--Foundation--145k runs validated: 30.
- Completed main-paper experiment total: 90.
- All 90 runs passed the evidence gates.
- Large checkpoint/logit hashes verified for the controlled runs.
- Core calibration seed count: 10 per dataset/configuration cell.
- Agri--Foundation--145k penalty runs independently reconstructed from saved
  predictions.
- HDR and TCI effects independently reconstructed for all 60 final dual-head runs.

## Scope

This study does not claim out-of-domain or cross-source generalizability. Its
objective is to audit hierarchical crop-condition classification under
split-integrity-controlled, in-domain evaluation.
