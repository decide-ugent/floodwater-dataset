# Dataset card for the Floodwater Dataset

## Dataset summary

The Floodwater Dataset is a binary semantic-segmentation dataset derived from oblique UAV video of flood-affected areas in Belgium. It was curated to study floodwater segmentation and temporally adaptive inference on continuous aerial video. The video release comprises 31 source videos divided into 98 chunks, with 231,024 labeled frames. A separate manual evaluation release contains 700 manually annotated image/mask pairs.

The video labels are SAM2-assisted pseudo-labels. The evaluation release contains directly annotated masks.

## Authors and organizations

- Vishisht Sharma — IDLab, Ghent University - imec; VITO
- Sam Leroux — IDLab, Ghent University - imec
- Lisa Landuyt — VITO
- Nick Witvrouwen — VITO
- Pieter Simoens — IDLab, Ghent University - imec

The paper acknowledges the Flanders Environment Agency (VMM) and Flanders Hydraulics Research for providing aerial video used in the study.

## Supported task and classes

Binary semantic segmentation of visible water in oblique aerial frames:

| Stored pixel | Training class | Meaning |
|---:|---:|---|
| 0 | 0 | Background / non-water |
| 255 | 1 | Water |

There is no ignore-label value. The labels do not distinguish floodwater from permanent water and do not have separate classes for roads, buildings, vegetation, shadows, or reflections. Muddy water, vegetated water, wet soil, and reflection may be ambiguous.

## Composition

| Split | Source videos | Chunks | Labeled frames | Water pixels |
|---|---:|---:|---:|---:|
| Train | 22 | 68 | 159,848 | 19.32% |
| Validation | 4 | 13 | 30,017 | 11.31% |
| Test | 5 | 17 | 41,159 | 19.17% |
| Total | 31 | 98 | 231,024 | 18.29% |

The video payload has 93 1280x720 chunks and five 1920x1080 chunks, all at 25 fps. Total chunk duration is 9,541 seconds (about 2 h 39 min). Video plus masks occupy approximately 5.30 GB before archive overhead.

The manual evaluation payload contains 700 pairs: 466 at 1280x720 and 234 at 1920x1080. Each JPEG image has a same-stem binary PNG mask.

## Organization and splits

`chunks.csv` has one row per chunk and records its source video, split, paths, counts, and video SHA-256. `samples.csv` has one row per labeled frame and records a stable sample ID, source/chunk IDs, split, zero-based frame index, time, paths, label type, and mask digest.

Split membership is frozen at source-video level. This prevents neighboring chunks from the same source crossing splits. It does not establish geographic or event independence; related Belgian flood scenes may occur in different source videos.

The manual evaluation set contains 700 image/mask pairs with matching `sample_XXXXXX` identifiers and has no train/validation/test subdivisions.

## Collection and annotation provenance

The paper describes a SAM2-assisted workflow: an annotator placed positive and negative water prompts, SAM2 propagated masks through video, and visible tracking failures were corrected with additional prompts and re-propagation.

The manual evaluation set contains 700 manually annotated image/mask pairs.
