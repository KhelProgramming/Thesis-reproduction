# Preserve, inventory, and compare AgriEye sources

This package keeps the authors' files intact. It does not remove Tungro, change class IDs, merge datasets, edit training configurations, or start training. The previously produced `agrieye_v6_without_tungro.zip` is an optional derivative and should not be used as the original source for this audit.

## Run in your local project

Extract this package into `/home/michaelreybelga/Downloads/agrieye_inventory_package`, then run:

```bash
project='/home/michaelreybelga/Fourth-Year/Thesis/agrieye_reproduction'
"$project/.venv/bin/python" \
  '/home/michaelreybelga/Downloads/agrieye_inventory_package/inspect_agrieye_sources.py' \
  --project "$project" \
  --disease '/home/michaelreybelga/Downloads/AgriEye_Verified disease no leaf finalch.v6i.yolov8.zip' \
  --healthy '/home/michaelreybelga/Downloads/Healthy.v1i.yolov8.zip' \
  --final '/home/michaelreybelga/Downloads/AgriEye_Verified disease.v5i.yolov8' \
  --stage
```

The source arguments accept either the original ZIP or the extracted dataset directory with `data.yaml` directly inside it. PyYAML is required; it is already used by the supplied reproduction project. If the Python interpreter path differs, use your project's active interpreter. No network connection is required.

The script copies all three datasets into a fresh `data/raw_author/<timestamp>/` directory. For ZIPs it preserves the original archive and creates an extracted working copy; identical duplicate ZIP entries are consolidated in the extracted copy. The original Downloads files remain unchanged. This registers sources for inventory only: it does not change `run.py`, its prepared dataset alias, or the training input. Copying the final dataset requires enough disk space for another complete copy. Omit `--stage` for a read-only inventory of the Downloads sources.

Reports are written to `reports/source_inventory_<timestamp>/`. Send that report folder as a ZIP after running it; the full image dataset is not needed for reviewing the inventory.

## Reports

- `directory_tree_before.txt` / `directory_tree_after.txt`: project structure through three levels; image/label/run/weight bulk folders summarized.
- `dataset_summaries.json`: source path, YAML, splits, empty/missing labels, polygon/box counts, Tungro-named counts, and orphan annotations.
- `class_counts.csv`: images with an actual class annotation and polygon instances, per split and overall. These are different units; images may carry both LEAF and lesion classes.
- `class_id_mapping.csv`: IDs and declared class meaning in each source. No IDs are changed.
- `image_manifest.csv`: SHA-256, label status, observed IDs, split, source-name key, polygon count, and semantic polygon hash.
- `matches.csv`: every exact-byte match or source-name candidate edge. Multiple candidates are kept rather than collapsed into an invented one-to-one correspondence.
- `raw_without_final_match.csv` / `final_without_raw_match.csv`: unmatched files after the documented matching rules.
- `exact_duplicate_image_groups.json`: repeated image hashes with dataset/split paths, allowing duplicate and split-overlap review.
- `annotation_issues.csv`: malformed class IDs, geometry, coordinates, or missing labels.
- `invocation.json`: run arguments, staged locations, and a record that original files/training configuration were not modified.

For identical image bytes, the script also compares polygons using declared class names rather than numeric IDs. An equal semantic hash requires the same coordinate values and row content after class-name interpretation; geometrically equivalent polygons with a different vertex ordering may still compare different. This strict comparison does not transform raw polygons onto augmented tiles. For differently encoded/tiled/augmented images, source-name matching is only an ancestry candidate. It cannot establish the random transformation or expert correctness of the annotations.

## Comparison completed in this session

The current runtime cannot access your `/home/michaelreybelga/...` directories. `verified_comparison/` therefore compares the two original uploaded ZIPs with the v5 manifest in your earlier `dataset_contents_check_20261007T055731327473Z.zip`. It is not a fresh scan of your laptop. The local command above provides that fresh scan and exact-image polygon comparison when the files exist.

Observed source totals:

- Disease v6: 1,145 images, 1,145 labels, 5,011 polygons, 210 empty labels, 35 Tungro-named images.
- Healthy v1: 197 images, 197 nonempty labels, 197 polygons; all in train.
- v5 manifest: 7,652 images; declared classes BLB, BLS, LEAF BLAST, NBS.

All 7,652 v5 files have a candidate source-name match in disease v6. None is byte-identical to a raw image. All 197 Healthy source images lack a candidate match in v5. There are 8,784 candidate edges because some raw source names repeat: 80 raw source-key groups contain multiple files, with up to three candidates per group. Therefore an exact raw-file ancestry assignment has not been proved. The v5 source-name set has 1,064 keys, compared with 1,145 disease files. Its 272 Tungro-named derivatives are recorded; Tungro images were preserved.

| Declared meaning | Raw disease v6 ID | Healthy v1 ID | v5 export ID | Paper's final disease order |
|---|---:|---:|---:|---:|
| BLB | 0 | — | 0 | 0 |
| BLS | 1 | — | 1 | 1 |
| LEAF | 2 | — | — | — |
| Healthy | — | 0 | — | 2 |
| LEAF BLAST | 3 | — | 2 | 3 |
| NBS | 4 | — | 3 | 4 |

Healthy ID 0 and disease ID 0 have different meanings. LEAF ID 2 is not Healthy. The v5 ID 2 is Leaf Blast. Blind concatenation or ID equality would corrupt the labels. This package compares those meanings without remapping them.

The v5 export is author-supplied, but its four-class mapping means it has not been established as the final five-class deployed model's training dataset. Source-name coverage does not settle that distinction.

## Paper pipeline extracted for the following step

Disease preparation, printed p. 47 / PDF p. 64:

- Roboflow preprocessing; 1024×1024 base resolution.
- A 2-row × 2-column tiling grid, described as overlapping segments.
- Auto-orientation.
- Stretch resulting tiles to 640×640.
- Auto-contrast using contrast stretching.

Dataset splitting, printed p. 48 / PDF p. 65: randomized Roboflow splitting described as 80/10/10 and ×2 augmentation. Reported final disease counts are 6,336 train, 664 validation, 672 test (7,672). These exact counts are not an exact 80/10/10 proportion. Table 10 and prose also disagree on totals; preserve these discrepancies in the reproduction record.

The provided v5 `README.roboflow.txt` supplies the offline augmentation ranges:

- Horizontal flip probability 50%.
- Vertical flip probability 50%.
- Brightness −9% to +9%.
- Exposure −15% to +15%.
- Gaussian blur 0 to 0.8 pixels.
- Salt-and-pepper noise on 1.41% of pixels.
- Two versions of each source image, according to export metadata.

That README lists auto-orientation, stretch to 640×640, and contrast stretching, but does not expose tile positions/overlap, split RNG seed and membership history, augmentation random draws, clipping behavior for segmentation polygons, or the exact order/implementation of every transformation. The paper provides a tile grid but not those missing parameters. Matching the ranges alone is not proof of pixel-identical outputs.

Training-time augmentation is a separate stage: the shown 50-epoch training cell uses HSV H/S/V 0.015/0.5/0.4, degrees 10, translate 0.1, scale 0.4, vertical/horizontal flip 0.5, mosaic 1.0, mixup 0.1, AdamW, lr0 0.0001, lrf 0.01, momentum 0.937, weight decay 0.0005, patience 20, cosine schedule, 640 image size, batch 16, and AMP. The selected Table 12 v8 row instead reports 100 epochs/five classes; the evaluation appendix points to a separately named Stage2 checkpoint. That final-run provenance remains unresolved. Do not silently replace it with an assumed 50+50 run.

## Decision for credible reproduction

Preserve the author exports and finish membership, ID, polygon, and split comparisons first. Keeping Tungro out of a class list does not exclude its images: empty labels are normally negative/background training examples. Its later inclusion or exclusion must be an explicit training-manifest choice based on the authors' procedure. No such choice is made here.

One-to-one raw-file provenance, equal preprocessing/training procedure, pixel-identical augmented outputs, and matching model metrics are separate claims. Use the existing author export directly when validating its recorded benchmark. Regenerating it can only be called exact after pixels, polygon geometry, and split membership have been checked. Stochastic training also requires the same initialization, environment, and random-state provenance for a strict replay; equal metrics cannot be guaranteed merely by copying the published hyperparameters.

The next step after the local report is to resolve any source-key ambiguities and identify the actual five-class final input. Then reconstruct only the documented offline pipeline, compare its generated images/labels/splits against the target export, and label any remaining assumptions explicitly. Custom thesis lighting augmentation comes after a verified baseline.
