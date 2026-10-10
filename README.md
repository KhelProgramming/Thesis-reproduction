# AgriEye baseline reproduction setup

This project reconstructs the training recipe shown in AgriEye's manuscript, audits an exported dataset, trains YOLOv8n-Seg, and records box and mask evaluation separately. It contains **no new illumination augmentation**. The paper's existing HSV, geometric, mosaic, and mixup augmentation remains enabled.

The supplied manuscript is identified by SHA256 `5d772ae210c99c6c39a507c99350dc037f9d955e619ee836eb2a1ab4191f60b2`. Appendix K's training code is on printed pages 158–161 (PDF pages 175–178). Its manual transcription is in `references/manuscript_training_extracted.py`.

## What can currently be reproduced

| Profile | Evidence | Interpretation |
|---|---|---|
| `configs/disease_paper_50e.yaml` | Appendix: 50 epochs, batch 16, image size 640, AdamW, explicit augmentation | Reimplementation of the documented recipe. With the supplied four-class export, this is a four-class pilot. |
| `configs/disease_v8_100e_candidate.yaml` | Table 12 v8: 100 epochs, batch 16, five classes | Other settings are inherited assumptions. Requires the five-class dataset. Not a verified final-model recipe. |
| `configs/leaf_paper_200e.yaml` | Figure 13: leaf model, 200 epochs, batch 16, image size 640 | Optional, separate leaf-mask dataset required. This profile adapts the paper's two GPUs to one. |

The supplied export lists `BLB, BLS, LEAF BLAST, NBS`. The released disease model and appendix's evaluation list `BLB, BLS, HEALTHY, LEAF BLAST, NBS`. Numeric IDs differ from ID 2 onward. Keep those tasks separate. Dataset version 5 does not imply experimental model v5; the numbering systems are unrelated unless the authors confirm otherwise.

The excerpt's training run is named `yolov8n_FinalModel#5 - Fine tune`, whereas its evaluation points to a `...Stage2/weights/best.pt` checkpoint. Table 12's final model is v8. The paper does not supply enough information to establish that all three refer to the same run.

## Directory layout

| Directory | Contents |
|---|---|
| `data/raw/` | Optional place to put extracted original exports; existing external folders also work. |
| `data/prepared/<name>/` | Audited independent copy, corrected dataset YAML, hashes, and provenance. |
| `configs/` | Documented recipe, clearly marked candidate recipe, and evaluation settings. |
| `references/` | Transcribed paper code, published values, source notes, and supplied export metadata. |
| `runs/<run-id>/` | Logs, environment, seed, dataset fingerprint, checkpoints, CSVs, evaluation, and progress. |
| `reports/` | Dataset audits, benchmark tables, test-access log, and records ZIPs. |
| `weights/` | Official starting checkpoint downloaded on first use, or your separately supplied weights. |
| `notebooks/` | Optional Colab launcher using this same project. |

## 1. Install in an isolated environment

Run commands from the extracted `agrieye_reproduction` folder. For the Fedora laptop that already has working GPU PyTorch:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py doctor
```

Using `--system-site-packages` reuses the existing PyTorch installation. This is a practical environment adaptation, not a recovery of the authors' original environment. The runner records actual versions using `pip freeze`. Keep the same environment for later comparisons.

Ultralytics `8.4.41` is pinned because that version appears in the released TFLite metadata; the original training version remains unconfirmed. Its public package and versioned source were checked. Python, PyTorch, CUDA, and all dependencies are not claimed to reproduce the authors' environment. The doctor command imports the packages and exercises CUDA arithmetic. A successful doctor check does not establish that segmentation batch 16 fits 4 GB VRAM.

For dataset preparation alone, `python -m pip install -r requirements-audit.txt` is sufficient. No GPU is needed.

## 2. Prepare the dataset without changing it

Use the **extracted folder** containing `data.yaml`, `train`, `valid`, and `test`:

```bash
python run.py prepare --source "/absolute/path/to/AgriEye_export" --name agrieye_current
```

This audits image–label pairing, polygons, class IDs, counts, and split-overlap candidates. It copies the dataset into `data/prepared/agrieye_current`, rewrites only the working YAML paths, and fingerprints the actual image and label bytes. Do not repeat Roboflow resizing, contrast stretching, or augmentation on an already processed export.

Malformed annotations or unreadable images stop preparation. Filename-family overlap is a candidate for review, not automatically proven leakage. Exact image overlap is explicitly recorded. The code does not move images between splits or relabel classes. A historical split can be studied for reproduction, but overlap disqualifies it from an unqualified independent-generalization claim. Resolve split provenance before the later augmentation study.

The supplied README reports 7,652 images, 640×640 stretching, contrast stretching, brightness ±9%, exposure ±15%, flips, blur, and salt-and-pepper noise. The full audit establishes actual counts. Neither the README nor two sample images establish that all original photographs were captured under sunny conditions.

## 3. Run one diagnostic training check

```bash
python run.py smoke --dataset agrieye_current
```

This uses the same architecture and 640-pixel input, at most 32 training and 16 validation images, one epoch, batch 1, and no final test images. It checks the data loader, forward/backward training, checkpoints, metric extraction, and records. Its scores are not scientific results and are excluded from benchmark tables by default.

The first model load downloads the official `yolov8n-seg.pt` checkpoint. Your images and annotations remain local to your chosen runtime. The prepared working copy protects raw data from framework caches or automatic file repairs.

## 4. Run the documented baseline recipe

After the audit and smoke run are understood:

```bash
python run.py train --dataset agrieye_current --recipe configs/disease_paper_50e.yaml --seed 0
```

This preserves the paper's batch 16 and early stopping patience 20; 50 is the maximum epoch count. A 4 GB GTX 1650 may not fit this batch. Prefer a GPU that fits the recipe when testing reproducibility. If using a smaller batch, make it explicit:

```bash
python run.py train --dataset agrieye_current --recipe configs/disease_paper_50e.yaml --seed 0 --batch 2 --workers 2
```

That is a recorded hardware-adapted run. Gradient accumulation, warmup, and batch-normalization behavior mean it is not automatically equivalent to batch 16. The runner counts actual optimizer steps via a PyTorch optimizer hook. If the framework attempts an automatic OOM retry, the run stops rather than silently reporting a changed batch as the requested recipe.

Only after obtaining the matching five-class dataset and confirming its relationship to the reported final run should you use:

```bash
python run.py train --dataset agrieye_final5 --recipe configs/disease_v8_100e_candidate.yaml --seed 0
```

The class-order check prevents applying that profile to the current four-class export. Confirm or revise the candidate recipe from author run files before calling it final-model reproduction. Use a new, documented recipe file for changes; do not modify a past run's records.

## 5. Read results and evaluate the test set once the protocol is fixed

Every successful run performs validation and prints a box/mask summary. It saves original Ultralytics `fit/results.csv`, `fit/args.yaml`, `fit/weights/best.pt`, and `fit/weights/last.pt`, plus the wrapper's records. The best checkpoint follows the pinned framework's fitness rule: for segmentation in 8.4.41, this combines box and mask mAP50–95. It is not silently changed to a mask-only selector.

```bash
python run.py benchmark
python run.py evaluate --run "RUN_ID_PRINTED_AFTER_TRAINING" --split test
python run.py bundle-records
```

Test evaluation is never run automatically by training or smoke checks. It is explicitly requested and logged. Fix the dataset, recipe, checkpoint rule, and intended seeds before using final test scores. Do not change settings repeatedly to make the test score approach a published number.

Each evaluation saves overall box/mask metrics, per-class CSVs using the actual AP class indices, inference timing, evaluation arguments, and environment. It records both macro-F1 and the harmonic mean of aggregate precision/recall because they are not generally identical. The paper's standalone IoU value is retained as a reference but not invented from mAP; its exact matching and averaging code is missing.

`benchmark` prints measured results and separately lists Table 12's published values. It does not claim successful reproduction based only on numerical similarity. Match the original dataset/splits, class order, checkpoint initialization, training recipe, metric type, and evaluation split first. Then use a modest predefined set of seeds and report mean, variability, and differences from the reference. Decide what practical difference matters before viewing final outcomes; there is no universal acceptable tolerance.

## Required evidence before claiming final-model reproduction

Request the exact dataset export containing Healthy, original train/validation/test assignment, final run's `args.yaml` and `results.csv`, `best.pt`, initial checkpoint and any staged fine-tuning history, dependency versions, and the metric namespace/split used in Table 12. The paper's appendix prints box metrics; other parts discuss mask metrics. That uncertainty must be resolved before matching a particular number.

A source recipe reimplementation can still provide a defensible matched baseline if exact original assets are unavailable, but that is a change in the reproduction claim. The later augmentation study should compare freshly trained baseline and intervention under one fixed protocol. This project does not implement that intervention yet.

## Records and scope

The project records seed, recipe, deviations, resolved training arguments, actual batch, actual AMP state, optimizer steps, class order, original checkpoint hash, dataset fingerprint, package versions, logs, checkpoints, and evaluation outputs. Optional Albumentations installed in the environment can enable Ultralytics default transforms; the package version and dataset transform representation are recorded. Keep that environment consistent.

The main scope is the disease-segmentation component. The optional leaf profile requires separate leaf masks. Android gating, custom postprocessing, severity calculations, TFLite export parity, and a complete app benchmark are not established by these component results. Add them only if the thesis claims improvement of the whole system.

See `VERIFICATION.md` for what was actually executed while preparing this package. A smoke test on synthetic data is a software integration check, not evidence of rice-disease performance.
