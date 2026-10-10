# AgriEye Stage 1 testing project

This is an isolated scaffold for training YOLOv8n-seg on four rice disease
classes plus intentional empty-label negative images.

It reproduces the displayed Appendix K training values (printed pp. 159-160)
and adds explicit data checks and reproducibility choices. It is an
AgriEye-inspired reconstruction, not a verified copy of the authors' full
historical training pipeline.

## Project location on your Linux machine

Use a separate subdirectory beneath your existing project, for example:

    /home/michaelreybelga/Fourth-Year/Thesis/agrieye_reproduction/testing_stage1/

Extract this ZIP there and enter the agrieye_stage1_project folder. The scaffold
was prepared in a separate workspace; it has not changed your Linux home folder.

Folders:
- configs: paper training values and our separate reproducibility choices.
- datasets/source_archives: unchanged source ZIPs, supplied separately.
- datasets/stage1_disease: the generated four-class Roboflow export.
- reports: audits, resolved YAML, training plan and environment.
- runs: model checkpoints and Ultralytics outputs.

## Dataset prerequisite

Finish the current Roboflow disease version first:
- Keep all 1,145 originals and the existing 768 / 188 / 189 assignments.
- Exclude LEAF through version-level Modify Classes.
- Keep null/negative images; do not enable Filter Null.
- Complete and document preprocessing and augmentation.
- Export YOLOv8 segmentation with four class names.
- Healthy stays separate until the later five-class fine-tuning stage.

The current author ZIP has five classes including LEAF. The checker deliberately
rejects it as Stage 1 input. Empty label files are valid negatives, not missing
labels. Excluding LEAF leaves one additional negative validation image.

Conditional generated expectation: 6,144 train / 752 validation / 756 test,
total 7,652. The script does not require those counts or alter data to reach them.

## Install

Recommended environment: Python 3.10 or 3.11 with a supported GPU setup.
From this project directory:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Ultralytics 8.3.79 is our explicit version choice, not a recovered author version.
PyTorch/torchvision/CUDA versions are recorded per training run; use the same
environment for the baseline and augmentation comparison.

For dry-run only, PyYAML is sufficient; YOLO and PyTorch are imported only when
actual training starts.

## First action: audit without training

After extracting the generated dataset under datasets/stage1_disease:

    python train_stage1.py --data datasets/stage1_disease/data.yaml --dry-run

The checker requires all three splits, the four target classes, valid segmentation
label format, paired labels and intentional training negatives. It rejects exact
image-byte duplicates across splits. It does not prove that related source images
are split independently: source-family grouping must be checked separately.

Roboflow's ../train/images paths are normalized only if absent and a matching
export-local split folder exists; the audit records the fallback. Source YAML,
images and labels remain unchanged.

## Training

Run the reproduced 50-epoch example:

    python train_stage1.py --data datasets/stage1_disease/data.yaml

Default starting weights: yolov8n-seg.pt. These may download on first actual run.
Default configuration: imgsz=640, batch=16, AdamW and the paper's online
augmentation settings. Our added controls: seed=0, deterministic=true, workers=2.

The displayed 50 epochs is not proof that the authors' first stage used 50.
Choose and record your project's budget before comparing variants. Early
stopping can stop the run before the maximum budget.

A small GPU smoke run before a full run:

    python train_stage1.py --data datasets/stage1_disease/data.yaml --epochs 1 --batch 2 --name stage1_smoke --report-dir reports/smoke

If CUDA is unavailable, explicitly choose --device cpu for a smoke run; CPU
full training is likely slow. If memory is limited, --batch 8 is supported by
the paper's comment. Keep comparison settings consistent and record the change.

Inspect the smoke run before starting the full run. A dry run does not verify
GPU training or actual Ultralytics data loading.

## Evaluate Stage 1

    python evaluate_stage1.py --weights runs/stage1_four_disease/weights/best.pt --data reports/stage1_data_resolved.yaml --split test

The saved metric keys distinguish bounding-box (B) and mask (M) metrics. Use
validation for checkpoint selection; hold the test set for final evaluation.

Stage 1 has four classes and must not be compared directly as if it were the
paper's final five-class V8 benchmark.

## Later Stage 2

The adopted reconstruction will load Stage 1 best.pt and fine-tune on a combined
five-class dataset containing diseases, Healthy and negatives. That dataset
needs a fixed Healthy split and explicit label-ID remapping. This package
implements Stage 1 only; Stage 2 must not train the Healthy-only YAML as if it
automatically adds a fifth class.

For the planned experiment, both Stage 2 variants start from the same Stage 1
checkpoint and differ only by the declared added illumination policy.

## Known limits and verification

No full training was run while preparing this package: the preparation runtime
has no PyTorch/Ultralytics training environment. See VERIFICATION.md for the
checks actually run. The author-supplied datasets and pretrained weights are
not bundled.

Do not redraw author masks. Zero-area polygons are reported, not rewritten;
Ultralytics may apply its own loader corrections. Preserve the source ZIPs and
record any such corrections.

lrf=0.01 is a final learning-rate multiplier, not an absolute final rate of 0.01.
No custom class weighting, frozen layers, or historical Stage 2 schedule is
invented in this implementation.

Sources: AgriEye manuscript, Appendix K pp. 158-160 and Figure 15 pp. 68-69;
author-supplied disease V6 and Healthy V1 archive metadata.

