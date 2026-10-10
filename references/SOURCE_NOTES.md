# Sources and adaptations

## Manuscript extraction

User-supplied `AgriEye_Manuscript(1).pdf`, 193 pages; SHA256 `5d772ae210c99c6c39a507c99350dc037f9d955e619ee836eb2a1ab4191f60b2`. The uploaded copy matches the manuscript used in the preceding audit.

- Printed pp.47–48: preprocessing and dataset split descriptions.
- Printed pp.66–67 / PDF pp.83–84: leaf training recipe.
- Printed pp.68–69 / PDF pp.85–86: disease recipe and explanation.
- Printed pp.74–75 / PDF pp.91–92: Table 12, model versions, final v8 selection.
- Printed pp.158–161 / PDF pp.175–178: environment setup, dataset extraction, training, evaluation, and TFLite export screenshots.

`AgriEye_training_excerpt.pdf` preserves the latter ten pages with their printed page numbers. `manuscript_training_extracted.py` manually transcribes the substantive calls and settings. It does not claim that the screenshots are a complete final-run notebook.

## Operational corrections and assumptions

- Replace the author's Windows/Kaggle paths with audited local paths.
- Use the official lowercase `yolov8n-seg.pt` filename. The screenshot spells `YOLOv8n-seg.pt`.
- Obtain checkpoint locations from `model.trainer`. The pinned release returns a metrics object from `model.train`; the screenshot's `train_result.save_dir` is not used.
- Evaluate the current run's recorded `best.pt`, rather than the appendix's unrelated hardcoded Stage2 path.
- Use `ap_class_index` to associate per-class results with labels. A fixed `zip(CLASS_NAMES, ap50)` can mislabel compact arrays when classes are missing.
- Save box and mask metrics separately. The appendix explicitly prints `m.box` values; Table 12 does not establish their namespace or split unambiguously.
- Preserve `lrf=0.01` as a multiplicative final LR factor. With `lr0=0.0001`, the nominal endpoint is `0.000001`, not an absolute learning rate of 0.01.
- Change output directories/run names and disable overwriting. Record seed 0 explicitly as an assumption, not a recovered author seed.
- Evaluation at 640, confidence 0.001, NMS IoU 0.7, max detections 300, FP32, and no test-time augmentation makes unspecified framework behavior explicit. Evaluation batch 4/workers 2 are resource choices. The paper does not fully specify these arguments.
- Keep early stopping and checkpoint selection from the pinned implementation. Segmentation fitness in 8.4.41 combines box and mask mAP50–95.
- Preserve raw data by copying it into a working snapshot before framework caches or verification routines can write to it.

These adaptations make the recipe executable and observable. They do not establish recovery of the exact training history, dataset, random state, or environment that produced the published result.

## Primary software sources checked

- [Ultralytics 8.4.41 package metadata](https://pypi.org/pypi/ultralytics/8.4.41/json): version availability and dependencies verified by direct retrieval.
- [Versioned default configuration](https://github.com/ultralytics/ultralytics/blob/v8.4.41/ultralytics/cfg/default.yaml).
- [Versioned trainer](https://github.com/ultralytics/ultralytics/blob/v8.4.41/ultralytics/engine/trainer.py): callbacks, optimizer stepping, gradient accumulation, OOM behavior, checkpoint saving, and final validation inspected.
- [Versioned model API](https://github.com/ultralytics/ultralytics/blob/v8.4.41/ultralytics/engine/model.py): training and validation return values inspected.
- [Versioned metrics](https://github.com/ultralytics/ultralytics/blob/v8.4.41/ultralytics/utils/metrics.py): separate box/mask metrics, AP class indexing, and fitness inspected.
- [Versioned augmentation source](https://github.com/ultralytics/ultralytics/blob/v8.4.41/ultralytics/data/augment.py): optional default Albumentations behavior inspected.
- [Official segmentation documentation](https://docs.ultralytics.com/tasks/segment/): general workflow. Current examples use newer architectures; this project retains YOLOv8n-Seg.
- [AgriEye source snapshot](https://github.com/Onins123/AgriEye/tree/75dd443153249760857938e38320b6043cb4d3f2): released app/model evidence from the preceding audit.

The Ultralytics package was subsequently installed and executed in an isolated CPU verification environment. See `VERIFICATION.md` for its precise limits. No claim is made that merely reading upstream source proves compatibility with the user's laptop.
