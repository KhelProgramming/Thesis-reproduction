# Verification

Prepared on 10 October 2026.

Passed:
- Full command-line dry run with a valid four-class segmentation fixture.
- Resolution of Roboflow ../train/images paths to actual export-local directories.
- Retention and counting of empty-label negative images.
- Rejection of missing labels.
- Rejection of detection-box labels used where segmentation polygons are required.
- Rejection of nonfinite normalized coordinates.
- Rejection of datasets with no empty-label training examples.
- Rejection of byte-identical images across train/test.
- Rejection of the raw author export's five-class LEAF configuration.
- Full read-only audit on a temporary copy of the author data with LEAF excluded
  and the remaining class IDs remapped: 1,145 images, 211 empty labels, splits
  768 / 188 / 189; no validation errors.

The temporary LEAF-excluded raw copy was only a checker test. It is not bundled
as a training dataset and does not replace Roboflow preprocessing/augmentation.
The author source archives were not modified.

Not verified:
- Actual Ultralytics installation/runtime compatibility on your GPU.
- Pretrained model download.
- A YOLO training epoch or final evaluation.
- The generated Roboflow export, which is not yet supplied.
- Independence of near-duplicate/source-family images across splits.
- The authors' exact historical training stages, package version or epoch schedule.

The preparation runtime has no torch/ultralytics. A 1-epoch smoke run in the
project README is the required next runtime check before a full training run.

