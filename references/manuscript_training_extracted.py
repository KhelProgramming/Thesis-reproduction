"""Manual transcription of training settings from the supplied manuscript.

Source SHA256: 5d772ae210c99c6c39a507c99350dc037f9d955e619ee836eb2a1ab4191f60b2
Printed pages 158-161 = PDF pages 175-178. Figure 13 is printed p.66.
This reference file does not execute training. Use run.py for the recorded setup.
Author-specific absolute paths are replaced with placeholders below.
The screenshot spells the checkpoint 'YOLOv8n-seg.pt'; the operational setup
uses the official lowercase filename 'yolov8n-seg.pt' on case-sensitive systems.
"""

PAPER_DATASET_ZIP_BASENAME = "AgriEye_No leaf with healthy.v2i.yolov8.zip"
PAPER_EVALUATION_CHECKPOINT_RUN = "yolov8n_FinalModel#5_Stage2"
PAPER_EVALUATION_CLASS_NAMES = ["BLB", "BLS", "HEALTHY", "LEAF BLAST", "NBS"]

# Source: bottom of printed p.159, continued on p.160, also Figure 15 p.68.
TRAIN_CONFIG = dict(
    data="agrieye_fixed.yaml",
    epochs=50,
    imgsz=640,
    batch=16,  # source comment: reduce to 8 if GPU out of memory
    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    patience=20,
    cos_lr=True,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.4,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    project="runs/agrieye",
    name="yolov8n_FinalModel#5 - Fine tune",
    exist_ok=True,
    save_period=10,
    plots=True,
    verbose=True,
    device=0,
    amp=True,
)

# Figure 13, printed p.66. These are a separate leaf-model recipe.
LEAF_TRAIN_CONFIG = dict(
    data="AUTHOR_LEAF_DATA_YAML",  # source has a Kaggle absolute path
    epochs=200,
    imgsz=640,
    batch=16,
    device="0,1",
    workers=2,
    project="AUTHOR_KAGGLE_RUNS_DIRECTORY",
    name="palay-seg-v2",
    patience=20,
    save=True,
    plots=True,
    val=True,
)

# Transcribed command sequence, paths symbolically replaced. Shown as a string
# to preserve the source operations without accidentally installing/training.
PAPER_COMMAND_SEQUENCE = r'''
# Environment setup, p.158:
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', 'pyyaml', '--quiet'])
import ultralytics
ultralytics.checks()

# Dataset extraction, p.158:
from pathlib import Path
import yaml, zipfile
DATASET_DIR = Path('datasets/agrieye')
FIXED_YAML = 'agrieye_fixed.yaml'
ZIP_PATH = 'AUTHOR_DATASET_ZIP'
DATASET_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(DATASET_DIR)

# Rewrite dataset paths, p.159:
YAML_PATH = DATASET_DIR / 'data.yaml'
ABS = str(DATASET_DIR.resolve())
with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)
cfg['train'] = f'{ABS}/train/images'
cfg['val'] = f'{ABS}/valid/images'
cfg['test'] = f'{ABS}/test/images'
with open(FIXED_YAML, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
for split in ['train', 'valid', 'test']:
    imgs = list((DATASET_DIR / split / 'images').glob('*.jpg'))
    lbls = list((DATASET_DIR / split / 'labels').glob('*.txt'))
    print(f'{split:6s}: {len(imgs)} images, {len(lbls)} labels')

# Training, pp.159-160 (TRAIN_CONFIG is transcribed above):
from ultralytics import YOLO
model = YOLO('YOLOv8n-seg.pt')
train_result = model.train(**TRAIN_CONFIG)
BEST_WEIGHTS = str(train_result.save_dir / 'weights' / 'best.pt')
RESULTS_CSV = str(train_result.save_dir / 'results.csv')

# Evaluation, p.160. Note the hardcoded five-class list and a Stage2 checkpoint:
CLASS_NAMES = ['BLB', 'BLS', 'HEALTHY', 'LEAF BLAST', 'NBS']
BEST_WEIGHTS = 'AUTHOR_STAGE2_BEST_PT'
model_eval = YOLO(BEST_WEIGHTS)
for split in ['val', 'test']:
    m = model_eval.val(data=FIXED_YAML, split=split, imgsz=640, verbose=False)
    print('mAP50:', m.box.map50)
    print('mAP50-95:', m.box.map)
    print('Precision:', m.box.mp)
    print('Recall:', m.box.mr)
    if split == 'val':
        for name, ap in zip(CLASS_NAMES, m.box.ap50):
            print(name, ap)

# Export, p.161:
model_eval.export(format='tflite', device=0)
'''

if __name__ == "__main__":
    print("Reference transcription only. Use ../run.py for recorded baseline runs.")
    print(TRAIN_CONFIG)
