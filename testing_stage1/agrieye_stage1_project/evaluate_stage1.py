#!/usr/bin/env python3
"""Evaluate Stage 1 with Ultralytics, keeping box and mask metric names intact."""
import argparse
from pathlib import Path
from dataset_tools import write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="Use reports/stage1_data_resolved.yaml")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--output", default="reports/stage1_test_metrics.json")
    a = p.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    metrics = model.val(data=str(Path(a.data).resolve()), split=a.split,
                        imgsz=640, batch=a.batch, device=a.device, plots=True)
    payload = {
        "weights": str(Path(a.weights).resolve()), "data": str(Path(a.data).resolve()),
        "split": a.split, "names": model.names,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
    }
    write_json(a.output, payload)
    print(payload)
    print("Saved:", a.output)

if __name__ == "__main__":
    main()

