#!/usr/bin/env python3
"""Train the four-disease + negative Stage 1 model, or audit with --dry-run."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import yaml

from dataset_tools import audit_dataset, read_yaml, write_json

ROOT = Path(__file__).resolve().parent
EXPECTED_ULTRALYTICS = "8.3.79"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Generated four-class YOLOv8 segmentation data.yaml")
    parser.add_argument("--config", default=str(ROOT / "configs/stage1.yaml"))
    parser.add_argument("--weights", default="yolov8n-seg.pt")
    parser.add_argument("--output-dir", default=str(ROOT / "runs"))
    parser.add_argument("--name", default="stage1_four_disease")
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    parser.add_argument("--dry-run", action="store_true", help="Check data and prepare config without importing YOLO")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", help="GPU index, e.g. 0, or cpu")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        summary, resolved = audit_dataset(args.data)
        cfg = read_yaml(args.config)
        cfg.update(read_yaml(ROOT / "configs/reproducibility.yaml"))
        for key in ("epochs", "batch", "workers", "seed", "device"):
            value = getattr(args, key)
            if value is not None:
                cfg[key] = value
        if int(cfg["epochs"]) < 1 or int(cfg["batch"]) < 1 or int(cfg["workers"]) < 0:
            raise ValueError("epochs/batch must be positive; workers must be nonnegative.")
        reports = Path(args.report_dir).expanduser().resolve()
        reports.mkdir(parents=True, exist_ok=True)
        write_json(reports / "stage1_dataset_audit.json", summary)
        print("Dataset audit")
        print(json.dumps({key: summary[key] for key in
                          ("names", "splits", "total_images", "total_empty_labels", "notices")},
                         indent=2))
        if summary["errors"]:
            for error in summary["errors"][:20]:
                print("ERROR:", error, file=sys.stderr)
            print(f"Audit failed with {len(summary['errors'])} issue(s).", file=sys.stderr)
            return 2
        resolved_yaml = reports / "stage1_data_resolved.yaml"
        resolved_yaml.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        cfg.update(data=str(resolved_yaml), project=str(Path(args.output_dir).expanduser().resolve()),
                   name=args.name)
        plan = {
            "stage": 1, "task": "segment", "weights": args.weights,
            "training_config": cfg,
            "expected_ultralytics": EXPECTED_ULTRALYTICS,
            "protocol": "AgriEye-inspired reconstruction; historical two-stage details unconfirmed",
            "dataset_audit": str(reports / "stage1_dataset_audit.json"),
        }
        write_json(reports / "stage1_training_plan.json", plan)
        print("Training plan:", reports / "stage1_training_plan.json")
        if args.dry_run:
            print("DRY RUN PASSED. Source images/labels were not changed. No training or weights download occurred.")
            return 0
        try:
            from ultralytics import YOLO
            import torch
        except ImportError as exc:
            print(f"Training dependency missing: {exc}. Install requirements.txt first.", file=sys.stderr)
            return 2
        installed = importlib.metadata.version("ultralytics")
        if installed != EXPECTED_ULTRALYTICS:
            print(f"Expected ultralytics {EXPECTED_ULTRALYTICS}, found {installed}. "
                  "Use the project's requirements to avoid changing behavior between runs.", file=sys.stderr)
            return 2
        if str(cfg.get("device")) != "cpu" and not torch.cuda.is_available():
            print("CUDA is unavailable. Select a GPU runtime, or explicitly use --device cpu "
                  "for a small smoke run.", file=sys.stderr)
            return 2
        provenance = {
            **plan, "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("ultralytics", "torch", "torchvision", "PyYAML")},
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        }
        write_json(reports / "stage1_training_environment.json", provenance)
        # Reproduces the appendix's YOLO initialization and model.train call.
        model = YOLO(args.weights)
        result = model.train(**cfg)
        save_dir = Path(result.save_dir)
        provenance["save_dir"] = str(save_dir.resolve())
        provenance["best_weights"] = str((save_dir / "weights/best.pt").resolve())
        provenance["last_weights"] = str((save_dir / "weights/last.pt").resolve())
        provenance["results_csv"] = str((save_dir / "results.csv").resolve())
        write_json(reports / "stage1_training_environment.json", provenance)
        write_json(save_dir / "reconstruction_provenance.json", provenance)
        print("Training finished:", save_dir)
        print("Best checkpoint:", save_dir / "weights/best.pt")
        return 0
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())

