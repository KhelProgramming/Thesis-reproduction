#!/usr/bin/env python3
"""Read-only audit for a Roboflow YOLO segmentation export.

Requires Pillow and PyYAML. Run: python agrieye_dataset_audit.py /path/to/export
Creates a new timestamped report folder in the working directory. Does not
modify the dataset, train a model, or certify diagnoses/annotation completeness.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Install dependencies: python -m pip install Pillow PyYAML") from exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", help="Folder containing data.yaml, train, valid, test")
    parser.add_argument("--output", help="New, nonexistent report directory")
    args = parser.parse_args()
    raw_path = args.dataset or input("Dataset folder (paste the complete path): ").strip()
    root = Path(raw_path.strip('\"\'')).expanduser().resolve()
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        raise SystemExit(f"Cannot find: {yaml_path}")
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit("data.yaml must contain a mapping.")
    names = cfg.get("names")
    if isinstance(names, dict):
        try:
            indexed = {int(k): str(v) for k, v in names.items()}
            if sorted(indexed) != list(range(len(indexed))):
                raise ValueError("Class IDs must be contiguous from zero")
            names = [indexed[i] for i in range(len(indexed))]
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"Invalid class-name mapping: {exc}") from exc
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise SystemExit("Expected a nonempty names list or indexed dictionary in data.yaml.")
    if "nc" in cfg and cfg["nc"] != len(names):
        raise SystemExit("nc disagrees with the number of class names.")

    # Inspect the actual export folders; do not assume ../ YAML paths resolve
    # identically in every training environment. Paths are recorded for review.
    split_dirs = {"train": root / "train", "val": root / "valid", "test": root / "test"}
    rows, issues, overlaps = [], [], []
    stats = {}
    hashes, families = defaultdict(list), defaultdict(list)

    def issue(severity, split, relative_path, kind, detail):
        issues.append(dict(severity=severity, split=split, path=relative_path,
                           kind=kind, detail=detail))

    for split, folder in split_dirs.items():
        images_dir, labels_dir = folder / "images", folder / "labels"
        stat = dict(images=0, readable_images=0, missing_labels=0, empty_labels=0,
                    polygon_instances=0, images_per_class=[0] * len(names),
                    instances_per_class=[0] * len(names))
        stats[split] = stat
        if not images_dir.is_dir() or not labels_dir.is_dir():
            issue("error", split, str(folder.relative_to(root)), "missing_directory",
                  "Expected images/ and labels/ directories")
            continue
        images = sorted(p for p in images_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        expected_labels = set()
        for number, image_path in enumerate(images, 1):
            relative_image = str(image_path.relative_to(root))
            label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
            relative_label = str(label_path.relative_to(root))
            if label_path in expected_labels:
                issue("error", split, relative_image, "shared_label_path",
                      "Multiple image extensions share one label filename")
            expected_labels.add(label_path)
            stat["images"] += 1
            width = height = 0
            digest = ""
            try:
                data = image_path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    width, height = image.size
                stat["readable_images"] += 1
            except Exception as exc:
                issue("error", split, relative_image, "unreadable_image", str(exc))
            item = dict(split=split, path=relative_image)
            if digest:
                hashes[digest].append(item)
            stem = image_path.stem
            # A filename heuristic, not proof of original leaf/plant identity.
            parent = re.split(r"\.rf\.", stem, maxsplit=1, flags=re.IGNORECASE)[0]
            families[parent.casefold()].append(item)
            classes, polygon_count = set(), 0
            if not label_path.is_file():
                stat["missing_labels"] += 1
                issue("review", split, relative_image, "missing_label",
                      "Confirm intended background image versus missing annotation")
            else:
                text = label_path.read_text(encoding="utf-8-sig", errors="replace")
                lines = [(i, s.strip()) for i, s in enumerate(text.splitlines(), 1) if s.strip()]
                if not lines:
                    stat["empty_labels"] += 1
                for line_number, line in lines:
                    tokens = line.split()
                    location = f"line {line_number}"
                    if len(tokens) == 5:
                        issue("error", split, relative_label, "box_format",
                              f"{location}: five fields resemble a detection box, not a polygon")
                        continue
                    if len(tokens) < 7 or (len(tokens) - 1) % 2:
                        issue("error", split, relative_label, "polygon_format",
                              f"{location}: need class ID followed by at least three x,y pairs")
                        continue
                    try:
                        if not re.fullmatch(r"\d+", tokens[0]):
                            raise ValueError("class ID must be a nonnegative integer")
                        class_id = int(tokens[0])
                        xy = [float(v) for v in tokens[1:]]
                        if not 0 <= class_id < len(names):
                            raise ValueError(f"class ID {class_id} is not in data.yaml")
                        if not all(math.isfinite(v) and 0 <= v <= 1 for v in xy):
                            raise ValueError("coordinates must be finite and within [0,1]")
                        points = list(zip(xy[::2], xy[1::2]))
                        if len(set(points)) < 3:
                            raise ValueError("fewer than three distinct vertices")
                        area = abs(sum(points[i][0] * points[(i+1) % len(points)][1]
                                       - points[(i+1) % len(points)][0] * points[i][1]
                                       for i in range(len(points)))) / 2
                        if area <= 1e-12:
                            raise ValueError("zero polygon area")
                    except (ValueError, OverflowError) as exc:
                        issue("error", split, relative_label, "invalid_annotation",
                              f"{location}: {exc}")
                        continue
                    classes.add(class_id)
                    polygon_count += 1
                    stat["instances_per_class"][class_id] += 1
                for class_id in classes:
                    stat["images_per_class"][class_id] += 1
            stat["polygon_instances"] += polygon_count
            rows.append(dict(split=split, image=relative_image, label=relative_label,
                             width=width, height=height, sha256=digest,
                             parent_name_hint=parent, valid_polygon_count=polygon_count,
                             class_ids=";".join(map(str, sorted(classes)))))
            if number % 500 == 0:
                print(f"{split}: inspected {number}/{len(images)} images", flush=True)
        for label_path in sorted(labels_dir.rglob("*.txt")):
            if label_path not in expected_labels:
                issue("review", split, str(label_path.relative_to(root)), "orphan_label",
                      "No matching supported image file")
        if not images:
            issue("error", split, str(folder.relative_to(root)), "empty_split", "No images found")
        for class_id, count in enumerate(stat["images_per_class"]):
            if count == 0:
                issue("review", split, "", "class_absent", f"No valid polygons for {names[class_id]}")

    for kind, groups in [("exact_image_bytes", hashes), ("parent_filename_candidate", families)]:
        for key, items in groups.items():
            splits = sorted({item["split"] for item in items})
            if len(splits) > 1:
                overlaps.append(dict(kind=kind, key=key, splits=";".join(splits),
                                     images=len(items), paths=" | ".join(i["path"] for i in items)))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path(args.output).expanduser().resolve() if args.output else Path.cwd() / f"agrieye_audit_{timestamp}"
    destination.mkdir(parents=True, exist_ok=False)
    status_counts = Counter(i["severity"] for i in issues)
    overlap_counts = Counter(o["kind"] for o in overlaps)
    summary = dict(audited_at_utc=timestamp, dataset_root=str(root), names=names,
                   declared_paths={k: cfg.get(k) for k in ["train", "val", "test"]},
                   inspected_paths={k: str(v) for k, v in split_dirs.items()},
                   roboflow=cfg.get("roboflow", {}), splits=stats,
                   issues_by_severity=dict(status_counts),
                   cross_split_overlap_groups=dict(overlap_counts),
                   review_required=bool(issues or overlaps), training_executed=False,
                   limitations=["Filename groups are candidates, not confirmed leaf/plant identities.",
                                "Exact hashes miss resized, re-encoded, cropped, and other near duplicates.",
                                "No diagnosis, annotation completeness, or polygon self-intersection verification.",
                                "An empty label can be a valid negative image; it does not establish a Healthy class.",
                                "No capture-light, farm, date, or session metadata verified."])
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(destination / "manifest.csv", rows,
              ["split", "image", "label", "width", "height", "sha256", "parent_name_hint", "valid_polygon_count", "class_ids"])
    write_csv(destination / "issues.csv", issues, ["severity", "split", "path", "kind", "detail"])
    write_csv(destination / "cross_split_overlap.csv", overlaps, ["kind", "key", "splits", "images", "paths"])
    print("\nCLASS ORDER:", dict(enumerate(names)))
    print(f"{'Split':<8}{'Images':>10}{'Polygons':>12}{'Missing labels':>17}{'Empty labels':>15}")
    for split, stat in stats.items():
        print(f"{split:<8}{stat['images']:>10}{stat['polygon_instances']:>12}{stat['missing_labels']:>17}{stat['empty_labels']:>15}")
    print("\nIMAGES WITH AT LEAST ONE VALID POLYGON PER CLASS (not mutually exclusive)")
    print(f"{'Class':<20}{'Train':>10}{'Val':>10}{'Test':>10}")
    for class_id, name in enumerate(names):
        print(f"{name:<20}" + "".join(f"{stats[s]['images_per_class'][class_id]:>10}" for s in split_dirs))
    print("\nAnnotation/file issues:", dict(status_counts))
    print("Cross-split overlap groups:", dict(overlap_counts))
    print("Outcome:", "REVIEW REQUIRED" if summary["review_required"] else "No issues found by these limited checks")
    print("Reports:", destination)
    print("Return summary.json, issues.csv, and cross_split_overlap.csv for interpretation.")
    print("No dataset files were changed. No model was trained.")


if __name__ == "__main__":
    main()
