#!/usr/bin/env python3
"""Copy Stage 1 data, excluding known exact duplicate tiles from held-out splits.

Original data is never modified. Requires the audit and duplicate inspection
reports from the preceding commands. Matching tiles must have empty labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import yaml

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(audit_file, inspection_file, output_dir, manifest_file):
    audit_file = Path(audit_file).resolve()
    inspection_file = Path(inspection_file).resolve()
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    inspection = json.loads(inspection_file.read_text(encoding="utf-8"))
    names = audit["names"]
    if len(names) != 4 or set(names) != {"BLB", "BLS", "LEAF BLAST", "NBS"}:
        raise ValueError("Expected the four target Stage 1 classes.")
    if not inspection.get("groups"):
        raise ValueError("No duplicate groups were supplied; no cleaning is needed.")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError(f"Destination already exists; choose a new --output-dir: {destination}")
    split_dirs = {s: Path(v["image_dir"]).resolve() for s,v in audit["splits"].items()}
    label_dirs = {s: Path(v["label_dir"]).resolve() for s,v in audit["splits"].items()}
    for folder in list(split_dirs.values()) + list(label_dirs.values()):
        if not folder.is_dir():
            raise ValueError(f"Source folder not found: {folder}")
        if destination == folder or folder in destination.parents or destination in folder.parents:
            raise ValueError("Working-copy destination must not overlap the source split folders.")
    exclusions = {}
    for group in inspection["groups"]:
        members = group["members"]
        present = {m["split"] for m in members}
        if len(present) < 2:
            raise ValueError("Inspection contains a group that does not span splits.")
        # Preserve training; for validation/test-only groups, keep test.
        keep_split = next(s for s in ("train", "test", "val") if s in present)
        for member in members:
            split = member["split"]
            image = Path(member["image"]).resolve()
            label = Path(member["label"]).resolve()
            if image.parent != split_dirs[split] or label.parent != label_dirs[split]:
                raise ValueError("Inspection paths do not agree with the dataset audit.")
            if not image.is_file() or not label.is_file():
                raise ValueError(f"Inspection member is missing: {image}")
            if sha256(image) != group["sha256"]:
                raise ValueError(f"Image changed since inspection: {image}")
            if label.read_text(encoding="utf-8").strip():
                raise ValueError(
                    f"Annotated duplicate needs separate review; it will not be removed: {label}"
                )
            if split != keep_split:
                exclusions[(split, image.name)] = {
                    "split": split, "image": image.name, "label": label.name,
                    "sha256": group["sha256"], "classification": group["classification"],
                    "retained_split": keep_split, "reason": "exact image bytes overlap another retained split",
                }
    if not exclusions:
        raise ValueError("The inspection produced no held-out exclusions.")
    # Validate pairing before writing the working copy.
    image_lists = {}
    for split, image_dir in split_dirs.items():
        images = sorted(p for p in image_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
        image_lists[split] = images
        if len(images) != audit["splits"][split]["images"]:
            raise ValueError(f"{split}: source image count changed since audit.")
        for image in images:
            if not (label_dirs[split] / (image.stem + ".txt")).is_file():
                raise ValueError(f"Missing label for {image}")
    mapping = {"train": "train", "val": "valid", "test": "test"}
    counts, empty = Counter(), Counter()
    destination.mkdir(parents=True)
    for split, images in image_lists.items():
        target_images = destination / mapping[split] / "images"
        target_labels = destination / mapping[split] / "labels"
        target_images.mkdir(parents=True)
        target_labels.mkdir(parents=True)
        for image in images:
            if (split, image.name) in exclusions:
                continue
            label = label_dirs[split] / (image.stem + ".txt")
            shutil.copy2(image, target_images / image.name)
            shutil.copy2(label, target_labels / label.name)
            counts[split] += 1
            empty[split] += int(not label.read_text(encoding="utf-8").strip())
    cfg = {"train": "train/images", "val": "valid/images", "test": "test/images",
           "nc": 4, "names": names}
    (destination / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest = {
        "source_audit": str(audit_file), "source_inspection": str(inspection_file),
        "output_dir": str(destination), "source_files_changed": False,
        "policy": "Keep training unchanged; omit exact matching empty-label tiles from other splits.",
        "exclusions": sorted(exclusions.values(), key=lambda x: (x["split"],x["image"])),
        "excluded_counts": dict(Counter(e["split"] for e in exclusions.values())),
        "output_image_counts": dict(counts), "output_empty_label_counts": dict(empty),
        "output_total_images": sum(counts.values()),
        "scope": "Known exact tile overlap only; near-duplicates and source-family independence are not established.",
    }
    manifest_file = Path(manifest_file)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "exclusion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Working copy created:", destination)
    print("Excluded:", manifest["excluded_counts"])
    print("Remaining images:", dict(counts), "total:", sum(counts.values()))
    print("Original dataset unchanged. Exclusion manifest:", manifest_file)
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="reports/stage1_dataset_audit.json")
    parser.add_argument("--inspection", default="reports/stage1_duplicate_inspection.json")
    parser.add_argument("--output-dir", default="datasets/stage1_disease_clean")
    parser.add_argument("--manifest", default="reports/stage1_clean_copy_manifest.json")
    args = parser.parse_args()
    try:
        prepare(args.audit, args.inspection, args.output_dir, args.manifest)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Preparation error: {exc}\n")

if __name__ == "__main__":
    main()

