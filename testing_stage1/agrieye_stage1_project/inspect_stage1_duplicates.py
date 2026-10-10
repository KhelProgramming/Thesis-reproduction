#!/usr/bin/env python3
"""Inspect exact duplicate tiles across splits without changing the dataset.

Run from agrieye_stage1_project:
    python inspect_stage1_duplicates.py
Uses reports/stage1_dataset_audit.json written by train_stage1.py --dry-run.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageStat


def inspect(audit_path, output_path):
    audit_path = Path(audit_path).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    hashes = defaultdict(list)
    for split, entry in audit["splits"].items():
        images = Path(entry["image_dir"])
        labels = Path(entry["label_dir"])
        if not images.is_dir():
            raise ValueError(f"Missing image directory: {images}")
        for path in sorted(images.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                continue
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[digest].append((split, path, labels / (path.stem + ".txt")))
    groups = []
    for digest, records in hashes.items():
        if len({r[0] for r in records}) < 2:
            continue
        members = []
        for split, image_path, label_path in records:
            label_text = label_path.read_text(encoding="utf-8").strip() if label_path.is_file() else None
            rows = label_text.splitlines() if label_text else []
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                extrema = rgb.getextrema()
                stat = ImageStat.Stat(rgb)
                uniform = all(low == high for low, high in extrema)
                size = list(image.size)
            members.append({
                "split": split,
                "image": str(image_path),
                "label": str(label_path),
                "label_exists": label_text is not None,
                "label_empty": label_text == "",
                "classes": sorted({int(row.split()[0]) for row in rows}),
                "label_rows": len(rows),
                "label_sha256": hashlib.sha256(label_text.encode()).hexdigest() if label_text is not None else None,
                "size": size,
                "uniform_pixels": uniform,
                "rgb_extrema": [list(pair) for pair in extrema],
                "rgb_mean": [round(x, 3) for x in stat.mean],
                "rgb_stddev": [round(x, 3) for x in stat.stddev],
            })
        if all(m["label_empty"] and m["uniform_pixels"] for m in members):
            kind = "uniform_negative_tiles"
        elif all(m["label_empty"] for m in members):
            kind = "nonuniform_negative_images"
        else:
            kind = "duplicates_with_target_annotations"
        groups.append({"sha256": digest, "classification": kind, "members": members})
    groups.sort(key=lambda g: (g["classification"], g["sha256"]))
    report = {
        "audit": str(audit_path),
        "cross_split_duplicate_groups": len(groups),
        "images_in_groups": sum(len(g["members"]) for g in groups),
        "classification_counts": {
            kind: sum(g["classification"] == kind for g in groups)
            for kind in ("uniform_negative_tiles", "nonuniform_negative_images", "duplicates_with_target_annotations")
        },
        "groups": groups,
        "interpretation": (
            "Uniform empty-label groups may be identical blank tiles from unrelated originals; "
            "their source origin is not established by this check. Nonuniform duplicate images "
            "or duplicates with annotations require source/split investigation. Do not bypass "
            "the original audit solely because labels are empty."
        ),
        "source_files_changed": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Cross-split duplicate groups: {len(groups)}")
    print("Group classifications:", json.dumps(report["classification_counts"]))
    for i, group in enumerate(groups, 1):
        print(f"\nGroup {i}: {group['classification']}")
        for member in group["members"]:
            print(
                f"  {member['split']}/{Path(member['image']).name}"
                f" | empty_label={member['label_empty']}"
                f" | classes={member['classes']}"
                f" | uniform_pixels={member['uniform_pixels']}"
                f" | RGB_stddev={member['rgb_stddev']}"
            )
    print("\nRead-only inspection saved:", output_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="reports/stage1_dataset_audit.json")
    parser.add_argument("--output", default="reports/stage1_duplicate_inspection.json")
    args = parser.parse_args()
    try:
        inspect(args.audit, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Inspection error: {exc}\n")


if __name__ == "__main__":
    main()

