#!/usr/bin/env python3
"""Read-only visual review of an existing AgriEye audit. No training or relabeling.

Run from agrieye_reproduction after the earlier prepare command:
    python review_agrieye_samples.py --dataset data/raw/agrieye_export
Requires Pillow (already installed by the baseline project).
"""
import argparse
import csv
import hashlib
import json
import math
import random
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageDraw, ImageFont, ImageOps


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path is outside the dataset: {relative}")
    return path


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def choose_audit(root, reports, explicit):
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = sorted(reports.glob("audit_*"), reverse=True)
    for directory in candidates:
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if Path(summary["dataset_root"]).resolve() == root:
            return directory, summary
    raise ValueError("No audit for this dataset path was found. Run run.py prepare first, or pass --audit PATH.")


def choose_groups(manifest, overlaps, seed):
    # One empty training example per filename parent, with a fixed shuffle.
    # Parent names are a sampling heuristic, not verified biological identities.
    empty = sorted((r for r in manifest if r["split"] == "train" and
                    int(r["valid_polygon_count"]) == 0), key=lambda r: r["image"])
    random.Random(seed).shuffle(empty)
    chosen, parents = [], set()
    for row in empty:
        parent = row["parent_name_hint"].casefold()
        if parent not in parents:
            chosen.append(row["image"])
            parents.add(parent)
        if len(chosen) == 12:
            break
    groups = [("empty", f"Empty training labels {i + 1}-{min(i + 4, len(chosen))}",
               chosen[i:i + 4]) for i in range(0, len(chosen), 4)]
    for kind, category, limit, per_split in [
        ("exact_image_bytes", "exact", None, 1),
        ("parent_filename_candidate", "family", 4, 2),
    ]:
        candidates = sorted((r for r in overlaps if r["kind"] == kind), key=lambda r: r["key"])
        if limit is not None:
            # Review larger groups first; this is a diagnostic sample, not random prevalence estimation.
            candidates = sorted(candidates, key=lambda r: (-int(r["images"]), r["key"]))[:limit]
        for number, group in enumerate(candidates, 1):
            by_split = defaultdict(list)
            for path in sorted(group["paths"].split(" | ")):
                by_split[path.split("/", 1)[0]].append(path)
            selected = [p for split in ("train", "valid", "val", "test")
                        for p in by_split[split][:per_split]]
            label = f"{category} group {number} | {group['images']} total files | {group['splits']}"
            groups.append((category, label, selected))
    return groups


def make_card(image_path, label_path, sample_id, row):
    # Outlines and resized previews are separate review products. Source bytes stay unchanged.
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise ValueError(f"Invalid polygon while making preview: {label_path}")
        coordinates = list(map(float, parts[1:]))
        points = [(coordinates[i] * image.width, coordinates[i + 1] * image.height)
                  for i in range(0, len(coordinates), 2)]
        draw.line(points + [points[0]], fill=(255, 40, 40), width=max(2, image.width // 250))
    card = Image.new("RGB", (320, 350), "white")
    preview = ImageOps.contain(image, (310, 260), Image.Resampling.LANCZOS)
    card.paste(preview, ((320 - preview.width) // 2, 5 + (260 - preview.height) // 2))
    draw = ImageDraw.Draw(card)
    title = f"{sample_id} | {row['split']} | polygons={row['valid_polygon_count']}"
    draw.text((6, 269), title, font=font(13), fill="black")
    short_name = Path(row["image"]).name
    for j, line in enumerate(textwrap.wrap(short_name, 41)[:3]):
        draw.text((6, 289 + j * 17), line, font=font(11), fill="black")
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/raw/agrieye_export")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--audit", help="Optional exact audit directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.dataset).expanduser().resolve()
    reports = Path(args.reports).expanduser().resolve()
    audit, summary = choose_audit(root, reports, args.audit)
    manifest = read_csv(audit / "manifest.csv")
    overlaps = read_csv(audit / "cross_split_overlap.csv")
    lookup = {r["image"]: r for r in manifest}
    groups = choose_groups(manifest, overlaps, args.seed)
    selected = sorted({p for _, _, paths in groups for p in paths})
    if not selected:
        raise ValueError("There are no matching examples to export.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = reports / f"visual_review_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    records, cards = [], {}
    for number, relative in enumerate(selected, 1):
        row = lookup[relative]
        image_path, label_path = contained(root, relative), contained(root, row["label"])
        image_bytes, label_bytes = image_path.read_bytes(), label_path.read_bytes()
        if digest(image_bytes) != row["sha256"]:
            raise ValueError(f"Image changed since audit; no training was run: {relative}")
        lines = [s for s in label_bytes.decode("utf-8-sig").splitlines() if s.strip()]
        if len(lines) != int(row["valid_polygon_count"]):
            raise ValueError(f"Label count changed since audit: {row['label']}")
        sample_id = f"S{number:03d}"
        for relative_file, content in [(relative, image_bytes), (row["label"], label_bytes)]:
            destination = output / "selected_files" / relative_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        reasons = "; ".join(label for _, label, paths in groups if relative in paths)
        records.append(dict(sample_id=sample_id, image=relative, label=row["label"],
                            split=row["split"], valid_polygon_count=row["valid_polygon_count"],
                            class_ids=row["class_ids"], reason=reasons,
                            image_sha256=digest(image_bytes), label_sha256=digest(label_bytes)))
        cards[relative] = make_card(image_path, label_path, sample_id, row)
    for category in ("empty", "exact", "family"):
        rows = [g for g in groups if g[0] == category]
        for start in range(0, len(rows), 5):
            batch = rows[start:start + 5]
            columns = max(len(g[2]) for g in batch)
            sheet = Image.new("RGB", (max(1280, 320 * columns), 55 + 380 * len(batch)), "#eeeeee")
            draw = ImageDraw.Draw(sheet)
            draw.text((12, 12), f"{category.upper()} REVIEW | red outlines = supplied annotations | no predictions",
                      font=font(19), fill="black")
            for row_index, (_, label, paths) in enumerate(batch):
                top = 55 + row_index * 380
                draw.text((10, top), label, font=font(14), fill="black")
                for column, relative in enumerate(paths):
                    sheet.paste(cards[relative], (column * 320, top + 25))
            sheet.save(output / f"{category}_{start // 5 + 1:02d}.jpg", quality=90)
    with (output / "selected_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    review = dict(audit_directory=str(audit), source_dataset=str(root), seed=args.seed,
                  selected_images=len(selected), class_names=summary["names"],
                  selection_groups=[dict(category=c, reason=r, paths=p) for c, r, p in groups],
                  raw_dataset_modified=False, model_training_executed=False,
                  model_test_evaluation_executed=False,
                  note="Test images, if selected, are only inspected for split and annotation provenance. No model predictions are generated.")
    (output / "review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        "VISUAL DATASET REVIEW\n\n"
        "empty: 12 training examples with no polygons, sampled with seed 0 by default, one per filename parent.\n"
        "exact: one example per split from each exact-image overlap group.\n"
        "family: up to two examples per split from four large filename candidate groups.\n"
        "Red outlines show the supplied polygons. There are no model predictions.\n"
        "Full-resolution unmodified selected images and label files are in selected_files/.\n"
        "Check: does an empty example lack a target necrotic lesion? Do identical images have different masks?\n"
        "Check: are filename candidates crops/augmentations of one original, or unrelated reused names?\n"
        "This sample cannot certify all annotations or estimate an error prevalence. Seek expert review for uncertain symptoms.\n"
        "Sources are read only. No images are removed, relabeled, split, augmented for training, or trained on.\n"
        "The ZIP contains selected dataset images; share only if your dataset permissions allow it.\n",
        encoding="utf-8")
    # Verify selected source files still match the exact bytes we copied.
    for row in records:
        for key, hash_key in (("image", "image_sha256"), ("label", "label_sha256")):
            if digest(contained(root, row[key]).read_bytes()) != row[hash_key]:
                raise RuntimeError(f"Source changed during review: {row[key]}")
    archive = reports / f"visual_review_to_share_{stamp}.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(output))
    print(f"Audit used: {audit}")
    print(f"Selected images: {len(selected)}")
    print(f"Preview folder: {output}")
    print(f"Review ZIP (includes selected images and labels): {archive}")
    print("Original images and labels unchanged. No model was trained or tested.")


if __name__ == "__main__":
    main()
