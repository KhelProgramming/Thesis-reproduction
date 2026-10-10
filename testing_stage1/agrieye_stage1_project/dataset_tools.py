"""Read-only checks and portable YAML preparation for Stage 1."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import re
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ALIASES = {
    "BLB": "BLB", "BACTERIALLEAFBLIGHT": "BLB",
    "BLS": "BLS", "BACTERIALLEAFSTREAK": "BLS",
    "LEAFBLAST": "LEAF BLAST",
    "NBS": "NBS", "NARROWBROWNSPOT": "NBS",
}
TARGETS = {"BLB", "BLS", "LEAF BLAST", "NBS"}

def read_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return result

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

def class_names(cfg):
    raw = cfg.get("names")
    if isinstance(raw, dict):
        indexed = {int(k): v for k, v in raw.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("Class IDs must be contiguous starting at 0.")
        raw = [indexed[i] for i in range(len(indexed))]
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(
            "Stage 1 requires exactly four classes: BLB, BLS, LEAF BLAST, NBS. "
            "Exclude LEAF in the generated Roboflow version; keep Healthy separate."
        )
    canonical = []
    for name in raw:
        key = re.sub(r"[^A-Z0-9]", "", str(name).upper())
        if key not in ALIASES:
            raise ValueError(f"Unexpected Stage 1 class: {name!r}")
        canonical.append(ALIASES[key])
    if set(canonical) != TARGETS or len(set(canonical)) != 4:
        raise ValueError("The four target classes must each appear exactly once.")
    if "nc" in cfg and int(cfg["nc"]) != 4:
        raise ValueError("nc disagrees with the four-class names list.")
    return canonical

def resolve_image_dir(yaml_path, cfg, split, notices):
    entry = cfg.get(split)
    if not isinstance(entry, str):
        raise ValueError(f"{split}: this scaffold expects a split image directory.")
    yaml_root = yaml_path.parent
    raw_root = Path(str(cfg.get("path", "."))).expanduser()
    root = raw_root if raw_root.is_absolute() else yaml_root / raw_root
    requested = Path(entry).expanduser()
    primary = requested if requested.is_absolute() else root / requested
    if primary.is_dir():
        return primary.resolve()
    # Roboflow exports sometimes use ../train/images in a root-level data.yaml.
    # Use this fallback only when the requested relative path is absent.
    if not requested.is_absolute():
        alternatives = [yaml_root / split / "images"]
        if split == "val":
            alternatives = [yaml_root / "valid" / "images", yaml_root / "val" / "images"]
        for alternative in alternatives:
            if alternative.is_dir():
                notices.append(
                    f"{split}: exported path {entry!r} was absent; using {alternative.resolve()}"
                )
                return alternative.resolve()
    raise ValueError(f"{split}: image directory not found: {primary}")

def audit_dataset(data_yaml, check_content_duplicates=True):
    path = Path(data_yaml).expanduser().resolve()
    cfg = read_yaml(path)
    names = class_names(cfg)
    notices, errors = [], []
    summary = {
        "input_yaml": str(path), "names": names, "splits": {},
        "notices": notices, "errors": errors,
        "duplicate_check": "exact image bytes only; source-family grouping is a separate check",
    }
    resolved = {"path": str(path.parent), "nc": 4, "names": names}
    hash_splits = {}
    for split in ("train", "val", "test"):
        image_dir = resolve_image_dir(path, cfg, split, notices)
        resolved[split] = str(image_dir)
        label_dir = image_dir.parent / "labels"
        images = sorted(p for p in image_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        labels = set(label_dir.glob("*.txt"))
        stems = Counter(p.stem for p in images)
        counts = Counter()
        instances = Counter()
        empty = missing = invalid = zero_area = 0
        if not images:
            errors.append(f"{split}: no images found.")
        for stem, count in stems.items():
            if count > 1:
                errors.append(f"{split}: multiple images share label stem {stem!r}.")
        for image in images:
            label = label_dir / (image.stem + ".txt")
            if not label.is_file():
                missing += 1
                errors.append(f"{split}: missing label for {image.name}")
                continue
            lines = label.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                empty += 1
            seen = set()
            for line_number, line in enumerate(lines, 1):
                try:
                    tokens = line.split()
                    cid = int(tokens[0])
                    xy = [float(v) for v in tokens[1:]]
                    if not 0 <= cid < 4:
                        raise ValueError("class ID outside 0..3")
                    if len(xy) < 6 or len(xy) % 2:
                        raise ValueError("expected class ID followed by at least three polygon vertices")
                    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in xy):
                        raise ValueError("polygon coordinates must be finite and normalized to 0..1")
                    vertices = list(zip(xy[::2], xy[1::2]))
                    area2 = sum(
                        vertices[i][0] * vertices[(i+1) % len(vertices)][1]
                        - vertices[(i+1) % len(vertices)][0] * vertices[i][1]
                        for i in range(len(vertices))
                    )
                    if abs(area2) < 1e-12:
                        zero_area += 1
                    seen.add(cid)
                    instances[cid] += 1
                except (ValueError, IndexError) as exc:
                    invalid += 1
                    errors.append(f"{split}/{label.name}:{line_number}: {exc}")
            counts.update(seen)
            if check_content_duplicates:
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                previous = hash_splits.get(digest)
                if previous is not None and previous[0] != split:
                    errors.append(
                        f"Identical image bytes across splits: {previous[0]}/{previous[1]} and {split}/{image.name}"
                    )
                else:
                    hash_splits[digest] = (split, image.name)
        orphan_labels = sorted(p.name for p in labels if p.stem not in stems)
        if orphan_labels:
            errors.append(f"{split}: {len(orphan_labels)} orphan label files.")
        if zero_area:
            notices.append(
                f"{split}: {zero_area} zero-area polygons found; source files were not changed. "
                "Ultralytics may ignore/correct them during loading."
            )
        absent = [names[i] for i in range(4) if not counts[i]]
        if absent:
            message = f"{split}: no annotated examples of {', '.join(absent)}."
            (errors if split == "train" else notices).append(message)
        summary["splits"][split] = {
            "image_dir": str(image_dir), "label_dir": str(label_dir),
            "images": len(images), "empty_labels": empty,
            "nonempty_labels": len(images) - empty - missing,
            "missing_labels": missing, "invalid_rows": invalid,
            "zero_area_polygons": zero_area, "orphan_labels": len(orphan_labels),
            "images_with_class": {name: counts[i] for i, name in enumerate(names)},
            "instances": {name: instances[i] for i, name in enumerate(names)},
        }
    if summary["splits"]["train"]["empty_labels"] == 0:
        errors.append(
            "No empty training labels found. This Stage 1 protocol requires intentional negatives. "
            "Check that Filter Null was not enabled."
        )
    summary["total_images"] = sum(s["images"] for s in summary["splits"].values())
    summary["total_empty_labels"] = sum(s["empty_labels"] for s in summary["splits"].values())
    return summary, resolved

