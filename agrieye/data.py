import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .common import ROOT, read_json, save_json, save_yaml, sha256, stamp


def alias_path(alias):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", alias):
        raise ValueError("Dataset name must contain only letters, digits, underscores, and hyphens.")
    return ROOT / "data" / "prepared" / alias


def fingerprint(rows, names):
    inventory = [dict(image=r["image"], image_sha256=r["sha256"],
                      label=r["label"], label_sha256=r["label_sha256"], split=r["split"]) for r in rows]
    payload = dict(names=names, files=sorted(inventory, key=lambda r: (r["split"], r["image"])))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(source, alias):
    source = Path(source).expanduser().resolve()
    destination = alias_path(alias)
    if destination.exists():
        raise ValueError(f"Prepared dataset already exists: {destination}. Use a new --name; existing snapshots are preserved.")
    if not (source / "data.yaml").is_file():
        raise ValueError("--source must be the extracted folder containing data.yaml, train, valid, and test.")
    report = ROOT / "reports" / f"audit_{alias}_{stamp()}"
    subprocess.run([sys.executable, str(ROOT / "tools" / "audit_dataset.py"), str(source), "--output", str(report)], check=True)
    audit = read_json(report / "summary.json")
    if audit["issues_by_severity"].get("error", 0):
        raise ValueError(f"Audit found annotation/file errors. Inspect {report / 'issues.csv'} before preparing training data.")
    with (report / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        label = source / row["label"]
        row["label_sha256"] = sha256(label) if label.is_file() else None
        row["class_ids"] = [int(i) for i in row["class_ids"].split(";") if i]
    print("Creating an independent working copy; raw images and annotations stay unchanged.", flush=True)
    destination.mkdir(parents=True)
    try:
        for split in ["train", "valid", "test"]:
            shutil.copytree(source / split, destination / split, ignore=shutil.ignore_patterns("*.cache", "*.npy"))
        source_info = destination / "source_export_info"
        source_info.mkdir()
        for name in ["data.yaml", "README.dataset.txt", "README.roboflow.txt"]:
            if (source / name).is_file():
                shutil.copy2(source / name, source_info / name)
        data = dict(path=str(destination), train="train/images", val="valid/images", test="test/images",
                    nc=len(audit["names"]), names=audit["names"])
        save_yaml(destination / "data.yaml", data)
        record = dict(alias=alias, prepared_at_utc=stamp(), source_root=str(source),
                      prepared_root=str(destination), names=audit["names"],
                      fingerprint=fingerprint(rows, audit["names"]), files=rows, audit=audit,
                      audit_directory=str(report), data_yaml_sha256=sha256(destination / "data.yaml"),
                      extra_illumination_augmentation=False,
                      comparison_status="Dataset-to-paper-run correspondence not confirmed; no reproduction claim.")
        save_json(destination / "dataset_record.json", record)
        verify_dataset(destination)
    except BaseException:
        save_json(destination / "PREPARATION_INCOMPLETE.json", {"message": "Preparation failed. Inspect before reuse; use a new alias after correction."})
        raise
    print(f"\nPrepared dataset: {alias}\nClass order: {audit['names']}\nFingerprint: {record['fingerprint']}")
    if audit["review_required"]:
        print("REVIEW REQUIRED: see audit reports. Split-overlap candidates are not automatically resolved.")
    print(f"Next: python run.py smoke --dataset {alias}")
    return record


def verify_dataset(folder):
    folder = Path(folder)
    record = read_json(folder / "dataset_record.json")
    if sha256(folder / "data.yaml") != record["data_yaml_sha256"]:
        raise ValueError("Prepared data.yaml changed after its snapshot; prepare a new dataset alias.")
    expected_images, expected_labels = set(), set()
    for row in record["files"]:
        expected_images.add(row["image"])
        if sha256(folder / row["image"]) != row["sha256"]:
            raise ValueError(f"Prepared image changed: {row['image']}")
        label = folder / row["label"]
        actual = sha256(label) if label.is_file() else None
        if actual != row["label_sha256"]:
            raise ValueError(f"Prepared annotation changed: {row['label']}")
        if actual is not None:
            expected_labels.add(row["label"])
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    actual_images = {str(p.relative_to(folder)) for split in ["train", "valid", "test"]
                     for p in (folder / split / "images").rglob("*") if p.is_file() and p.suffix.lower() in image_ext}
    if actual_images != expected_images:
        raise ValueError("Prepared image membership changed; create a new snapshot.")
    # Existing orphan labels were recorded by the audit and are not training inputs.
    return record


def smoke_subset(folder, run_folder, seed):
    folder, run_folder = Path(folder), Path(run_folder)
    record = read_json(folder / "dataset_record.json")
    destination = run_folder / "smoke_data"
    destination.mkdir()
    chosen = []
    rng = random.Random(seed)
    for split, split_dir, limit in [("train", "train", 32), ("val", "valid", 16)]:
        candidates = [r for r in record["files"] if r["split"] == split]
        rng.shuffle(candidates)
        selected = []
        # Include one available example per class before filling the small sample.
        for class_id in range(len(record["names"])):
            match = next((r for r in candidates if class_id in r["class_ids"] and r not in selected), None)
            if match is not None:
                selected.append(match)
        selected += [r for r in candidates if r not in selected][:max(0, limit - len(selected))]
        for row in selected[:limit]:
            for key in ["image", "label"]:
                src, dst = folder / row[key], destination / row[key]
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            chosen.append(row)
        (destination / split_dir / "labels").mkdir(parents=True, exist_ok=True)
    save_yaml(destination / "data.yaml", dict(path=str(destination), train="train/images", val="valid/images",
                                              nc=len(record["names"]), names=record["names"]))
    save_json(destination / "sample_manifest.json", chosen)
    return destination / "data.yaml"
