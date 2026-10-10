import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(clean(value), sort_keys=False), encoding="utf-8")


def load_recipe(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError("Cyclic recipe inheritance")
    seen.add(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = cfg.pop("extends", None)
    if parent:
        base = load_recipe(path.parent / parent, seen)
        for key, value in cfg.items():
            if key == "train":
                base.setdefault("train", {}).update(value)
            else:
                base[key] = value
        cfg = base
    cfg["recipe_file"] = str(path)
    return cfg


def table(rows, columns):
    if not rows:
        print("No rows.")
        return
    strings = [[str(row.get(c, "")) for c in columns] for row in rows]
    widths = [max(len(c), *(len(row[i]) for row in strings)) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in strings:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def environment(torch_module=None):
    versions = {}
    for name in ["ultralytics", "torch", "torchvision", "numpy", "Pillow", "PyYAML", "opencv-python", "albumentations"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=versions)
    if torch_module is not None:
        torch = torch_module
        result.update(cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
                      cudnn_version=torch.backends.cudnn.version())
        result["gpus"] = [dict(name=torch.cuda.get_device_name(i),
                              memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                          for i in range(torch.cuda.device_count())]
    return result


def save_environment(folder, torch_module=None):
    save_json(Path(folder) / "environment.json", environment(torch_module))
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True)
    (Path(folder) / "pip-freeze.txt").write_text(proc.stdout if proc.returncode == 0 else proc.stderr, encoding="utf-8")


class Tee:
    def __init__(self, original, file):
        self.original, self.file = original, file

    def write(self, text):
        self.original.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.file.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.original, name)
