"""Small checks for risks that would invalidate experimental records.

Run from project root: python -m unittest discover -s tests -v
No GPU or training packages are needed.
"""
import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image

from agrieye import data, pipeline
from agrieye.common import ROOT, load_recipe
from agrieye.metrics import metric_block


class MetricLabels(unittest.TestCase):
    def test_absent_classes_are_not_assigned_someone_elses_scores(self):
        metric = SimpleNamespace(ap_class_index=[0, 3], p=[0.8, 0.2], r=[0.5, 0.9],
                                 f1=[0.615, 0.327], ap50=[0.9, 0.3], ap=[0.7, 0.1],
                                 mp=0.5, mr=0.7, map50=0.6, map=0.4)
        result = metric_block(metric, ["BLB", "BLS", "LEAF BLAST", "NBS"])
        self.assertIsNone(result["per_class"][1]["ap50"])
        self.assertIsNone(result["per_class"][2]["ap50"])
        self.assertEqual(result["per_class"][3]["ap50"], 0.3)
        self.assertNotAlmostEqual(result["f1_macro"], result["f1_from_mean_precision_recall"])


class DatasetRecords(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.project = self.root / "project"
        self.source = self.root / "raw_export"
        for name in ["tools", "reports", "data/prepared", "runs"]:
            (self.project / name).mkdir(parents=True)
        shutil.copy2(ROOT / "tools" / "audit_dataset.py", self.project / "tools" / "audit_dataset.py")
        self.source.mkdir()
        (self.source / "data.yaml").write_text(yaml.safe_dump(dict(nc=4, names=["BLB", "BLS", "LEAF BLAST", "NBS"])))
        for i, split in enumerate(["train", "valid", "test"]):
            for folder in ["images", "labels"]:
                (self.source / split / folder).mkdir(parents=True)
            image = self.source / split / "images" / f"leaf{i}.rf.abc.jpg"
            Image.new("RGB", (32, 32), (20 + i * 50, 160, 60)).save(image)
            image.with_name(image.stem + ".txt")  # filenames remain unchanged
            (self.source / split / "labels" / (image.stem + ".txt")).write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepare_preserves_raw_and_detects_changed_annotation(self):
        original = {str(p.relative_to(self.source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.source.rglob("*") if p.is_file()}
        with patch.object(data, "ROOT", self.project):
            record = data.prepare(self.source, "fixture")
            prepared = data.alias_path("fixture")
            self.assertEqual(len(record["files"]), 3)
            data.verify_dataset(prepared)
            run = self.project / "runs" / "sample"
            run.mkdir()
            sample_yaml = data.smoke_subset(prepared, run, 0)
            cfg = yaml.safe_load(sample_yaml.read_text())
            self.assertNotIn("test", cfg)
            self.assertFalse((sample_yaml.parent / "test").exists())
            label = prepared / record["files"][0]["label"]
            label.write_text("0 0.1 0.1 0.8 0.1 0.8 0.8\n")
            with self.assertRaisesRegex(ValueError, "annotation changed"):
                data.verify_dataset(prepared)
        after = {str(p.relative_to(self.source)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(original, after)

    def test_five_class_candidate_rejects_four_class_dataset_before_training(self):
        with patch.object(data, "ROOT", self.project), patch.object(pipeline, "ROOT", self.project):
            data.prepare(self.source, "fixture")
            with self.assertRaisesRegex(ValueError, "requires class order"):
                pipeline.prepare_run("fixture", ROOT / "configs" / "disease_v8_100e_candidate.yaml",
                                     False, 0, None, None, None, None)

    def test_exact_and_parent_overlap_are_reported_not_silently_repaired(self):
        src = next((self.source / "train" / "images").glob("*.jpg"))
        dst = self.source / "test" / "images" / src.name
        shutil.copy2(src, dst)
        shutil.copy2(self.source / "train" / "labels" / (src.stem + ".txt"),
                     self.source / "test" / "labels" / (src.stem + ".txt"))
        with patch.object(data, "ROOT", self.project):
            record = data.prepare(self.source, "overlap")
            self.assertEqual(record["audit"]["cross_split_overlap_groups"]["exact_image_bytes"], 1)
            self.assertEqual(record["audit"]["cross_split_overlap_groups"]["parent_filename_candidate"], 1)
            self.assertTrue((data.alias_path("overlap") / "test" / "images" / src.name).exists())


if __name__ == "__main__":
    unittest.main()
