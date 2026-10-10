#!/usr/bin/env python3
import argparse
from pathlib import Path

from agrieye.common import ROOT
from agrieye.data import prepare
from agrieye.pipeline import benchmark, bundle_records, doctor, evaluate, train_run


def main():
    for name in ["data/raw", "data/prepared", "runs", "reports", "weights"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="AgriEye baseline reproduction: audit, train, evaluate, and record.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Record environment and check CUDA arithmetic")
    p = sub.add_parser("prepare", help="Audit an extracted export and create a separate working copy")
    p.add_argument("--source", required=True)
    p.add_argument("--name", default="agrieye_current")
    for command in ["smoke", "train"]:
        p = sub.add_parser(command)
        p.add_argument("--dataset", required=True, help="Prepared dataset name")
        p.add_argument("--recipe", type=Path, default=ROOT / "configs" / "disease_paper_50e.yaml")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--batch", type=int, help="Explicit hardware adaptation; smoke always uses 1")
        p.add_argument("--device", help="Single GPU index or cpu; default from recipe")
        p.add_argument("--workers", type=int)
        p.add_argument("--weights", help="Optional starting .pt checkpoint; provenance is recorded")
    p = sub.add_parser("evaluate", help="Evaluate best.pt; test is only used when explicitly requested")
    p.add_argument("--run", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--device", default="0")
    p = sub.add_parser("benchmark", help="Print measured tables and separately list paper reference values")
    p.add_argument("--include-smoke", action="store_true")
    sub.add_parser("bundle-records", help="Create a small ZIP of run/audit reports for sharing")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.name)
    elif args.command == "doctor":
        doctor()
    elif args.command in ["smoke", "train"]:
        train_run(args)
    elif args.command == "evaluate":
        evaluate(args.run, args.split, args.device)
    elif args.command == "benchmark":
        benchmark(args.include_smoke)
    elif args.command == "bundle-records":
        bundle_records()


if __name__ == "__main__":
    main()
