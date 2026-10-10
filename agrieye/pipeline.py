import contextlib
import csv
import importlib.util
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

import yaml

from .common import ROOT, Tee, clean, environment, load_recipe, read_json, save_environment, save_json, save_yaml, sha256, stamp, table, write_csv
from .data import alias_path, smoke_subset, verify_dataset
from .metrics import collect_metrics


def ml_modules():
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt in the active Python environment first.") from exc
    if ultralytics.__version__ != "8.4.41":
        raise RuntimeError(f"Expected ultralytics 8.4.41, found {ultralytics.__version__}. Use the pinned environment.")
    return torch, ultralytics, YOLO


def check_device(torch, device):
    if str(device) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate a working GPU runtime; --device cpu is an explicit slow diagnostic option.")
    if "," in str(device):
        raise ValueError("This logging wrapper supports one GPU. The paper's two-GPU leaf run is documented, not implemented here.")


def doctor():
    folder = ROOT / "reports" / f"environment_{stamp()}"
    folder.mkdir(parents=True)
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("ultralytics") is None:
        save_environment(folder)
        print("Training packages missing. Dataset preparation still works with requirements-audit.txt.")
        print(f"Environment record: {folder}")
        return False
    torch, _, YOLO = ml_modules()
    save_environment(folder, torch)
    info = environment(torch)
    print(json.dumps(info, indent=2))
    # Exercise CUDA rather than relying only on device enumeration.
    if torch.cuda.is_available():
        x = torch.ones((32, 32), device="cuda")
        float((x @ x).sum().item())
        torch.cuda.synchronize()
        print("CUDA arithmetic check passed.")
    print(f"Environment record: {folder}")
    return True


def normalized(names):
    return [" ".join(n.split()).casefold() for n in names]


def prepare_run(alias, recipe_path, smoke, seed, batch, device, workers, weights):
    folder = alias_path(alias)
    record = verify_dataset(folder)
    recipe = load_recipe(recipe_path)
    expected = recipe.get("expected_classes")
    if expected and normalized(expected) != normalized(record["names"]):
        raise ValueError(f"This recipe requires class order {expected}, but the dataset has {record['names']}. No class IDs were changed.")
    kind = "smoke" if smoke else "baseline"
    run = ROOT / "runs" / f"{stamp()}_{kind}_{recipe['id']}_s{seed}"
    run.mkdir(parents=True)
    train = dict(recipe["train"])
    deviations = []
    for key, value in [("batch", batch), ("device", device), ("workers", workers)]:
        if value is not None:
            previous = train.get(key, "pinned framework default")
            if str(previous) != str(value):
                deviations.append(f"{key}: {previous} -> {value}")
            train[key] = value
    if weights is not None:
        deviations.append("Starting checkpoint supplied explicitly; provenance must be checked against the paper.")
    data_yaml = folder / "data.yaml"
    if smoke:
        data_yaml = smoke_subset(folder, run, seed)
        train.update(epochs=1, batch=1, workers=0, plots=False, save_period=-1)
        deviations.append("Smoke-only: at most 32 train / 16 validation images; one epoch; batch 1; workers 0; plots off.")
    # No new photometric transformations; retain the paper's ordinary augmentations.
    train.update(data=str(data_yaml), project=str(run), name="fit", exist_ok=False,
                 seed=seed, deterministic=True, save=True, val=True)
    state = dict(run_id=run.name, run_kind=kind, status="prepared", recipe=recipe,
                 recipe_sha256=sha256(recipe_path), dataset_alias=alias,
                 dataset_fingerprint=record["fingerprint"], class_names=record["names"],
                 data_yaml=str(data_yaml), dataset_audit=record["audit"],
                 requested_train_arguments=train, deviations=deviations,
                 seed_note="Seed recorded explicitly; authors' seed not recovered.",
                 extra_illumination_augmentation=False, optimizer_updates=0,
                 raw_images_unchanged=True, benchmark_reproduction_confirmed=False)
    save_json(run / "run.json", state)
    save_yaml(run / "requested_train.yaml", train)
    save_yaml(run / "recipe_resolved.yaml", recipe)
    save_json(run / "dataset_snapshot.json", record)
    return run, state


def train_run(args):
    run, state = prepare_run(args.dataset, args.recipe, args.command == "smoke", args.seed,
                             args.batch, args.device, args.workers, args.weights)
    log_path = run / "console.log"
    counts = dict(optimizer_updates=0, completed_epochs=0)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        try:
            torch, ultralytics, YOLO = ml_modules()
            save_environment(run, torch)
            train = state["requested_train_arguments"]
            check_device(torch, train.get("device", 0))
            if environment()["packages"].get("albumentations"):
                print("NOTE: Albumentations is installed. Ultralytics may enable its own optional default transforms; this environment is recorded.")
            initial = args.weights or state["recipe"]["model"]
            if initial == "yolov8n-seg.pt":
                # Ultralytics downloads the official checkpoint here on first use.
                initial = str(ROOT / "weights" / "yolov8n-seg.pt")
            model = YOLO(initial, task="segment")
            if model.task != "segment":
                raise RuntimeError("Expected a segmentation model.")
            ckpt_path = Path(model.ckpt_path).resolve()
            state["initial_checkpoint"] = dict(path=str(ckpt_path), sha256=sha256(ckpt_path))
            state["model_yaml"] = clean(model.model.yaml)
            state["status"] = "training"
            save_json(run / "run.json", state)

            def before_training(trainer):
                save_json(run / "resolved_training_arguments.json", vars(trainer.args))
                (run / "training_transforms.txt").write_text(repr(trainer.train_loader.dataset.transforms), encoding="utf-8")
                state["effective_amp"] = bool(trainer.amp)
                state["optimizer_class"] = type(trainer.optimizer).__name__
                state["initial_gradient_accumulation"] = int(trainer.accumulate)
                save_json(run / "run.json", state)
                def count_update(optimizer, hook_args, hook_kwargs):
                    counts["optimizer_updates"] += 1
                if not hasattr(trainer.optimizer, "register_step_post_hook"):
                    raise RuntimeError("This recorder requires PyTorch optimizer step hooks (use a current working PyTorch environment).")
                trainer.optimizer.register_step_post_hook(count_update)

            def epoch_start(trainer):
                # 8.4.41 can silently reduce batch on OOM. Stop instead of treating
                # an automatically changed recipe as the requested reproduction.
                if int(trainer.batch_size) != int(train["batch"]) or getattr(trainer, "_oom_retries", 0):
                    raise RuntimeError("Ultralytics attempted an OOM retry/batch change. Use a larger GPU, or explicitly choose a smaller --batch for a newly labeled adapted run.")

            def epoch_end(trainer):
                counts["completed_epochs"] = int(trainer.epoch) + 1
                save_json(run / "training_progress.json", counts)

            model.add_callback("on_pretrain_routine_end", before_training)
            model.add_callback("on_train_epoch_start", epoch_start)
            model.add_callback("on_train_epoch_end", epoch_end)
            state["setup_wall_seconds"] = time.perf_counter() - start
            train_started = time.perf_counter()
            model.train(**train)
            state.update(counts)
            state.update(status="trained", training_wall_seconds=time.perf_counter() - train_started,
                         best_checkpoint=str(model.trainer.best), last_checkpoint=str(model.trainer.last),
                         best_checkpoint_sha256=sha256(model.trainer.best),
                         actual_batch=int(model.trainer.batch_size))
            save_json(run / "run.json", state)
            # A separate, recorded validation run with explicit evaluation settings.
            evaluate(run, split="val", device=train.get("device", 0), smoke=state["run_kind"] == "smoke")
            state["status"] = "complete"
            save_json(run / "run.json", state)
            print(f"\nRUN COMPLETE: {run}")
        except BaseException as exc:
            state.update(counts)
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         elapsed_wall_seconds=time.perf_counter() - start, error=str(exc))
            save_json(run / "run.json", state)
            (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"\nRun records preserved: {run}")
            raise
    return run


def resolve_run(value):
    path = Path(value).expanduser()
    if not path.is_absolute() and not path.exists():
        path = ROOT / "runs" / value
    path = path.resolve()
    if not (path / "run.json").is_file():
        raise ValueError("Supply a run ID or a run folder containing run.json.")
    return path


def evaluate(run, split="val", device=0, smoke=False):
    run = resolve_run(run)
    state = read_json(run / "run.json")
    smoke = smoke or state["run_kind"] == "smoke"
    if split == "test" and smoke:
        raise ValueError("Smoke runs use validation only and are excluded from scientific comparisons.")
    if split not in ["val", "test"]:
        raise ValueError("Use val or test.")
    verify_dataset(alias_path(state["dataset_alias"]))
    torch, _, YOLO = ml_modules()
    check_device(torch, device)
    model = YOLO(state["best_checkpoint"], task="segment")
    model_names = [model.names[i] for i in range(len(model.names))]
    if normalized(model_names) != normalized(state["class_names"]):
        raise ValueError("Checkpoint class order differs from the prepared dataset.")
    if sha256(state["best_checkpoint"]) != state["best_checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes changed after this run was recorded.")
    cfg = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text())
    if smoke:
        cfg.update(batch=1, workers=0, plots=False)
    cfg.update(data=state["data_yaml"], split=split, device=device,
               project=str(run / "evaluations"), name=f"{split}_{stamp()}", exist_ok=False)
    # A named test command is the only route to final test evaluation. No search
    # or repeated test-driven tuning is implemented by this setup.
    event = dict(at_utc=stamp(), run_id=run.name, split=split,
                 dataset_fingerprint=state["dataset_fingerprint"], checkpoint_sha256=state["best_checkpoint_sha256"])
    if split == "test":
        with (ROOT / "reports" / "test_access.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
    started = time.perf_counter()
    validator_output = {}
    def capture_validator(validator):
        validator_output["save_dir"] = str(validator.save_dir)
    model.add_callback("on_val_start", capture_validator)
    result = model.val(**cfg)
    metrics = collect_metrics(result, state["class_names"])
    metrics.update(event, run_kind=state["run_kind"], evaluation_wall_seconds=time.perf_counter() - started,
                   evaluation_arguments=cfg, environment=environment(torch),
                   class_names=state["class_names"], reproduction_confirmed=False)
    directory = Path(validator_output.get("save_dir", Path(cfg["project"]) / cfg["name"]))
    save_json(directory / "metrics.json", metrics)
    save_yaml(directory / "evaluation_arguments.yaml", cfg)
    per_class = [dict(kind=kind, **row) for kind in ["box", "mask"] for row in metrics[kind]["per_class"]]
    write_csv(directory / "per_class.csv", per_class,
              ["kind", "class_id", "class_name", "evaluated", "precision", "recall", "f1", "ap50", "ap50_95"])
    table([dict(Task=kind, Split=split, Precision=f"{metrics[kind]['precision']:.4f}",
                Recall=f"{metrics[kind]['recall']:.4f}", Macro_F1=f"{metrics[kind]['f1_macro']:.4f}",
                mAP50=f"{metrics[kind]['map50']:.4f}", mAP50_95=f"{metrics[kind]['map50_95']:.4f}")
           for kind in ["box", "mask"]], ["Task", "Split", "Precision", "Recall", "Macro_F1", "mAP50", "mAP50_95"])
    print(f"Metrics saved: {directory}")
    return metrics


def benchmark(include_smoke=False):
    rows = []
    for path in sorted((ROOT / "runs").glob("*/evaluations/*/metrics.json")):
        m = read_json(path)
        if m["run_kind"] == "smoke" and not include_smoke:
            continue
        run = read_json(path.parents[2] / "run.json")
        for kind in ["box", "mask"]:
            block = m[kind]
            rows.append(dict(run=m["run_id"], kind=kind, split=m["split"], classes=len(m["class_names"]),
                             precision=block["precision"], recall=block["recall"], f1_macro=block["f1_macro"],
                             f1_from_mean_precision_recall=block["f1_from_mean_precision_recall"],
                             map50=block["map50"], map50_95=block["map50_95"],
                             optimizer_updates=run.get("optimizer_updates"),
                             completed_epochs=run.get("completed_epochs"),
                             dataset_fingerprint=m["dataset_fingerprint"], metrics_file=str(path)))
    output = ROOT / "reports" / f"benchmark_{stamp()}"
    output.mkdir(parents=True)
    fields = ["run", "kind", "split", "classes", "precision", "recall", "f1_macro", "f1_from_mean_precision_recall", "map50", "map50_95", "optimizer_updates", "completed_epochs", "dataset_fingerprint", "metrics_file"]
    write_csv(output / "measured_results.csv", rows, fields)
    table([dict(run=r["run"], kind=r["kind"], split=r["split"], classes=r["classes"],
                P=f"{r['precision']:.4f}", R=f"{r['recall']:.4f}", F1=f"{r['f1_macro']:.4f}",
                mAP50=f"{r['map50']:.4f}", mAP50_95=f"{r['map50_95']:.4f}") for r in rows],
          ["run", "kind", "split", "classes", "P", "R", "F1", "mAP50", "mAP50_95"])
    refs = read_json(ROOT / "references" / "paper_benchmarks.json")
    save_json(output / "paper_references.json", refs)
    print("\nPublished reference values (metric namespace, exact split, and dataset version not yet confirmed):")
    table(refs["rows"], ["version", "classes", "epochs", "precision", "recall", "f1", "map50", "map50_95", "iou"])
    print("No pass/fail reproduction verdict or automatic closeness claim is generated from unmatched references.")
    print(f"Benchmark records: {output}")
    return output


def bundle_records():
    import zipfile
    output = ROOT / "reports" / f"records_to_share_{stamp()}.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for base in [ROOT / "runs", ROOT / "reports"]:
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".yaml", ".txt", ".log"}:
                    if "smoke_data" not in path.parts:
                        archive.write(path, path.relative_to(ROOT))
    print(f"Records ZIP (no model weights or dataset images): {output}")
    return output
