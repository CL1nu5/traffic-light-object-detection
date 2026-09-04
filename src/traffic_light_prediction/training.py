"""YOLO11 training stage."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile
from typing import Any

from .config import load_config, resolve_device
from .data import validate_yolo_dataset


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _save_epoch_metrics_csv(run_dir: Path) -> Path:
    """Keep a clearly named copy of Ultralytics' per-epoch metrics."""

    source = run_dir / "results.csv"
    destination = run_dir / "epoch_metrics.csv"
    if not source.is_file():
        raise RuntimeError(f"Training did not produce per-epoch metrics: {source}")
    copyfile(source, destination)
    return destination


def train_model(config_path: str | Path = ".config/config.toml") -> Path:
    """Train YOLO and return the best checkpoint path."""

    from ultralytics import YOLO

    config = load_config(config_path)
    settings = config.section("training")
    dataset_yaml = config.path("processed_data") / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_yaml}")
    dataset_stats = validate_yolo_dataset(
        config.path("processed_data"), clear_caches=True
    )
    print(
        f"Validated {dataset_stats['label_files']} label files with "
        f"{dataset_stats['instances']} instances; removed "
        f"{dataset_stats['removed_caches']} stale label caches"
    )

    project = config.path("output") / "training"
    run_name = str(settings["run_name"])
    model = YOLO(str(settings["model"]))
    results = model.train(
        data=str(dataset_yaml),
        imgsz=int(settings["image_size"]),
        epochs=int(settings["epochs"]),
        batch=int(settings["batch"]),
        optimizer=str(settings.get("optimizer", "AdamW")),
        lr0=float(settings.get("lr0", 0.001)),
        momentum=float(settings.get("momentum", 0.9)),
        warmup_bias_lr=float(settings.get("warmup_bias_lr", 0.0)),
        device=resolve_device(settings.get("device")),
        workers=int(settings["workers"]),
        seed=int(settings["seed"]),
        deterministic=bool(settings["deterministic"]),
        patience=int(settings["patience"]),
        cache=settings["cache"],
        project=str(project),
        name=run_name,
        exist_ok=True,
    )

    run_dir = project / run_name
    best_weights = run_dir / "weights" / "best.pt"
    epoch_metrics_csv = _save_epoch_metrics_csv(run_dir)
    summary = {
        "model": str(settings["model"]),
        "image_size": int(settings["image_size"]),
        "optimizer": str(settings.get("optimizer", "AdamW")),
        "lr0": float(settings.get("lr0", 0.001)),
        "momentum": float(settings.get("momentum", 0.9)),
        "warmup_bias_lr": float(settings.get("warmup_bias_lr", 0.0)),
        "best_weights": str(best_weights),
        "epoch_metrics_csv": str(epoch_metrics_csv),
        "metrics": {
            key: _json_value(value)
            for key, value in getattr(results, "results_dict", {}).items()
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return best_weights
