"""YOLO11 training stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_config, resolve_device


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def train_model(config_path: str | Path = ".config/config.toml") -> Path:
    """Train YOLO and return the best checkpoint path."""

    from ultralytics import YOLO

    config = load_config(config_path)
    settings = config.section("training")
    dataset_yaml = config.path("processed_data") / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_yaml}")

    project = config.path("output") / "training"
    run_name = str(settings["run_name"])
    model = YOLO(str(settings["model"]))
    results = model.train(
        data=str(dataset_yaml),
        imgsz=int(settings["image_size"]),
        epochs=int(settings["epochs"]),
        batch=int(settings["batch"]),
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
    summary = {
        "model": str(settings["model"]),
        "image_size": int(settings["image_size"]),
        "best_weights": str(best_weights),
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
