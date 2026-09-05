"""Configuration loading and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowConfig:
    """Parsed workflow configuration with paths anchored to the project root."""

    root: Path
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        try:
            return self.values[name]
        except KeyError as exc:
            raise KeyError(f"Missing [{name}] section in config") from exc

    def path(self, name: str) -> Path:
        value = self.section("paths").get(name)
        if not value:
            raise KeyError(f"Missing paths.{name} in config")
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()


def load_config(path: str | Path = ".config/config.toml") -> WorkflowConfig:
    """Load the TOML configuration and validate important invariants."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("rb") as handle:
        values = tomllib.load(handle)

    required = {"paths", "dataset", "tiling", "training", "evaluation", "inference"}
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    dataset = values["dataset"]
    ratios = [
        float(dataset.get("train_ratio", 0)),
        float(dataset.get("validation_ratio", 0)),
        float(dataset.get("test_ratio", 0)),
    ]
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Dataset split ratios must be positive and sum to 1.0")

    tiling = values["tiling"]
    if int(tiling.get("size", 0)) <= 0:
        raise ValueError("tiling.size must be positive")
    overlap = float(tiling.get("overlap_ratio", -1))
    if not 0 <= overlap < 1:
        raise ValueError("tiling.overlap_ratio must be in [0, 1)")
    min_area = float(tiling.get("min_box_area_ratio", -1))
    if not 0 < min_area <= 1:
        raise ValueError("tiling.min_box_area_ratio must be in (0, 1]")
    if float(tiling.get("max_empty_to_positive_ratio", -1)) < 0:
        raise ValueError("tiling.max_empty_to_positive_ratio must be non-negative")
    if not 1 <= int(tiling.get("jpeg_quality", 0)) <= 100:
        raise ValueError("tiling.jpeg_quality must be in [1, 100]")
    if int(tiling.get("inference_batch", 0)) <= 0:
        raise ValueError("tiling.inference_batch must be positive")
    if int(values["training"].get("image_size", 0)) != int(tiling["size"]):
        raise ValueError("training.image_size must match tiling.size")
    training = values["training"]
    if int(training.get("epochs", 100)) <= 0:
        raise ValueError("training.epochs must be positive")
    batch = int(training.get("batch", 16))
    if batch != -1 and batch <= 0:
        raise ValueError("training.batch must be -1 for auto-batch or a positive integer")
    if float(training.get("lr0", 0.001)) <= 0:
        raise ValueError("training.lr0 must be positive")
    if float(training.get("lrf", 0.01)) <= 0:
        raise ValueError("training.lrf must be positive")
    if float(training.get("weight_decay", 0.0005)) < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if float(training.get("warmup_epochs", 3.0)) < 0:
        raise ValueError("training.warmup_epochs must be non-negative")
    if float(training.get("warmup_bias_lr", 0.0)) < 0:
        raise ValueError("training.warmup_bias_lr must be non-negative")
    momentum = float(training.get("momentum", 0.9))
    if not 0 <= momentum < 1:
        raise ValueError("training.momentum must be in [0, 1)")
    if int(training.get("close_mosaic", 10)) < 0:
        raise ValueError("training.close_mosaic must be non-negative")
    for name in ("cls_pw", "mosaic", "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "fliplr"):
        value = float(training.get(name, 0.0))
        if not 0 <= value <= 1:
            raise ValueError(f"training.{name} must be in [0, 1]")

    root = config_path.parent.parent
    return WorkflowConfig(root=root, values=values)


def resolve_device(requested: str | int | None) -> str | int:
    """Resolve 'auto' consistently across CUDA, Apple Silicon, and CPU."""

    if requested not in (None, "", "auto"):
        return requested

    import torch

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
