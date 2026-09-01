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

    required = {"paths", "dataset", "training", "evaluation", "inference"}
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
