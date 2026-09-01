"""Test-set evaluation and structured inference."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .classes import CLASS_METADATA
from .config import WorkflowConfig, load_config, resolve_device


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _weights_path(config: WorkflowConfig) -> Path:
    evaluation = config.section("evaluation")
    configured = str(evaluation.get("weights", "")).strip()
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_absolute() else (config.root / path).resolve()
    run_name = str(config.section("training")["run_name"])
    return config.path("output") / "training" / run_name / "weights" / "best.pt"


def _sample_test_images(config: WorkflowConfig, count: int) -> list[str]:
    manifest = config.path("processed_data") / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    images: list[str] = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                images.append(str(config.path("processed_data") / row["image"]))
                if len(images) >= count:
                    break
    if not images:
        raise ValueError("No test images found in the dataset manifest")
    return images


def _prediction_record(result: Any) -> dict[str, Any]:
    height, width = result.orig_shape
    predictions: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(result.names[class_id])
            metadata = CLASS_METADATA.get(
                class_name, {"color": "unknown", "direction": "unknown"}
            )
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    **metadata,
                    "confidence": float(box.conf.item()),
                    "box_xyxy_pixels": [float(value) for value in box.xyxy[0].tolist()],
                    "box_xyxy_normalized": [float(value) for value in box.xyxyn[0].tolist()],
                }
            )
    return {
        "source": str(result.path),
        "image_size": {"width": int(width), "height": int(height)},
        "predictions": predictions,
    }


def evaluate_and_infer(
    config_path: str | Path = ".config/config.toml",
    *,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate on the test split, then run structured inference."""

    from ultralytics import YOLO

    config = load_config(config_path)
    evaluation = config.section("evaluation")
    inference = config.section("inference")
    weights = _weights_path(config)
    if not weights.is_file():
        raise FileNotFoundError(f"Trained weights not found: {weights}")

    model = YOLO(str(weights))
    output_root = config.path("output")
    metrics = model.val(
        data=str(config.path("processed_data") / "dataset.yaml"),
        split="test",
        device=resolve_device(evaluation.get("device")),
        batch=int(evaluation["batch"]),
        workers=int(evaluation["workers"]),
        plots=True,
        project=str(output_root / "evaluation"),
        name=str(evaluation["run_name"]),
        exist_ok=True,
    )
    metrics_dict = {
        key: _plain(value) for key, value in getattr(metrics, "results_dict", {}).items()
    }
    evaluation_dir = output_root / "evaluation" / str(evaluation["run_name"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "metrics.json").write_text(
        json.dumps(metrics_dict, indent=2) + "\n", encoding="utf-8"
    )

    configured_source = source or str(inference.get("source", "")).strip()
    prediction_source: str | list[str]
    if configured_source:
        source_path = Path(configured_source).expanduser()
        prediction_source = str(
            source_path.resolve()
            if source_path.is_absolute()
            else (config.root / source_path).resolve()
        )
    else:
        prediction_source = _sample_test_images(config, int(inference["sample_count"]))

    prediction_records = [
        _prediction_record(result)
        for result in model.predict(
            source=prediction_source,
            stream=True,
            conf=float(inference["confidence"]),
            iou=float(inference["iou"]),
            device=resolve_device(inference.get("device")),
            save=True,
            project=str(output_root / "inference"),
            name=str(inference["run_name"]),
            exist_ok=True,
        )
    ]
    inference_dir = output_root / "inference" / str(inference["run_name"])
    inference_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = inference_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(prediction_records, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "weights": str(weights),
        "metrics": metrics_dict,
        "predictions": str(predictions_path),
        "prediction_count": sum(len(item["predictions"]) for item in prediction_records),
    }
