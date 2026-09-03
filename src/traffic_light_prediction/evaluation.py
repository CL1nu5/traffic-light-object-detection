"""Tile-level evaluation and stitched full-frame inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .classes import CLASS_METADATA, CLASS_TO_ID
from .config import WorkflowConfig, load_config, resolve_device
from .data import IMAGE_SUFFIXES

VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


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


def _sahi_device(requested: str | int | None) -> str:
    resolved = resolve_device(requested)
    if isinstance(resolved, int) or str(resolved).isdigit():
        return f"cuda:{resolved}"
    return str(resolved)


def _load_sliced_model(
    weights: Path, *, confidence: float, device: str | int | None
) -> Any:
    from sahi import AutoDetectionModel

    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=confidence,
        device=_sahi_device(device),
    )


def _sliced_prediction(
    image: str | Any,
    model: Any,
    config: WorkflowConfig,
    *,
    confidence_threshold: float | None = None,
) -> Any:
    from sahi.predict import get_sliced_prediction

    tiling = config.section("tiling")
    return get_sliced_prediction(
        image,
        model,
        slice_height=int(tiling["size"]),
        slice_width=int(tiling["size"]),
        overlap_height_ratio=float(tiling["overlap_ratio"]),
        overlap_width_ratio=float(tiling["overlap_ratio"]),
        perform_standard_pred=bool(tiling["perform_standard_prediction"]),
        postprocess_type=str(tiling["postprocess_type"]),
        postprocess_match_metric=str(tiling["postprocess_match_metric"]),
        postprocess_match_threshold=float(tiling["postprocess_match_threshold"]),
        postprocess_class_agnostic=False,
        batch_size=int(tiling["inference_batch"]),
        verbose=0,
        confidence_threshold=confidence_threshold,
    )


def _prediction_record(
    result: Any,
    *,
    source: str,
    width: int,
    height: int,
    frame_index: int | None = None,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for prediction in result.object_prediction_list:
        class_name = str(prediction.category.name)
        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"Checkpoint predicted unsupported class {class_name!r}; "
                "use a checkpoint trained on the tiled three-class dataset"
            )
        x1, y1, x2, y2 = [float(value) for value in prediction.bbox.to_xyxy()]
        predictions.append(
            {
                "class_id": CLASS_TO_ID[class_name],
                "class_name": class_name,
                **CLASS_METADATA[class_name],
                "confidence": float(prediction.score.value),
                "box_xyxy_pixels": [x1, y1, x2, y2],
                "box_xyxy_normalized": [x1 / width, y1 / height, x2 / width, y2 / height],
            }
        )
    record: dict[str, Any] = {
        "source": source,
        "image_size": {"width": width, "height": height},
        "predictions": predictions,
    }
    if frame_index is not None:
        record["frame_index"] = frame_index
    return record


def _draw_pil(image: Image.Image, record: dict[str, Any]) -> Image.Image:
    rendered = image.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    for prediction in record["predictions"]:
        box = prediction["box_xyxy_pixels"]
        label = f"{prediction['class_name']} {prediction['confidence']:.2f}"
        draw.rectangle(box, outline=(255, 80, 20), width=3)
        draw.text((box[0] + 3, max(0, box[1] - 14)), label, fill=(255, 80, 20))
    return rendered


def _full_test_data(config: WorkflowConfig) -> tuple[Path, dict[str, Any]]:
    dataset = config.path("processed_data")
    path = dataset / "full_test_coco.json"
    if not path.is_file():
        raise FileNotFoundError(f"Full-frame test annotations not found: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _evaluate_full_frames(
    config: WorkflowConfig, detection_model: Any, evaluation_dir: Path
) -> dict[str, Any]:
    from sahi.scripts.coco_evaluation import evaluate

    coco_path, ground_truth = _full_test_data(config)
    predictions: list[dict[str, Any]] = []
    dataset = config.path("processed_data")
    images = ground_truth["images"]
    for index, image in enumerate(images, start=1):
        source = dataset / image["file_name"]
        result = _sliced_prediction(
            str(source),
            detection_model,
            config,
            confidence_threshold=float(config.section("evaluation")["confidence"]),
        )
        for prediction in result.object_prediction_list:
            class_name = str(prediction.category.name)
            if class_name not in CLASS_TO_ID:
                raise ValueError(
                    f"Checkpoint predicted unsupported class {class_name!r}; "
                    "use a checkpoint trained on the tiled three-class dataset"
                )
            x1, y1, x2, y2 = [float(value) for value in prediction.bbox.to_xyxy()]
            predictions.append(
                {
                    "image_id": int(image["id"]),
                    "category_id": CLASS_TO_ID[class_name] + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(prediction.score.value),
                }
            )
        if index % 100 == 0 or index == len(images):
            print(f"Full-frame evaluation: {index}/{len(images)} images")
    prediction_path = evaluation_dir / "full_frame_predictions_coco.json"
    prediction_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    result = evaluate(
        dataset_json_path=str(coco_path),
        result_json_path=str(prediction_path),
        out_dir=str(evaluation_dir / "full_frame"),
        type="bbox",
        classwise=True,
        max_detections=100,
        return_dict=True,
    )
    return {
        **result["eval_results"],
        "ground_truth": str(coco_path),
        "predictions": str(prediction_path),
    }


def _configured_sources(
    config: WorkflowConfig, source: str | Path | None
) -> tuple[list[Path], Path | None]:
    configured = source or str(config.section("inference").get("source", "")).strip()
    if configured:
        path = Path(configured).expanduser()
        path = path.resolve() if path.is_absolute() else (config.root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Inference source not found: {path}")
        if path.is_dir():
            images = sorted(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
            )
            if not images:
                raise ValueError(f"No supported images found below {path}")
            return images, None
        if path.suffix.casefold() in VIDEO_SUFFIXES:
            return [], path
        if path.suffix.casefold() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported inference source: {path}")
        return [path], None

    _, ground_truth = _full_test_data(config)
    dataset = config.path("processed_data")
    count = int(config.section("inference")["sample_count"])
    return [dataset / item["file_name"] for item in ground_truth["images"][:count]], None


def _infer_images(
    paths: list[Path], model: Any, config: WorkflowConfig, output_dir: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            width, height = opened.size
            result = _sliced_prediction(str(path), model, config)
            record = _prediction_record(result, source=str(path), width=width, height=height)
            _draw_pil(opened, record).save(
                output_dir / f"{index:05d}_{path.stem}_annotated.jpg", quality=95
            )
            records.append(record)
    return records


def _infer_video(
    path: Path, model: Any, config: WorkflowConfig, output_dir: Path
) -> list[dict[str, Any]]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    destination = output_dir / f"{path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    records: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = _sliced_prediction(frame, model, config)
            record = _prediction_record(
                result,
                source=str(path),
                width=width,
                height=height,
                frame_index=frame_index,
            )
            for prediction in record["predictions"]:
                x1, y1, x2, y2 = map(int, prediction["box_xyxy_pixels"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 80, 255), 2)
                cv2.putText(
                    frame,
                    f"{prediction['class_name']} {prediction['confidence']:.2f}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (20, 80, 255),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            records.append(record)
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Video inference: processed {frame_index} frames")
    finally:
        capture.release()
        writer.release()
    return records


def evaluate_and_infer(
    config_path: str | Path = ".config/config.toml",
    *,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate tiles and stitched test frames, then run structured inference."""

    from ultralytics import YOLO

    config = load_config(config_path)
    evaluation = config.section("evaluation")
    inference = config.section("inference")
    weights = _weights_path(config)
    if not weights.is_file():
        raise FileNotFoundError(f"Trained weights not found: {weights}")

    output_root = config.path("output")
    evaluation_dir = output_root / "evaluation" / str(evaluation["run_name"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    tile_result = model.val(
        data=str(config.path("processed_data") / "dataset.yaml"),
        split="test",
        device=resolve_device(evaluation.get("device")),
        batch=int(evaluation["batch"]),
        workers=int(evaluation["workers"]),
        plots=True,
        project=str(output_root / "evaluation"),
        name=f"{evaluation['run_name']}_tiles",
        exist_ok=True,
    )
    tile_metrics = {
        key: _plain(value)
        for key, value in getattr(tile_result, "results_dict", {}).items()
    }

    del model, tile_result
    evaluation_model = _load_sliced_model(
        weights,
        confidence=float(evaluation["confidence"]),
        device=evaluation.get("device"),
    )
    full_metrics = _evaluate_full_frames(config, evaluation_model, evaluation_dir)
    del evaluation_model
    metrics = {"tiles": tile_metrics, "full_frames": full_metrics}
    (evaluation_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    inference_dir = output_root / "inference" / str(inference["run_name"])
    inference_dir.mkdir(parents=True, exist_ok=True)
    sliced_model = _load_sliced_model(
        weights,
        confidence=float(inference["confidence"]),
        device=inference.get("device"),
    )
    image_paths, video_path = _configured_sources(config, source)
    prediction_records = (
        _infer_video(video_path, sliced_model, config, inference_dir)
        if video_path
        else _infer_images(image_paths, sliced_model, config, inference_dir)
    )
    predictions_path = inference_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(prediction_records, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "weights": str(weights),
        "metrics": metrics,
        "predictions": str(predictions_path),
        "prediction_count": sum(len(item["predictions"]) for item in prediction_records),
    }
