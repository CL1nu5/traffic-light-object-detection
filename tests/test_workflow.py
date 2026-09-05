from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from traffic_light_prediction.classes import CLASS_METADATA, CLASS_REMAP, MODEL_CLASSES
from traffic_light_prediction.config import WorkflowConfig, load_config, resolve_device
from traffic_light_prediction.data import (
    Box,
    Frame,
    read_lisa_frames,
    prepare_dataset,
    split_frames,
    tile_frame,
    validate_yolo_dataset,
    write_yolo_dataset,
)
from traffic_light_prediction.evaluation import _match_predictions, _prediction_record
from traffic_light_prediction.training import _save_epoch_metrics_csv


def _make_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="gray").save(path)


def _config(root: Path) -> WorkflowConfig:
    return WorkflowConfig(
        root=root,
        values={
            "paths": {
                "raw_data": "data/raw/lisa",
                "processed_data": "data/processed/lisa_yolo",
                "output": "out",
            },
            "dataset": {
                "materialization": "copy",
                "overwrite": True,
                "split_seed": 42,
            },
            "tiling": {
                "size": 640,
                "overlap_ratio": 0.2,
                "min_box_area_ratio": 0.5,
                "max_empty_to_positive_ratio": 0.25,
                "jpeg_quality": 95,
            },
        },
    )


def test_project_config_and_class_metadata() -> None:
    config = load_config(".config/config.toml")
    assert config.path("output").name == "out"
    assert config.section("training")["model"] == "yolo26l.pt"
    assert config.section("training")["image_size"] == 640
    assert config.section("training")["batch"] == -1
    assert config.section("training")["mosaic"] == 0.0
    assert config.section("evaluation")["batch"] == 1
    assert CLASS_METADATA["stop"] == {"color": "red"}
    assert MODEL_CLASSES == ["go", "warning", "stop"]
    assert CLASS_REMAP["stopLeft"] == "stop"
    assert resolve_device("cpu") == "cpu"


def test_training_metrics_csv_is_preserved(tmp_path: Path) -> None:
    run_dir = tmp_path / "out" / "training" / "test_run"
    run_dir.mkdir(parents=True)
    results = run_dir / "results.csv"
    contents = "epoch,train/box_loss,metrics/mAP50(B)\n1,0.5,0.75\n"
    results.write_text(contents, encoding="utf-8")

    metrics_csv = _save_epoch_metrics_csv(run_dir)

    assert metrics_csv == run_dir / "epoch_metrics.csv"
    assert metrics_csv.read_text(encoding="utf-8") == contents


def test_config_rejects_invalid_ratios(tmp_path: Path) -> None:
    config = tmp_path / ".config" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        """
[paths]
raw_data = "raw"
processed_data = "processed"
output = "out"
[dataset]
train_ratio = 0.8
validation_ratio = 0.15
test_ratio = 0.15
[tiling]
size = 640
overlap_ratio = 0.2
min_box_area_ratio = 0.5
max_empty_to_positive_ratio = 0.25
[training]
[evaluation]
[inference]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_config(config)


def test_read_convert_and_grouped_split(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "lisa"
    annotation_file = raw / "Annotations" / "dayTrain" / "frameAnnotationsBOX.csv"
    annotation_file.parent.mkdir(parents=True)
    header = [
        "Filename",
        "Annotation tag",
        "Upper left corner X",
        "Upper left corner Y",
        "Lower right corner X",
        "Lower right corner Y",
        "Origin file",
        "Origin frame number",
        "Origin track",
        "Origin track frame number",
    ]
    rows = []
    classes = ("goLeft", "warningLeft", "stopLeft")
    for index, (group, class_name) in enumerate(
        zip(("dayClip1", "dayClip2", "dayClip3"), classes, strict=True), start=1
    ):
        relative = Path("dayTrain") / group / "frames" / f"{group}--00001.jpg"
        _make_image(raw / relative)
        rows.append(
            [str(relative), class_name, "10", "8", "30", "48", f"{group}.avi", "1", "1", "1"]
        )
    with annotation_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)

    frames, stats = read_lisa_frames(raw)
    assert stats["images"] == 3
    assert all(len(frame.boxes) == 1 for frame in frames)
    assert {frame.boxes[0].class_name for frame in frames} == {"go", "warning", "stop"}
    assert {frame.group for frame in frames} == {"dayclip1", "dayclip2", "dayclip3"}

    splits, assignment = split_frames(frames, (0.7, 0.15, 0.15), seed=42, attempts=100)
    assert all(splits.values())
    assert set(assignment) == {"dayclip1", "dayclip2", "dayclip3"}
    assert sum(len(items) for items in splits.values()) == 3

    summary = write_yolo_dataset(_config(tmp_path), splits, assignment, stats)
    processed = tmp_path / "data" / "processed" / "lisa_yolo"
    assert (processed / "dataset.yaml").is_file()
    assert (processed / "manifest.csv").is_file()
    assert sum(item["selected_tiles"] for item in summary["splits"].values()) == 3
    metadata = json.loads((processed / "class_metadata.json").read_text())
    assert metadata["0"]["color"] == "green"
    assert len(metadata) == 3

    label_files = list((processed / "labels").rglob("*.txt"))
    values = label_files[0].read_text().split()
    assert values[0] in {"0", "1", "2"}
    assert [float(value) for value in values[1:]] == pytest.approx(
        [20 / 640, 28 / 640, 20 / 640, 40 / 640]
    )

    cache = processed / "labels" / "val.cache"
    cache.write_bytes(b"stale cache")
    validation = validate_yolo_dataset(processed, clear_caches=True)
    assert validation["label_files"] == 3
    assert validation["instances"] == 3
    assert validation["removed_caches"] == 1
    assert not cache.exists()


def test_dataset_validation_rejects_old_class_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "processed"
    labels = dataset / "labels" / "val"
    labels.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: go\n  1: warning\n  2: stop\n",
        encoding="utf-8",
    )
    bad_label = labels / "old.txt"
    bad_label.write_text("5 0.5 0.5 0.1 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid class ID '5'"):
        validate_yolo_dataset(dataset, clear_caches=True)


def test_tiling_covers_edges_and_drops_small_fragments(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    _make_image(image, (1280, 960))
    frame = Frame(
        source=image,
        group="dayclip1",
        width=1280,
        height=960,
        boxes=[Box("stop", 635, 100, 675, 180)],
    )

    tiles = tile_frame(frame, size=640, overlap_ratio=0.2, min_box_area_ratio=0.5)

    assert {(tile.x, tile.y) for tile in tiles} == {
        (0, 0),
        (512, 0),
        (640, 0),
        (0, 320),
        (512, 320),
        (640, 320),
    }
    assert any(tile.boxes and tile.boxes[0].x1 == 123 for tile in tiles)
    assert any(tile.intersects_ignored_box for tile in tiles)


def test_group_never_crosses_splits(tmp_path: Path) -> None:
    frames = []
    for group_index in range(8):
        image = tmp_path / f"group{group_index}.jpg"
        _make_image(image)
        for _ in range(group_index + 1):
            frames.append(
                Frame(
                    source=image,
                    group=f"video{group_index}",
                    width=100,
                    height=80,
                    boxes=[Box("stop", 10, 10, 20, 30)],
                )
            )

    splits, _ = split_frames(frames, (0.7, 0.15, 0.15), seed=7, attempts=500)
    seen: dict[str, str] = {}
    for split, split_frames_list in splits.items():
        for frame in split_frames_list:
            assert frame.group not in seen or seen[frame.group] == split
            seen[frame.group] = split


def test_prepare_reuses_existing_raw_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "data" / "raw" / "lisa"
    annotation_file = raw / "Annotations" / "dayTrain" / "frameAnnotationsBOX.csv"
    annotation_file.parent.mkdir(parents=True)
    header = [
        "Filename",
        "Annotation tag",
        "Upper left corner X",
        "Upper left corner Y",
        "Lower right corner X",
        "Lower right corner Y",
        "Origin file",
    ]
    rows = []
    for group in ("dayClip1", "dayClip2", "dayClip3"):
        relative = Path("dayTrain") / group / "frames" / f"{group}--00001.jpg"
        _make_image(raw / relative)
        rows.append([str(relative), "stopLeft", "10", "8", "30", "48", f"{group}.avi"])
    with annotation_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)

    config = tmp_path / ".config" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        """
[paths]
raw_data = "data/raw/lisa"
processed_data = "data/processed/tiled"
output = "out"
[dataset]
kaggle_handle = "example/dataset"
annotation_kind = "BOX"
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15
split_seed = 42
split_attempts = 100
materialization = "copy"
overwrite = true
[tiling]
size = 640
overlap_ratio = 0.2
min_box_area_ratio = 0.5
max_empty_to_positive_ratio = 0.25
jpeg_quality = 95
inference_batch = 1
perform_standard_prediction = true
postprocess_type = "GREEDYNMM"
postprocess_match_metric = "IOU"
postprocess_match_threshold = 0.5
[training]
image_size = 640
[evaluation]
[inference]
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "traffic_light_prediction.data.download_dataset",
        lambda *_args, **_kwargs: pytest.fail("download should not be called"),
    )
    summary = prepare_dataset(config)

    assert summary["source"]["raw_data_action"] == "reused"
    assert (tmp_path / "data/processed/tiled/dataset.yaml").is_file()


def test_stitched_prediction_record_uses_full_frame_coordinates() -> None:
    class FakeBbox:
        @staticmethod
        def to_xyxy() -> list[float]:
            return [640.0, 100.0, 680.0, 180.0]

    result = SimpleNamespace(
        object_prediction_list=[
            SimpleNamespace(
                category=SimpleNamespace(name="stop"),
                score=SimpleNamespace(value=0.9),
                bbox=FakeBbox(),
            )
        ]
    )

    record = _prediction_record(
        result, source="frame.jpg", width=1280, height=960, frame_index=7
    )

    assert record["frame_index"] == 7
    assert record["predictions"][0]["class_id"] == 2
    assert record["predictions"][0]["color"] == "red"
    assert record["predictions"][0]["box_xyxy_normalized"] == pytest.approx(
        [0.5, 100 / 960, 680 / 1280, 180 / 960]
    )


def test_full_frame_matching_uses_class_and_iou() -> None:
    import torch

    matched = _match_predictions(
        torch.tensor([[0.9, 0.8]]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.5, 0.95]),
    )

    assert matched.tolist() == [[True, False], [False, False]]
