from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from traffic_light_prediction.classes import CLASS_METADATA, LISA_CLASSES
from traffic_light_prediction.config import WorkflowConfig, load_config, resolve_device
from traffic_light_prediction.data import Box, Frame, read_lisa_frames, split_frames, write_yolo_dataset


def _make_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="gray").save(path)


def _config(root: Path) -> WorkflowConfig:
    return WorkflowConfig(
        root=root,
        values={
            "paths": {
                "raw_data": ".data/raw/lisa",
                "processed_data": ".data/processed/lisa_yolo",
                "output": ".out",
            },
            "dataset": {"materialization": "copy", "overwrite": True},
        },
    )


def test_project_config_and_class_metadata() -> None:
    config = load_config(".config/config.toml")
    assert config.section("training")["model"] == "yolo11s.pt"
    assert config.section("training")["image_size"] == 640
    assert CLASS_METADATA["stopLeft"] == {"color": "red", "direction": "left"}
    assert len(LISA_CLASSES) == 7
    assert resolve_device("cpu") == "cpu"


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
[training]
[evaluation]
[inference]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_config(config)


def test_read_convert_and_grouped_split(tmp_path: Path) -> None:
    raw = tmp_path / ".data" / "raw" / "lisa"
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
    for index, group in enumerate(("dayClip1", "dayClip2", "dayClip3"), start=1):
        relative = Path("dayTrain") / group / "frames" / f"{group}--00001.jpg"
        _make_image(raw / relative)
        rows.append(
            [str(relative), "go", "10", "8", "30", "48", f"{group}.avi", "1", "1", "1"]
        )
    with annotation_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)

    frames, stats = read_lisa_frames(raw)
    assert stats["images"] == 3
    assert all(len(frame.boxes) == 1 for frame in frames)
    assert {frame.group for frame in frames} == {"dayclip1", "dayclip2", "dayclip3"}

    splits, assignment = split_frames(frames, (0.7, 0.15, 0.15), seed=42, attempts=100)
    assert all(splits.values())
    assert set(assignment) == {"dayclip1", "dayclip2", "dayclip3"}
    assert sum(len(items) for items in splits.values()) == 3

    summary = write_yolo_dataset(_config(tmp_path), splits, assignment, stats)
    processed = tmp_path / ".data" / "processed" / "lisa_yolo"
    assert (processed / "dataset.yaml").is_file()
    assert (processed / "manifest.csv").is_file()
    assert sum(item["images"] for item in summary["splits"].values()) == 3
    metadata = json.loads((processed / "class_metadata.json").read_text())
    assert metadata["0"]["color"] == "green"

    label_files = list((processed / "labels").rglob("*.txt"))
    values = label_files[0].read_text().split()
    assert values[0] == "0"
    assert [float(value) for value in values[1:]] == pytest.approx([0.2, 0.35, 0.2, 0.5])


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
