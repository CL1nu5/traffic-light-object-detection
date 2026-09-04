"""Download LISA and convert its annotations to a grouped YOLO dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from PIL import Image

from .classes import (
    CLASS_METADATA,
    CLASS_REMAP,
    CLASS_TO_ID,
    LISA_CLASSES,
    MODEL_CLASSES,
)
from .config import WorkflowConfig, load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CANONICAL_ROOTS = {
    "daytrain",
    "nighttrain",
    "daysequence1",
    "daysequence2",
    "nightsequence1",
    "nightsequence2",
}
GROUP_PATTERN = re.compile(r"(?:day|night)(?:clip|sequence)\d+", re.IGNORECASE)


@dataclass(frozen=True)
class Box:
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Frame:
    source: Path
    group: str
    width: int
    height: int
    boxes: list[Box]


@dataclass(frozen=True)
class Tile:
    frame: Frame
    x: int
    y: int
    size: int
    boxes: tuple[Box, ...]
    intersects_ignored_box: bool = False


def download_dataset(config: WorkflowConfig, force: bool = False) -> Path:
    """Download and extract LISA with the API token stored in `.env`."""

    load_dotenv(config.root / ".env")
    token = os.getenv("KAGGLE_API_KEY", "").strip()
    if not token:
        raise RuntimeError("KAGGLE_API_KEY is missing from .env")

    import kagglehub

    kagglehub.config.set_kaggle_api_token(token)
    raw_dir = config.path("raw_data")
    raw_dir.mkdir(parents=True, exist_ok=True)
    result = kagglehub.dataset_download(
        config.section("dataset")["kaggle_handle"],
        output_dir=str(raw_dir),
        force_download=force,
    )
    return Path(result).resolve()


def _is_canonical_image(path: Path, raw_dir: Path) -> bool:
    relative_parts = [part.casefold() for part in path.relative_to(raw_dir).parts]
    if any(part.startswith("sample-") for part in relative_parts):
        return False
    return any(part in CANONICAL_ROOTS for part in relative_parts)


def _discover_images(raw_dir: Path) -> list[Path]:
    images = [
        path.resolve()
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in IMAGE_SUFFIXES
        and _is_canonical_image(path, raw_dir)
    ]
    if not images:
        raise FileNotFoundError(f"No canonical LISA images found below {raw_dir}")
    return sorted(images)


def _build_suffix_index(images: Iterable[Path], raw_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        parts = image.relative_to(raw_dir).parts
        for start in range(len(parts)):
            suffix = "/".join(parts[start:]).casefold()
            index[suffix].append(image)
    return index


def _resolve_image(filename: str, suffix_index: dict[str, list[Path]]) -> Path | None:
    normalized = filename.strip().replace("\\", "/").lstrip("./")
    parts = tuple(part for part in normalized.split("/") if part)
    for start in range(len(parts)):
        matches = suffix_index.get("/".join(parts[start:]).casefold(), [])
        if len(matches) == 1:
            return matches[0]
    return None


def _normalized_row(row: dict[str, str]) -> dict[str, str]:
    return {re.sub(r"[^a-z0-9]", "", key.casefold()): value for key, value in row.items()}


def _find_annotation_files(raw_dir: Path, annotation_kind: str) -> list[Path]:
    expected = f"frameannotations{annotation_kind}.csv".casefold()
    files = [
        path
        for path in raw_dir.rglob("*.csv")
        if path.name.casefold() == expected
        and not any(part.casefold().startswith("sample-") for part in path.parts)
    ]
    if not files:
        raise FileNotFoundError(f"No frameAnnotations{annotation_kind}.csv files found")
    return sorted(files)


def _group_from_path(path: Path, origin: str = "") -> str:
    for value in (*reversed(path.parts), origin):
        match = GROUP_PATTERN.search(value)
        if match:
            return match.group(0).casefold()
    raise ValueError(f"Could not derive source video group for {path}")


def read_lisa_frames(
    raw_dir: Path, annotation_kind: str = "BOX"
) -> tuple[list[Frame], dict[str, object]]:
    """Read LISA annotations and return every canonical frame, including negatives."""

    raw_dir = raw_dir.resolve()
    images = _discover_images(raw_dir)
    suffix_index = _build_suffix_index(images, raw_dir)
    annotations: dict[Path, list[Box]] = defaultdict(list)
    annotation_groups: dict[Path, str] = {}
    stats: Counter[str] = Counter()

    for annotation_file in _find_annotation_files(raw_dir, annotation_kind):
        with annotation_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for source_row in reader:
                row = _normalized_row(source_row)
                image = _resolve_image(row.get("filename", ""), suffix_index)
                if image is None:
                    stats["unresolved_rows"] += 1
                    continue

                class_name = row.get("annotationtag", "").strip()
                if class_name not in CLASS_REMAP:
                    raise ValueError(f"Unknown LISA class {class_name!r} in {annotation_file}")
                try:
                    box = Box(
                        class_name=CLASS_REMAP[class_name],
                        x1=float(row["upperleftcornerx"]),
                        y1=float(row["upperleftcornery"]),
                        x2=float(row["lowerrightcornerx"]),
                        y2=float(row["lowerrightcornery"]),
                    )
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Malformed annotation row in {annotation_file}: {source_row}") from exc

                annotations[image].append(box)
                stats[f"source_class::{class_name}"] += 1
                annotation_groups[image] = _group_from_path(
                    image, row.get("originfile", "")
                )
                stats["annotation_rows"] += 1

    frames: list[Frame] = []
    for image in images:
        with Image.open(image) as opened:
            width, height = opened.size

        valid_boxes: list[Box] = []
        for box in annotations.get(image, []):
            clipped = Box(
                class_name=box.class_name,
                x1=max(0.0, min(float(width), box.x1)),
                y1=max(0.0, min(float(height), box.y1)),
                x2=max(0.0, min(float(width), box.x2)),
                y2=max(0.0, min(float(height), box.y2)),
            )
            if clipped.x2 <= clipped.x1 or clipped.y2 <= clipped.y1:
                stats["invalid_boxes"] += 1
                continue
            valid_boxes.append(clipped)

        frames.append(
            Frame(
                source=image,
                group=annotation_groups.get(image, _group_from_path(image)),
                width=width,
                height=height,
                boxes=valid_boxes,
            )
        )

    stats["images"] = len(frames)
    stats["negative_images"] = sum(not frame.boxes for frame in frames)
    if stats["unresolved_rows"]:
        raise ValueError(
            f"Could not resolve {stats['unresolved_rows']} annotation rows to images; "
            "the downloaded dataset layout may have changed"
        )
    result: dict[str, object] = {
        key: value for key, value in stats.items() if not key.startswith("source_class::")
    }
    result["source_class_instances"] = {
        name: stats[f"source_class::{name}"] for name in LISA_CLASSES
    }
    return frames, result


def _assignment_score(
    assignment: dict[str, str],
    group_image_counts: dict[str, int],
    group_class_counts: dict[str, Counter[str]],
    class_group_counts: dict[str, int],
    targets: dict[str, float],
) -> float:
    split_images: Counter[str] = Counter()
    split_classes: dict[str, Counter[str]] = defaultdict(Counter)

    for group, image_count in group_image_counts.items():
        split = assignment[group]
        split_images[split] += image_count
        split_classes[split].update(group_class_counts[group])

    total_images = sum(split_images.values())
    score = 10.0 * sum(
        abs(split_images[split] / total_images - target)
        for split, target in targets.items()
    )

    for class_name in MODEL_CLASSES:
        total = sum(split_classes[split][class_name] for split in targets)
        if total:
            score += sum(
                abs(split_classes[split][class_name] / total - target)
                for split, target in targets.items()
            )
            if class_group_counts[class_name] >= len(targets):
                score += 20.0 * sum(
                    split_classes[split][class_name] == 0 for split in targets
                )
    return score


def split_frames(
    frames: list[Frame],
    ratios: tuple[float, float, float],
    seed: int,
    attempts: int,
) -> tuple[dict[str, list[Frame]], dict[str, str]]:
    """Find a deterministic video-grouped split close to the target ratios."""

    split_names = ("train", "val", "test")
    targets = dict(zip(split_names, ratios, strict=True))
    group_frames: dict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        group_frames[frame.group].append(frame)
    groups = sorted(group_frames)
    if len(groups) < 3:
        raise ValueError("At least three source videos are required for grouped splitting")

    rng = random.Random(seed)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    weights = [targets[name] for name in split_names]
    group_image_counts = {group: len(items) for group, items in group_frames.items()}
    group_class_counts = {
        group: Counter(box.class_name for frame in items for box in frame.boxes)
        for group, items in group_frames.items()
    }
    class_group_counts = {
        class_name: sum(counts[class_name] > 0 for counts in group_class_counts.values())
        for class_name in MODEL_CLASSES
    }

    for _ in range(max(1, attempts)):
        assignment = {
            group: rng.choices(split_names, weights=weights, k=1)[0] for group in groups
        }
        if set(assignment.values()) != set(split_names):
            continue
        score = _assignment_score(
            assignment,
            group_image_counts,
            group_class_counts,
            class_group_counts,
            targets,
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Unable to create non-empty grouped train/val/test splits")

    splits: dict[str, list[Frame]] = {name: [] for name in split_names}
    for group, split in best_assignment.items():
        splits[split].extend(group_frames[group])
    for split in splits:
        splits[split].sort(key=lambda frame: str(frame.source))
    return splits, best_assignment


def _safe_reset_directory(path: Path, root: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Processed dataset already exists: {path}")
        if path in {Path(path.anchor), root, root.parent}:
            raise ValueError(f"Refusing to remove unsafe processed-data path: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _output_name(frame: Frame) -> str:
    safe_group = re.sub(r"[^a-zA-Z0-9_-]", "_", frame.group)
    return f"{safe_group}__{frame.source.name}"


def _materialize(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    elif mode != "copy":
        raise ValueError("dataset.materialization must be 'hardlink' or 'copy'")
    shutil.copy2(source, destination)


def _yolo_line(box: Box, width: int, height: int) -> str:
    center_x = ((box.x1 + box.x2) / 2.0) / width
    center_y = ((box.y1 + box.y2) / 2.0) / height
    box_width = (box.x2 - box.x1) / width
    box_height = (box.y2 - box.y1) / height
    return (
        f"{CLASS_TO_ID[box.class_name]} {center_x:.8f} {center_y:.8f} "
        f"{box_width:.8f} {box_height:.8f}"
    )


def _tile_origins(length: int, size: int, overlap_ratio: float) -> list[int]:
    """Return deterministic, edge-anchored slice origins covering a dimension."""

    if length <= size:
        return [0]
    stride = max(1, round(size * (1.0 - overlap_ratio)))
    final_origin = length - size
    origins = list(range(0, final_origin + 1, stride))
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def tile_frame(
    frame: Frame,
    *,
    size: int,
    overlap_ratio: float,
    min_box_area_ratio: float,
) -> list[Tile]:
    """Slice a frame and translate sufficiently visible boxes into tile coordinates."""

    tiles: list[Tile] = []
    for y in _tile_origins(frame.height, size, overlap_ratio):
        for x in _tile_origins(frame.width, size, overlap_ratio):
            boxes: list[Box] = []
            intersects_ignored_box = False
            for box in frame.boxes:
                ix1 = max(box.x1, float(x))
                iy1 = max(box.y1, float(y))
                ix2 = min(box.x2, float(x + size))
                iy2 = min(box.y2, float(y + size))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                original_area = (box.x2 - box.x1) * (box.y2 - box.y1)
                retained_area = (ix2 - ix1) * (iy2 - iy1)
                if retained_area / original_area < min_box_area_ratio:
                    intersects_ignored_box = True
                    continue
                boxes.append(
                    Box(
                        class_name=box.class_name,
                        x1=ix1 - x,
                        y1=iy1 - y,
                        x2=ix2 - x,
                        y2=iy2 - y,
                    )
                )
            tiles.append(
                Tile(
                    frame=frame,
                    x=x,
                    y=y,
                    size=size,
                    boxes=tuple(boxes),
                    intersects_ignored_box=intersects_ignored_box,
                )
            )
    return tiles


def _tile_name(tile: Tile) -> str:
    source_stem = Path(_output_name(tile.frame)).stem
    return f"{source_stem}__x{tile.x:04d}_y{tile.y:04d}.jpg"


def _empty_tile_key(tile: Tile, split: str, seed: int) -> str:
    value = f"{seed}|{split}|{tile.frame.source}|{tile.x}|{tile.y}".encode()
    return hashlib.sha256(value).hexdigest()


def _select_tiles(
    tiles: list[Tile], *, split: str, seed: int, max_empty_ratio: float
) -> tuple[list[Tile], dict[str, int]]:
    positives = [tile for tile in tiles if tile.boxes]
    empty = [
        tile for tile in tiles if not tile.boxes and not tile.intersects_ignored_box
    ]
    ambiguous = len(tiles) - len(positives) - len(empty)
    empty_limit = min(len(empty), int(len(positives) * max_empty_ratio))
    selected_empty = sorted(
        empty, key=lambda tile: _empty_tile_key(tile, split, seed)
    )[:empty_limit]
    selected = sorted(
        [*positives, *selected_empty],
        key=lambda tile: (str(tile.frame.source), tile.y, tile.x),
    )
    return selected, {
        "candidate_tiles": len(tiles),
        "positive_tiles": len(positives),
        "available_empty_tiles": len(empty),
        "selected_empty_tiles": len(selected_empty),
        "dropped_ambiguous_tiles": ambiguous,
        "selected_tiles": len(selected),
    }


def _save_tile(opened: Image.Image, tile: Tile, destination: Path, quality: int) -> None:
    crop = opened.crop((tile.x, tile.y, tile.x + tile.size, tile.y + tile.size))
    if crop.size != (tile.size, tile.size):
        padded = Image.new("RGB", (tile.size, tile.size), color=(0, 0, 0))
        padded.paste(crop.convert("RGB"), (0, 0))
        crop = padded
    elif crop.mode != "RGB":
        crop = crop.convert("RGB")
    crop.save(destination, format="JPEG", quality=quality, subsampling=0)


def _write_full_test_artifacts(
    output_dir: Path, frames: list[Frame], materialization: str
) -> None:
    full_dir = output_dir / "full_images" / "test"
    full_dir.mkdir(parents=True)
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for image_id, frame in enumerate(frames, start=1):
        name = _output_name(frame)
        destination = full_dir / name
        _materialize(frame.source, destination, materialization)
        images.append(
            {
                "id": image_id,
                "file_name": destination.relative_to(output_dir).as_posix(),
                "width": frame.width,
                "height": frame.height,
                "group": frame.group,
            }
        )
        for box in frame.boxes:
            width = box.x2 - box.x1
            height = box.y2 - box.y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CLASS_TO_ID[box.class_name] + 1,
                    "bbox": [box.x1, box.y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    coco = {
        "info": {"description": "LISA full-frame three-class test split"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "traffic_light"}
            for index, name in enumerate(MODEL_CLASSES)
        ],
    }
    (output_dir / "full_test_coco.json").write_text(
        json.dumps(coco, indent=2) + "\n", encoding="utf-8"
    )


def validate_yolo_dataset(
    dataset_dir: Path, *, clear_caches: bool = False
) -> dict[str, int]:
    """Validate generated YOLO labels and optionally remove stale label caches."""

    dataset_dir = dataset_dir.resolve()
    dataset_yaml = dataset_dir / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {dataset_yaml}")

    declared_names = {
        int(match.group(1)): match.group(2).strip()
        for match in re.finditer(
            r"(?m)^\s{2}(\d+):\s*(.+?)\s*$",
            dataset_yaml.read_text(encoding="utf-8"),
        )
    }
    expected_names = dict(enumerate(MODEL_CLASSES))
    if declared_names != expected_names:
        raise ValueError(
            f"{dataset_yaml} declares classes {declared_names}, expected {expected_names}. "
            "Re-run data preparation with the current three-class configuration."
        )

    label_files = sorted((dataset_dir / "labels").glob("*/*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No YOLO labels found below {dataset_dir / 'labels'}")

    instances = 0
    for label_path in label_files:
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = line.split()
            location = f"{label_path}:{line_number}"
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label at {location}: expected 5 fields")
            try:
                values = [float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric YOLO label at {location}: {line!r}") from exc
            class_value, center_x, center_y, width, height = values
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite YOLO label at {location}: {line!r}")
            class_id = int(class_value)
            if class_value != class_id or class_id not in expected_names:
                raise ValueError(
                    f"Invalid class ID {fields[0]!r} at {location}; expected one of "
                    f"{sorted(expected_names)}. Re-run data preparation before training."
                )
            if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
                raise ValueError(f"Box center is outside the image at {location}: {line!r}")
            if not (0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Box size is invalid at {location}: {line!r}")
            instances += 1

    removed_caches = 0
    if clear_caches:
        for cache_path in (dataset_dir / "labels").glob("*.cache"):
            cache_path.unlink()
            removed_caches += 1

    return {
        "label_files": len(label_files),
        "instances": instances,
        "removed_caches": removed_caches,
    }


def write_yolo_dataset(
    config: WorkflowConfig,
    splits: dict[str, list[Frame]],
    assignment: dict[str, str],
    source_stats: dict[str, object],
    *,
    raw_data_action: str = "reused",
) -> dict[str, object]:
    """Materialize tiled YOLO images, labels, metadata, and full-frame test data."""

    output_dir = config.path("processed_data")
    dataset_config = config.section("dataset")
    tiling = config.section("tiling")
    _safe_reset_directory(output_dir, config.root, bool(dataset_config["overwrite"]))

    manifest_rows: list[dict[str, object]] = []
    split_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_tile_stats: dict[str, dict[str, int]] = {}
    for split, frames in splits.items():
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        all_tiles = [
            tile
            for frame in frames
            for tile in tile_frame(
                frame,
                size=int(tiling["size"]),
                overlap_ratio=float(tiling["overlap_ratio"]),
                min_box_area_ratio=float(tiling["min_box_area_ratio"]),
            )
        ]
        selected, tile_stats = _select_tiles(
            all_tiles,
            split=split,
            seed=int(dataset_config["split_seed"]),
            max_empty_ratio=float(tiling["max_empty_to_positive_ratio"]),
        )
        split_tile_stats[split] = tile_stats
        print(
            f"Preparing {split}: {len(frames)} source images -> "
            f"{len(selected)} selected tiles"
        )
        by_source: dict[Path, list[Tile]] = defaultdict(list)
        for tile in selected:
            by_source[tile.frame.source].append(tile)
        used_names: set[str] = set()
        written = 0
        for source, source_tiles in by_source.items():
            with Image.open(source) as opened:
                for tile in source_tiles:
                    output_name = _tile_name(tile)
                    if output_name in used_names:
                        raise ValueError(f"Duplicate output image name: {output_name}")
                    used_names.add(output_name)
                    output_image = image_dir / output_name
                    _save_tile(opened, tile, output_image, int(tiling["jpeg_quality"]))
                    label_path = label_dir / f"{Path(output_name).stem}.txt"
                    lines = [_yolo_line(box, tile.size, tile.size) for box in tile.boxes]
                    label_path.write_text(
                        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                    )
                    split_class_counts[split].update(box.class_name for box in tile.boxes)
                    manifest_rows.append(
                        {
                            "split": split,
                            "group": tile.frame.group,
                            "source": str(tile.frame.source),
                            "image": output_image.relative_to(output_dir).as_posix(),
                            "tile_x": tile.x,
                            "tile_y": tile.y,
                            "tile_size": tile.size,
                            "source_width": tile.frame.width,
                            "source_height": tile.frame.height,
                            "objects": len(tile.boxes),
                        }
                    )
                    written += 1
                    if written % 5000 == 0 or written == len(selected):
                        print(f"  {split}: wrote {written}/{len(selected)} tiles")

    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(MODEL_CLASSES)],
        "",
    ]
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    class_metadata = {
        str(index): {"name": name, **CLASS_METADATA[name]}
        for index, name in enumerate(MODEL_CLASSES)
    }
    (output_dir / "class_metadata.json").write_text(
        json.dumps(class_metadata, indent=2) + "\n", encoding="utf-8"
    )

    _write_full_test_artifacts(
        output_dir, splits["test"], str(dataset_config["materialization"])
    )
    total_source_images = sum(len(frames) for frames in splits.values())
    summary: dict[str, object] = {
        "source": {**source_stats, "raw_data_action": raw_data_action},
        "class_remap": CLASS_REMAP,
        "tiling": {
            "size": int(tiling["size"]),
            "overlap_ratio": float(tiling["overlap_ratio"]),
            "min_box_area_ratio": float(tiling["min_box_area_ratio"]),
            "max_empty_to_positive_ratio": float(tiling["max_empty_to_positive_ratio"]),
        },
        "splits": {
            split: {
                "source_images": len(frames),
                "source_ratio": len(frames) / total_source_images,
                "groups": sorted(group for group, assigned in assignment.items() if assigned == split),
                "class_instances": dict(split_class_counts[split]),
                **split_tile_stats[split],
            }
            for split, frames in splits.items()
        },
        "classes": class_metadata,
    }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def prepare_dataset(
    config_path: str | Path = ".config/config.toml",
    *,
    download: bool = True,
    force_download: bool = False,
) -> dict[str, object]:
    """Run the complete download, conversion, grouped split, and validation stage."""

    config = load_config(config_path)
    raw_dir = config.path("raw_data")
    dataset_config = config.section("dataset")
    annotation_kind = str(dataset_config["annotation_kind"])
    raw_data_action = "reused"
    try:
        _discover_images(raw_dir)
        _find_annotation_files(raw_dir, annotation_kind)
        raw_available = True
    except FileNotFoundError:
        raw_available = False

    if force_download:
        print(f"Force-downloading LISA into {raw_dir}")
        download_dataset(config, force=True)
        raw_data_action = "force_downloaded"
    elif not raw_available:
        if not download:
            raise FileNotFoundError(
                f"Complete LISA images and annotations were not found below {raw_dir}; "
                "rerun without --skip-download"
            )
        print(f"LISA data not found; downloading into {raw_dir}")
        download_dataset(config)
        raw_data_action = "downloaded"
    else:
        print(f"Reusing existing LISA data in {raw_dir}")

    processed_dir = config.path("processed_data")
    if processed_dir == raw_dir or processed_dir in raw_dir.parents:
        raise ValueError(
            "paths.processed_data must not equal or contain paths.raw_data; "
            "preparation replaces the processed directory"
        )
    frames, stats = read_lisa_frames(raw_dir, annotation_kind)
    print(f"Loaded {len(frames)} source frames; computing deterministic video split")
    splits, assignment = split_frames(
        frames,
        (
            float(dataset_config["train_ratio"]),
            float(dataset_config["validation_ratio"]),
            float(dataset_config["test_ratio"]),
        ),
        seed=int(dataset_config["split_seed"]),
        attempts=int(dataset_config["split_attempts"]),
    )
    return write_yolo_dataset(
        config, splits, assignment, stats, raw_data_action=raw_data_action
    )
