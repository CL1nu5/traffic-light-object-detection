"""Download LISA and convert its annotations to a grouped YOLO dataset."""

from __future__ import annotations

import csv
import json
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

from .classes import CLASS_METADATA, CLASS_TO_ID, LISA_CLASSES
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


def read_lisa_frames(raw_dir: Path, annotation_kind: str = "BOX") -> tuple[list[Frame], dict[str, int]]:
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
                if class_name not in CLASS_TO_ID:
                    raise ValueError(f"Unknown LISA class {class_name!r} in {annotation_file}")
                try:
                    box = Box(
                        class_name=class_name,
                        x1=float(row["upperleftcornerx"]),
                        y1=float(row["upperleftcornery"]),
                        x2=float(row["lowerrightcornerx"]),
                        y2=float(row["lowerrightcornery"]),
                    )
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Malformed annotation row in {annotation_file}: {source_row}") from exc

                annotations[image].append(box)
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
    return frames, dict(stats)


def _assignment_score(
    assignment: dict[str, str],
    group_frames: dict[str, list[Frame]],
    targets: dict[str, float],
) -> float:
    split_images: Counter[str] = Counter()
    split_classes: dict[str, Counter[str]] = defaultdict(Counter)
    class_groups: dict[str, set[str]] = defaultdict(set)

    for group, frames in group_frames.items():
        split = assignment[group]
        split_images[split] += len(frames)
        for frame in frames:
            split_classes[split].update(box.class_name for box in frame.boxes)
            for box in frame.boxes:
                class_groups[box.class_name].add(group)

    total_images = sum(split_images.values())
    score = 10.0 * sum(
        abs(split_images[split] / total_images - target)
        for split, target in targets.items()
    )

    for class_name in LISA_CLASSES:
        total = sum(split_classes[split][class_name] for split in targets)
        if total:
            score += sum(
                abs(split_classes[split][class_name] / total - target)
                for split, target in targets.items()
            )
            if len(class_groups[class_name]) >= len(targets):
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

    for _ in range(max(1, attempts)):
        assignment = {
            group: rng.choices(split_names, weights=weights, k=1)[0] for group in groups
        }
        if set(assignment.values()) != set(split_names):
            continue
        score = _assignment_score(assignment, group_frames, targets)
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


def write_yolo_dataset(
    config: WorkflowConfig,
    splits: dict[str, list[Frame]],
    assignment: dict[str, str],
    source_stats: dict[str, int],
) -> dict[str, object]:
    """Materialize YOLO images, labels, metadata, and the dataset YAML."""

    output_dir = config.path("processed_data")
    dataset_config = config.section("dataset")
    _safe_reset_directory(output_dir, config.root, bool(dataset_config["overwrite"]))

    manifest_rows: list[dict[str, object]] = []
    split_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split, frames in splits.items():
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        used_names: set[str] = set()

        for frame in frames:
            output_name = _output_name(frame)
            if output_name in used_names:
                raise ValueError(f"Duplicate output image name: {output_name}")
            used_names.add(output_name)
            output_image = image_dir / output_name
            _materialize(frame.source, output_image, str(dataset_config["materialization"]))

            label_path = label_dir / f"{Path(output_name).stem}.txt"
            lines = [_yolo_line(box, frame.width, frame.height) for box in frame.boxes]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_class_counts[split].update(box.class_name for box in frame.boxes)
            manifest_rows.append(
                {
                    "split": split,
                    "group": frame.group,
                    "source": str(frame.source),
                    "image": str(output_image.relative_to(output_dir)),
                    "objects": len(frame.boxes),
                }
            )

    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(LISA_CLASSES)],
        "",
    ]
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    class_metadata = {
        str(index): {"name": name, **CLASS_METADATA[name]}
        for index, name in enumerate(LISA_CLASSES)
    }
    (output_dir / "class_metadata.json").write_text(
        json.dumps(class_metadata, indent=2) + "\n", encoding="utf-8"
    )

    total_images = sum(len(frames) for frames in splits.values())
    summary: dict[str, object] = {
        "source": source_stats,
        "splits": {
            split: {
                "images": len(frames),
                "ratio": len(frames) / total_images,
                "groups": sorted(group for group, assigned in assignment.items() if assigned == split),
                "class_instances": dict(split_class_counts[split]),
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
    if download:
        download_dataset(config, force=force_download)
    raw_dir = config.path("raw_data")
    processed_dir = config.path("processed_data")
    if processed_dir == raw_dir or processed_dir in raw_dir.parents:
        raise ValueError(
            "paths.processed_data must not equal or contain paths.raw_data; "
            "preparation replaces the processed directory"
        )
    dataset_config = config.section("dataset")
    frames, stats = read_lisa_frames(raw_dir, str(dataset_config["annotation_kind"]))
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
    return write_yolo_dataset(config, splits, assignment, stats)
