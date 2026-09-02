# Traffic-light prediction

An end-to-end YOLO11 workflow for detecting LISA traffic lights and classifying their color and arrow direction. It uses three stages: data preparation, training, and evaluation/inference.

The default model is `yolo11s.pt` at 640 px. Runtime settings live in [`.config/config.toml`](.config/config.toml); generated data and model artifacts stay in `data/` and `out/`.

## Setup

Install [UV](https://docs.astral.sh/uv/getting-started/installation/), then create the environment:

```shell
uv sync
```

Create `.env` in the project root with the API token from your Kaggle account settings:

```dotenv
KAGGLE_API_KEY=KGAT_your_token_here
```

The token is loaded only for the download request and is never printed or copied into project output. `.env` is ignored by Git.

## Run the workflow

### 1. Prepare the data

```shell
uv run traffic-light data-prep
```

This downloads [`mbornoe/lisa-traffic-light-dataset`](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset), uses the `frameAnnotationsBOX.csv` annotations, creates YOLO labels, and assigns complete source videos to splits targeting 70% train, 15% validation, and 15% test. Whole videos are kept together to prevent neighboring frames leaking into evaluation, so the exact percentages depend on video sizes.

If the files are already present in `data/raw/lisa`, conversion can be rerun without downloading:

```shell
uv run traffic-light data-prep --skip-download
```

The prepared dataset contains `dataset.yaml`, images and labels for each split, `manifest.csv`, `class_metadata.json`, and `preparation_summary.json`.

### 2. Train

```shell
uv run traffic-light train
```

The best checkpoint is saved to `out/training/<run_name>/weights/best.pt`. Per-epoch losses, precision, recall, mAP, and learning rates are saved to `out/training/<run_name>/epoch_metrics.csv`. With `device = "auto"`, training uses NVIDIA CUDA when available, Apple MPS on Apple Silicon, and otherwise CPU.

Training can be stopped with `Ctrl+C` and resumed from the last completed epoch using Ultralytics' `last.pt` checkpoint:

```shell
uv run yolo train resume model=out/training/<run_name>/weights/last.pt
```

Replace `<run_name>` with the configured training run name, such as `lisa_yolo11n_640`. Running `uv run traffic-light train` again starts training from the configured base model instead of resuming the interrupted run.

### 3. Evaluate and infer

```shell
uv run traffic-light evaluate
```

This evaluates only on the held-out test split and runs inference on sample test images. Metrics and plots go to `out/evaluation`; annotated images and structured predictions go to `out/inference`. Each JSON prediction includes bounding-box coordinates, confidence, original LISA class, color, and direction.

To infer on another image, directory, or video:

```shell
uv run traffic-light evaluate --source path/to/input
```

## Notebooks

The notebooks are intentionally thin and call the same tested Python modules as the CLI:

- `notebooks/01_data_preparation.ipynb`
- `notebooks/02_training.ipynb`
- `notebooks/03_evaluation_and_inference.ipynb`

Start Jupyter with the UV environment:

```shell
uv run jupyter lab
```

## Classes

LISA supplies seven combined state/direction classes:

| LISA class | Color | Direction |
|---|---|---|
| `go` | green | general |
| `goForward` | green | forward |
| `goLeft` | green | left |
| `warning` | yellow | general |
| `warningLeft` | yellow | left |
| `stop` | red | general |
| `stopLeft` | red | left |

The dataset has no right-arrow examples, so this model cannot learn right-arrow recognition without additional annotated data.

## Configuration

Edit `.config/config.toml` to change:

- data and output locations;
- dataset split ratios, seed, search attempts, and copy/link behavior;
- YOLO model size (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, etc.);
- image size, epochs, batch size, workers, patience, cache, and device;
- evaluation checkpoint and inference source/confidence/IoU settings.

On a Windows NVIDIA system, install a current NVIDIA driver. The PyTorch dependency selected by UV detects CUDA at runtime. All Python entry points use the Windows-safe `__main__` guard.

## Tests

```shell
uv run pytest
```

Tests use a tiny synthetic LISA-shaped dataset; they do not download the full dataset or train a model.

## Dataset license

The Kaggle mirror identifies LISA as **CC BY-NC-SA 4.0**. Review its attribution, non-commercial, and share-alike requirements before distributing data, trained weights, or derived datasets. Ultralytics is distributed under its own AGPL/commercial licensing terms.
