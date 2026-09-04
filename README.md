# Traffic-light prediction

[License: CC BY-NC-SA 4.0](LICENSE)

An end-to-end YOLO11 workflow for detecting small LISA traffic lights and classifying their state as `go`, `warning`, or `stop`. It uses overlapping image tiles during training and inference so distant lights are not lost when full-resolution frames are resized.

The default model is `yolo11n.pt` at 640 px. Runtime settings live in [`.config/config.toml`](.config/config.toml); generated data and model artifacts stay in `data/` and `out/`.

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

### NVIDIA CUDA on Windows

The cross-platform lockfile may install the CPU-only PyTorch build on Windows. After `uv sync`, replace it with the CUDA build selected for the installed NVIDIA driver:

```powershell
uv pip install --reinstall torch torchvision --torch-backend=auto
```

Verify that PyTorch can access the GPU:

```powershell
uv run --no-sync python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

The command should report a CUDA version, `True`, and the NVIDIA GPU name. Run every subsequent workflow command with `--no-sync` after this override so UV does not restore the CPU packages from the lockfile:

```powershell
uv run --no-sync traffic-light train
```

Running `uv sync` or a plain `uv run` may replace the CUDA build, in which case repeat the CUDA installation command. Dataset download, extraction, and preparation use the CPU and disk; CUDA accelerates training and evaluation.

## Run the workflow

### 1. Prepare the data

```shell
uv run traffic-light data-prep
```

This reuses a valid dataset already under `data/raw/lisa`, or downloads [`mbornoe/lisa-traffic-light-dataset`](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset) when it is missing. It assigns complete source videos to splits targeting 70% train, 15% validation, and 15% test before slicing frames into overlapping 640×640 tiles. Whole videos are kept together to prevent neighboring frames leaking into evaluation, so the exact percentages depend on video sizes.

Normal preparation automatically avoids a second download. For strict offline operation, require the raw files to be present:

```shell
uv run traffic-light data-prep --skip-download
```

Every preparation run deterministically rebuilds the split and tiles from the configured seed. Change `split_seed` for a different reproducible video-grouped split, or use `--force-download` to explicitly refresh the raw dataset.

The new dataset is written to `data/processed/lisa_yolo_tiled_3class`, leaving older processed datasets untouched. It contains tiled YOLO images and labels, `manifest.csv` with source/tile coordinates, `class_metadata.json`, `preparation_summary.json`, and full-frame COCO test annotations. All positive tiles are kept; truly empty tiles are sampled to limit training time. With the default split this produces about 132,000 training tiles, so an epoch will take longer than full-frame training even though each input is smaller.

### 2. Train

```shell
uv run traffic-light train
```

The best checkpoint is saved to `out/training/<run_name>/weights/best.pt`. Per-epoch losses, precision, recall, mAP, and learning rates are saved to `out/training/<run_name>/epoch_metrics.csv`. With `device = "auto"`, training uses NVIDIA CUDA when available, Apple MPS on Apple Silicon, and otherwise CPU.

Before Ultralytics starts, the training command validates every generated label against the three configured classes and removes its cached label indexes. This prevents an older seven-class `labels/*.cache` file from silently contaminating a new three-class run. The configured AdamW learning rate and bias warmup are deliberately conservative for fine-tuning; do not remove them when changing model size.

Training can be stopped with `Ctrl+C` and resumed from the last completed epoch using Ultralytics' `last.pt` checkpoint:

```shell
uv run --no-sync yolo train resume model=out/training/<run_name>/weights/last.pt
```

Replace `<run_name>` with the configured training run name, `lisa_yolo11n_tiled_640_3class`. Running `traffic-light train` again starts training from the configured base model instead of resuming the interrupted run.

### 3. Evaluate and infer

```shell
uv run traffic-light evaluate
```

This reports the same Ultralytics precision, recall, mAP, per-class summaries, PR/F1/P/R curves, and confusion matrices for both held-out tiles and stitched original frames. It also reports COCO small-object metrics for the full frames. Inference slices each full-resolution input, merges overlapping predictions, and saves annotated images/videos plus structured JSON under `out/inference`.

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

LISA's arrow variants are merged into three state classes:

| Model class | LISA source labels | Color |
|---|---|---|
| `go` | `go`, `goForward`, `goLeft` | green |
| `warning` | `warning`, `warningLeft` | yellow |
| `stop` | `stop`, `stopLeft` | red |

The model intentionally does not predict arrow direction.

## Configuration

Edit `.config/config.toml` to change:

- data and output locations;
- dataset split ratios, seed, search attempts, and copy/link behavior;
- tile size, overlap, retained-box threshold, negative sampling, and merge behavior;
- YOLO model size (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, etc.) and optimizer;
- image size, epochs, batch size, workers, patience, cache, and device;
- evaluation checkpoint/metric threshold and inference source/confidence settings.

On a Windows NVIDIA system, install a current NVIDIA driver. The PyTorch dependency selected by UV detects CUDA at runtime. All Python entry points use the Windows-safe `__main__` guard.

## Tests

```shell
uv run pytest
```

Tests use a tiny synthetic LISA-shaped dataset; they do not download the full dataset, generate all production tiles, or train a model.

## Credits

The [LISA Traffic Light Dataset](https://vbn.aau.dk/en/datasets/lisa-traffic-light-dataset/) was created by Morten Bornø Jensen and Mark Philip Philipsen, under the supervision of Andreas Møgelmose, Thomas B. Moeslund, and Mohan M. Trivedi. This project downloads the dataset through [Morten Bornø Jensen's Kaggle distribution](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset).

When using the dataset or derived model weights, please cite M. B. Jensen, M. P. Philipsen, A. Møgelmose, T. B. Moeslund, and M. M. Trivedi, “[Vision for Looking at Traffic Lights: Issues, Survey, and Perspectives](https://doi.org/10.1109/TITS.2015.2509509),” *IEEE Transactions on Intelligent Transportation Systems*, vol. 17, no. 7, pp. 1800–1815, 2016.

## License

This repository, including its distributed dataset-derived model weights, is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE), matching the license identified by the LISA Kaggle distribution. Reuse requires attribution, is limited to non-commercial purposes, and must be shared under the same license. Third-party components retain their own licenses; Ultralytics is distributed under its AGPL/commercial licensing terms.
