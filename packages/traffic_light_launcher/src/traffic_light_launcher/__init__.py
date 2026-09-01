"""Launch the repository workflow without relying on editable `.pth` files."""

from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "traffic_light_prediction" / "cli.py").is_file():
            return candidate
    raise RuntimeError("Run traffic-light from the project directory or one of its subdirectories")


def main() -> None:
    source_root = _project_root() / "src"
    sys.path.insert(0, str(source_root))

    from traffic_light_prediction.cli import main as workflow_main

    workflow_main()
