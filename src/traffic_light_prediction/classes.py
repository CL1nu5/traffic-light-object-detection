"""LISA source labels and the reduced model class definitions."""

from __future__ import annotations

LISA_CLASSES = [
    "go",
    "goForward",
    "goLeft",
    "warning",
    "warningLeft",
    "stop",
    "stopLeft",
]

MODEL_CLASSES = ["go", "warning", "stop"]

CLASS_REMAP = {
    "go": "go",
    "goForward": "go",
    "goLeft": "go",
    "warning": "warning",
    "warningLeft": "warning",
    "stop": "stop",
    "stopLeft": "stop",
}

CLASS_METADATA = {
    "go": {"color": "green"},
    "warning": {"color": "yellow"},
    "stop": {"color": "red"},
}

CLASS_TO_ID = {name: index for index, name in enumerate(MODEL_CLASSES)}
