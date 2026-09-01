"""LISA class definitions and structured prediction metadata."""

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

CLASS_METADATA = {
    "go": {"color": "green", "direction": "general"},
    "goForward": {"color": "green", "direction": "forward"},
    "goLeft": {"color": "green", "direction": "left"},
    "warning": {"color": "yellow", "direction": "general"},
    "warningLeft": {"color": "yellow", "direction": "left"},
    "stop": {"color": "red", "direction": "general"},
    "stopLeft": {"color": "red", "direction": "left"},
}

CLASS_TO_ID = {name: index for index, name in enumerate(LISA_CLASSES)}
