"""Default paths and constants for the Floodwater Dataset utilities."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "FLOODWATER_DATA",
        REPO_ROOT / "data" / "floodwater_dataset",
    )
).expanduser().resolve()

MANUAL_DATA_ROOT = Path(
    os.environ.get(
        "FLOODWATER_MANUAL_DATA",
        REPO_ROOT / "data" / "floodwater_manual",
    )
).expanduser().resolve()

FPS = 25
MASK_BACKGROUND = 0
MASK_WATER = 255
SPLITS = ("train", "val", "test")
