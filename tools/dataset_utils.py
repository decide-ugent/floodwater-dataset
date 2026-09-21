"""Manifest-based access to the Floodwater Dataset."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import DATA_ROOT, SPLITS


@dataclass(frozen=True)
class FloodwaterSample:
    sample_id: str
    source_video_id: str
    chunk_id: str
    split: str
    frame_index: int
    time_seconds: float
    video_path: Path
    mask_path: Path
    label_type: str


def resolve_data_root(value: str | Path | None = None) -> Path:
    return Path(value).expanduser().resolve() if value else DATA_ROOT


def _manifest(root: Path, name: str) -> Path:
    path = root / "metadata" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Extract the dataset into data/floodwater_dataset or set "
            "FLOODWATER_DATA to the extracted dataset directory."
        )
    return path


def iter_samples(
    data_root: str | Path | None = None,
    split: str | None = None,
    chunk_id: str | None = None,
) -> Iterator[FloodwaterSample]:
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    root = resolve_data_root(data_root)
    with _manifest(root, "samples.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["split"] != split:
                continue
            if chunk_id is not None and row["chunk_id"] != chunk_id:
                continue
            yield FloodwaterSample(
                sample_id=row["sample_id"],
                source_video_id=row["source_video_id"],
                chunk_id=row["chunk_id"],
                split=row["split"],
                frame_index=int(row["chunk_frame_index"]),
                time_seconds=float(row["chunk_time_seconds"]),
                video_path=root / row["video_path"],
                mask_path=root / row["mask_path"],
                label_type=row.get("label_type", "sam2_pseudo"),
            )


def chunk_ids(data_root: str | Path | None = None, split: str | None = None) -> list[str]:
    root = resolve_data_root(data_root)
    values=[]
    with _manifest(root, "chunks.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if split is None or row["split"] == split:
                values.append(row["chunk_id"])
    return values


def decode(sample: FloodwaterSample):
    """Decode a sample and return ``(bgr_frame, uint8_mask)``."""
    import cv2
    import numpy as np
    from PIL import Image

    capture=cv2.VideoCapture(str(sample.video_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {sample.video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
    ok, frame=capture.read()
    capture.release()
    if not ok:
        raise OSError(f"Cannot decode {sample.sample_id}")
    mask=np.asarray(Image.open(sample.mask_path).convert("L"))
    if frame.shape[:2] != mask.shape:
        raise ValueError(f"Frame/mask size mismatch for {sample.sample_id}")
    values=set(np.unique(mask).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"Unexpected mask values for {sample.sample_id}: {values}")
    return frame, mask
