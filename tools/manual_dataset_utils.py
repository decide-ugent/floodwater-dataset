"""Access utilities for the manually annotated Floodwater evaluation set."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import MANUAL_DATA_ROOT


@dataclass(frozen=True)
class ManualSample:
    sample_id: str
    image_path: Path
    mask_path: Path


def resolve_manual_data_root(value: str | Path | None = None) -> Path:
    return Path(value).expanduser().resolve() if value else MANUAL_DATA_ROOT


def iter_manual_samples(
    data_root: str | Path | None = None,
) -> Iterator[ManualSample]:
    root = resolve_manual_data_root(data_root)
    image_directory = root / "images"
    mask_directory = root / "masks"
    if not image_directory.is_dir() or not mask_directory.is_dir():
        raise FileNotFoundError(
            f"Missing images or masks under {root}. Extract the manual dataset into "
            "data/floodwater_manual or set FLOODWATER_MANUAL_DATA."
        )

    images = {path.stem: path for path in image_directory.glob("*.jpg")}
    masks = {path.stem: path for path in mask_directory.glob("*.png")}
    if images.keys() != masks.keys():
        missing_images = sorted(masks.keys() - images.keys())
        missing_masks = sorted(images.keys() - masks.keys())
        raise ValueError(
            f"Unpaired manual samples: missing images={missing_images[:10]}, "
            f"missing masks={missing_masks[:10]}"
        )

    for sample_id in sorted(images):
        yield ManualSample(
            sample_id=sample_id,
            image_path=images[sample_id],
            mask_path=masks[sample_id],
        )


def decode_manual(sample: ManualSample):
    """Decode a manual sample and return (bgr_image, uint8_mask)."""
    import cv2
    import numpy as np
    from PIL import Image

    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot decode image: {sample.image_path}")
    mask = np.asarray(Image.open(sample.mask_path).convert("L"))
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask size mismatch for {sample.sample_id}")
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"Unexpected mask values for {sample.sample_id}: {values}")
    return image, mask
