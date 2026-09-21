#!/usr/bin/env python3
"""Validate the manually annotated Floodwater image/mask pairs."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .manual_dataset_utils import iter_manual_samples, resolve_manual_data_root

EXPECTED_SAMPLES = 700
EXPECTED_RESOLUTIONS = {
    "1280x720": 466,
    "1920x1080": 234,
}
SAMPLE_ID = re.compile(r"sample_[0-9]{6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    root = resolve_manual_data_root(args.data_root)

    errors: list[str] = []
    resolutions: Counter[str] = Counter()
    samples = []

    try:
        samples = list(iter_manual_samples(root))
    except (FileNotFoundError, ValueError) as error:
        errors.append(str(error))

    expected_ids = {f"sample_{index:06d}" for index in range(1, EXPECTED_SAMPLES + 1)}
    actual_ids = {sample.sample_id for sample in samples}
    if actual_ids != expected_ids:
        errors.append(
            f"sample ID set mismatch: expected {EXPECTED_SAMPLES}, found {len(actual_ids)}"
        )

    for sample in samples:
        if SAMPLE_ID.fullmatch(sample.sample_id) is None:
            errors.append(f"invalid sample ID: {sample.sample_id}")
        try:
            with Image.open(sample.image_path) as image:
                image.load()
                image_size = image.size
                if image.format != "JPEG" or image.mode != "RGB":
                    errors.append(
                        f"unexpected image format for {sample.sample_id}: "
                        f"{image.format}/{image.mode}"
                    )
                if image.getexif():
                    errors.append(f"embedded EXIF metadata: {sample.sample_id}")
            with Image.open(sample.mask_path) as mask:
                mask.load()
                mask_array = np.asarray(mask)
                if mask.format != "PNG" or mask.mode != "L":
                    errors.append(
                        f"unexpected mask format for {sample.sample_id}: "
                        f"{mask.format}/{mask.mode}"
                    )
                if mask.size != image_size:
                    errors.append(f"image/mask size mismatch: {sample.sample_id}")
                if not np.all((mask_array == 0) | (mask_array == 255)):
                    errors.append(f"non-binary mask: {sample.sample_id}")
                if not np.any(mask_array):
                    errors.append(f"empty mask: {sample.sample_id}")
                if np.all(mask_array):
                    errors.append(f"all-water mask: {sample.sample_id}")
            resolutions[f"{image_size[0]}x{image_size[1]}"] += 1
        except OSError as error:
            errors.append(f"decode failure for {sample.sample_id}: {error}")

    if dict(resolutions) != EXPECTED_RESOLUTIONS:
        errors.append(
            f"resolution counts mismatch: expected {EXPECTED_RESOLUTIONS}, "
            f"found {dict(resolutions)}"
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "data_root": str(root),
        "samples": len(samples),
        "resolutions": dict(resolutions),
        "errors": errors[:100],
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
