#!/usr/bin/env python3
"""Browse the manually annotated Floodwater image/mask pairs."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .manual_dataset_utils import decode_manual, iter_manual_samples


def render(image: np.ndarray, mask: np.ndarray, label: str, alpha: float) -> np.ndarray:
    result = image.copy()
    water = mask == 255
    color = np.zeros_like(result)
    color[..., 0] = 255
    color[..., 1] = 145
    result[water] = cv2.addWeighted(result, 1.0 - alpha, color, alpha, 0)[water]
    contours, _ = cv2.findContours(
        water.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (255, 220, 40), 1)
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--save-preview", type=Path)
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")

    samples = list(iter_manual_samples(args.data_root))
    if not samples:
        parser.error("No manual samples found")

    if args.sample_id:
        positions = {
            sample.sample_id: position for position, sample in enumerate(samples)
        }
        if args.sample_id not in positions:
            parser.error(f"Unknown manual sample: {args.sample_id}")
        position = positions[args.sample_id]
    else:
        position = min(max(args.index, 0), len(samples) - 1)

    while True:
        sample = samples[position]
        image, mask = decode_manual(sample)
        shown = render(image, mask, sample.sample_id, args.alpha)

        if args.save_preview:
            args.save_preview.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save_preview), shown):
                raise OSError(f"Could not write {args.save_preview}")
            print(args.save_preview)
            return

        cv2.imshow("Floodwater Manual | N/P sample | Q quit", shown)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            position = (position - 1) % len(samples)
        else:
            position = (position + 1) % len(samples)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
