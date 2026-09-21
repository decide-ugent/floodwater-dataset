#!/usr/bin/env python3
"""Browse Floodwater videos with the released water masks overlaid."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .dataset_utils import chunk_ids, decode, iter_samples, resolve_data_root


def render(frame: np.ndarray, mask: np.ndarray, label: str, alpha: float) -> np.ndarray:
    result=frame.copy()
    water=mask == 255
    color=np.zeros_like(result)
    color[..., 0]=255
    color[..., 1]=145
    result[water]=cv2.addWeighted(result, 1.0-alpha, color, alpha, 0)[water]
    contours, _=cv2.findContours(water.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 220, 40), 1)
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(result, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--chunk-id")
    parser.add_argument("--frame", type=int, default=0, help="Labeled-frame position within the selected chunk")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--save-preview", type=Path, help="Write one overlay image and exit")
    args=parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")

    root=resolve_data_root(args.data_root)
    chunks=chunk_ids(root, args.split)
    if not chunks:
        parser.error(f"No {args.split} chunks found under {root}")
    if args.chunk_id:
        if args.chunk_id not in chunks:
            parser.error(f"Unknown {args.split} chunk: {args.chunk_id}")
        chunk_position=chunks.index(args.chunk_id)
    else:
        chunk_position=0

    paused=False
    sample_position=max(0, args.frame)
    while True:
        chunk=chunks[chunk_position]
        samples=list(iter_samples(root, split=args.split, chunk_id=chunk))
        if not samples:
            raise RuntimeError(f"No samples for {chunk}")
        sample_position=min(sample_position, len(samples)-1)
        sample=samples[sample_position]
        frame, mask=decode(sample)
        label=f"{sample.chunk_id} | frame {sample.frame_index:05d} | {sample.split}"
        shown=render(frame, mask, label, args.alpha)

        if args.save_preview:
            args.save_preview.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save_preview), shown):
                raise OSError(f"Could not write {args.save_preview}")
            print(args.save_preview)
            return

        cv2.imshow("Floodwater Dataset | Space pause | N/P chunk | Q quit", shown)
        key=cv2.waitKey(0 if paused else 40) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused=not paused
        elif key == ord("n"):
            chunk_position=(chunk_position+1) % len(chunks)
            sample_position=0
        elif key == ord("p"):
            chunk_position=(chunk_position-1) % len(chunks)
            sample_position=0
        elif key in (81, ord("a")):
            sample_position=max(0, sample_position-1)
        else:
            sample_position=(sample_position+1) % len(samples)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
