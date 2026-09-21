#!/usr/bin/env python3
"""Generate the static and animated showcases used by the repository README."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools.dataset_utils import decode, iter_samples, resolve_data_root
from tools.manual_dataset_utils import (
    decode_manual,
    iter_manual_samples,
    resolve_manual_data_root,
)

VIDEO_SOURCES = (
    "video_00001",
    "video_00003",
    "video_00006",
    "video_00021",
    "video_00010",
    "video_00027",
)
MANUAL_SAMPLES = (
    "sample_000105",
    "sample_000550",
    "sample_000521",
)
VISUALISATIONS_DIR = Path(__file__).resolve().parent
ANIMATION_FRAME_COUNT = 12
ANIMATION_STRIDE = 12


def make_panel(
    image: np.ndarray,
    mask: np.ndarray,
    label: str,
    width: int,
    height: int,
) -> np.ndarray:
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    water = mask == 255
    overlay = np.empty_like(image)
    overlay[:] = (245, 155, 45)
    image[water] = cv2.addWeighted(image, 0.48, overlay, 0.52, 0)[water]
    contours, _ = cv2.findContours(
        water.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (255, 224, 150), 1, cv2.LINE_AA)

    shade = image.copy()
    cv2.rectangle(shade, (0, height - 39), (width, height), (10, 13, 17), -1)
    image = cv2.addWeighted(shade, 0.72, image, 0.28, 0)
    cv2.putText(
        image,
        label,
        (13, height - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def temporal_samples(samples, frame_count: int, stride: int):
    """Select a centered, evenly spaced sequence from one video chunk."""
    center = samples[len(samples) // 2]
    chunk_samples = [
        sample for sample in samples if sample.chunk_id == center.chunk_id
    ]
    chunk_samples.sort(key=lambda sample: sample.frame_index)
    if len(chunk_samples) < frame_count:
        raise RuntimeError(f"Not enough samples in {center.chunk_id}")

    stride = min(stride, max(1, (len(chunk_samples) - 1) // (frame_count - 1)))
    span = (frame_count - 1) * stride
    start = max(0, (len(chunk_samples) - 1 - span) // 2)
    return [chunk_samples[start + index * stride] for index in range(frame_count)]


def manual_animation_sequences(manual_by_id, frame_count: int):
    """Build changing manual-example sequences from telemetry-bearing images."""
    candidates = []
    for sample in manual_by_id.values():
        with Image.open(sample.image_path) as image:
            if image.size == (1920, 1080):
                candidates.append(sample)
    candidates.sort(key=lambda sample: sample.sample_id)
    if len(candidates) < frame_count:
        raise RuntimeError("Not enough 1920x1080 manual samples for the animation")

    positions = {sample.sample_id: index for index, sample in enumerate(candidates)}
    missing = [sample_id for sample_id in MANUAL_SAMPLES if sample_id not in positions]
    if missing:
        raise RuntimeError(f"Manual animation anchors are unavailable: {missing}")

    stride = max(1, len(candidates) // frame_count)
    return [
        [
            candidates[(positions[sample_id] + frame * stride) % len(candidates)]
            for frame in range(frame_count)
        ]
        for sample_id in MANUAL_SAMPLES
    ]


def make_animation_canvas(
    video_panels: list[np.ndarray],
    manual_panels: list[np.ndarray],
    frame_number: int,
    frame_count: int,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    gap, header, rows = 9, 76, 3
    canvas = np.full(
        (
            header + rows * panel_height + (rows + 1) * gap,
            3 * panel_width + 4 * gap,
            3,
        ),
        (255, 255, 255),
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        "Floodwater Dataset",
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (18, 47), (37, 64), (245, 155, 45), -1)
    cv2.putText(
        canvas,
        f"Video mask propagation | changing manual examples | {frame_number + 1:02d}/{frame_count:02d}",
        (47, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )

    for index, panel in enumerate(video_panels):
        column, row = index % 3, index // 3
        x = gap + column * (panel_width + gap)
        y = header + gap + row * (panel_height + gap)
        canvas[y : y + panel_height, x : x + panel_width] = panel

    for column, panel in enumerate(manual_panels):
        x = gap + column * (panel_width + gap)
        y = header + gap + 2 * (panel_height + gap)
        canvas[y : y + panel_height, x : x + panel_width] = panel
    return canvas


def write_animation(
    output: Path,
    by_source,
    manual_by_id,
    frame_count: int,
    stride: int,
) -> None:
    panel_width, panel_height = 320, 180
    selected = {
        source: temporal_samples(by_source[source], frame_count, stride)
        for source in VIDEO_SOURCES
    }
    manual_sequences = manual_animation_sequences(manual_by_id, frame_count)

    canvases = []
    for frame_number in range(frame_count):
        video_panels = []
        for source in VIDEO_SOURCES:
            sample = selected[source][frame_number]
            image, mask = decode(sample)
            video_panels.append(
                make_panel(
                    image,
                    mask,
                    (
                        f"Video sequence | Source {int(source.split('_')[-1]):02d}"
                        f" | {sample.split}"
                    ),
                    panel_width,
                    panel_height,
                )
            )
        manual_panels = []
        for sequence in manual_sequences:
            sample = sequence[frame_number]
            image, mask = decode_manual(sample)
            manual_panels.append(
                make_panel(
                    image,
                    mask,
                    f"Manual annotation | {sample.sample_id}",
                    panel_width,
                    panel_height,
                )
            )
        canvases.append(
            make_animation_canvas(
                video_panels,
                manual_panels,
                frame_number,
                frame_count,
                panel_width,
                panel_height,
            )
        )

    rgb_frames = [
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        for canvas in canvases
    ]
    gif_frames = [
        frame.quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in rgb_frames
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        output,
        save_all=True,
        append_images=gif_frames[1:],
        duration=260,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manual-data-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=VISUALISATIONS_DIR / "floodwater_showcase.jpg",
    )
    parser.add_argument(
        "--animation-output",
        type=Path,
        default=VISUALISATIONS_DIR / "floodwater_showcase.gif",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=ANIMATION_FRAME_COUNT,
    )
    parser.add_argument(
        "--animation-stride",
        type=int,
        default=ANIMATION_STRIDE,
        help="Number of source-video frames between animation frames.",
    )
    args = parser.parse_args()
    data_root = resolve_data_root(args.data_root)
    manual_root = resolve_manual_data_root(args.manual_data_root)

    by_source = defaultdict(list)
    for sample in iter_samples(data_root):
        if sample.source_video_id in VIDEO_SOURCES:
            by_source[sample.source_video_id].append(sample)

    manual_by_id = {
        sample.sample_id: sample for sample in iter_manual_samples(manual_root)
    }

    panel_width, panel_height = 600, 338
    gap, header, rows = 12, 82, 3
    canvas = np.full(
        (header + rows * panel_height + (rows + 1) * gap, 3 * panel_width + 4 * gap, 3),
        (255, 255, 255),
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        "Floodwater Dataset",
        (22, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (22, 53), (43, 70), (245, 155, 45), -1)
    cv2.putText(
        canvas,
        "Water mask overlay | video pseudo-labels and manual annotations",
        (54, 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )

    for index, source in enumerate(VIDEO_SOURCES):
        samples = by_source[source]
        if not samples:
            raise RuntimeError(f"No released samples found for {source}")
        sample = samples[len(samples) // 2]
        image, mask = decode(sample)
        panel = make_panel(
            image,
            mask,
            f"Video pseudo-label | Source {int(source.split('_')[-1]):02d} | {sample.split}",
            panel_width,
            panel_height,
        )
        column, row = index % 3, index // 3
        x = gap + column * (panel_width + gap)
        y = header + gap + row * (panel_height + gap)
        canvas[y : y + panel_height, x : x + panel_width] = panel

    for column, sample_id in enumerate(MANUAL_SAMPLES):
        sample = manual_by_id.get(sample_id)
        if sample is None:
            raise RuntimeError(f"No manual sample found for {sample_id}")
        image, mask = decode_manual(sample)
        panel = make_panel(
            image,
            mask,
            f"Manual annotation | {sample.sample_id}",
            panel_width,
            panel_height,
        )
        x = gap + column * (panel_width + gap)
        y = header + gap + 2 * (panel_height + gap)
        canvas[y : y + panel_height, x : x + panel_width] = panel

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write {args.output}")
    print(args.output)
    write_animation(
        args.animation_output,
        by_source,
        manual_by_id,
        args.animation_frames,
        args.animation_stride,
    )
    print(args.animation_output)


if __name__ == "__main__":
    main()
