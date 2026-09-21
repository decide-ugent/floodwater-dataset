#!/usr/bin/env python3
"""Validate manifests, split isolation, and optionally every released asset path."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .dataset_utils import resolve_data_root

EXPECTED={
    "train": {"videos": 22, "chunks": 68, "masks": 159848},
    "val": {"videos": 4, "chunks": 13, "masks": 30017},
    "test": {"videos": 5, "chunks": 17, "masks": 41159},
}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--check-assets", action="store_true")
    args=parser.parse_args()
    root=resolve_data_root(args.data_root)
    errors=[]
    chunks={}
    split_chunks=defaultdict(set)
    split_videos=defaultdict(set)
    with (root/"metadata/chunks.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            chunk=row["chunk_id"]
            if chunk in chunks:
                errors.append(f"duplicate chunk: {chunk}")
            chunks[chunk]=row
            split_chunks[row["split"]].add(chunk)
            split_videos[row["split"]].add(row["source_video_id"])
            if args.check_assets:
                if not (root/row["video_path"]).is_file():
                    errors.append(f"missing video: {row['video_path']}")
                if not (root/row["mask_directory"]).is_dir():
                    errors.append(f"missing mask directory: {row['mask_directory']}")
    counts=Counter()
    seen=set()
    source_split={}
    checked=0
    with (root/"metadata/samples.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            counts[row["split"]]+=1
            if row["sample_id"] in seen:
                errors.append(f"duplicate sample: {row['sample_id']}")
            seen.add(row["sample_id"])
            chunk=chunks.get(row["chunk_id"])
            if chunk is None or chunk["split"] != row["split"]:
                errors.append(f"invalid chunk reference: {row['sample_id']}")
            previous=source_split.setdefault(row["source_video_id"], row["split"])
            if previous != row["split"]:
                errors.append(f"source video crosses splits: {row['source_video_id']}")
            if args.check_assets and not (root/row["mask_path"]).is_file():
                errors.append(f"missing mask: {row['mask_path']}")
            checked+=int(args.check_assets)
    actual={s:{"videos":len(split_videos[s]), "chunks":len(split_chunks[s]), "masks":counts[s]} for s in EXPECTED}
    if actual != EXPECTED:
        errors.append(f"count mismatch: {actual}")
    report={"status":"PASS" if not errors else "FAIL", "data_root":str(root), "verified":actual, "total_samples":sum(counts.values()), "checked_asset_paths":checked, "errors":errors[:100]}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
