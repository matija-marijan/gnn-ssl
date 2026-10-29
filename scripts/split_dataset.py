#!/usr/bin/env python
"""Split a SYDRA dataset into deterministic index ranges.

Layout of a SYDRA dataset:
    <src>/metadata.json
    <src>/samples/<i>/<mic>.wav

`signals_dir` in each metadata row is stored relative to the dataset root
(e.g. "samples/7500"), and `SydraDataset` prefixes it with the dataset
directory at load time. Sample directories are therefore moved without
renaming, and the metadata list is sliced by index. This makes the split
lossless and leaves the on-disk format identical to a freshly generated
dataset.

Usage:
    python scripts/split_dataset.py --src datasets/upstream_raw/raw_5m \
        --dst datasets/full \
        --split train_5m:0:7500 \
        --split val_5m:7500:10000 \
        --split test_5m:10000:12500
"""
import argparse
import json
import shutil
from pathlib import Path


def parse_split(value):
    name, start, end = value.split(":")
    return name, int(start), int(end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source SYDRA dataset")
    parser.add_argument("--dst", required=True, help="destination root")
    parser.add_argument("--split", action="append", required=True,
                        help="name:start:end (end exclusive)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    metadata = json.loads((src / "metadata.json").read_text())
    splits = [parse_split(s) for s in args.split]

    # Validate everything before touching the filesystem.
    for name, start, end in splits:
        if not (0 <= start < end <= len(metadata)):
            raise SystemExit(
                f"{name}: invalid range {start}:{end} for dataset with "
                f"{len(metadata)} samples")
        for i in range(start, end):
            if not (src / "samples" / str(i)).is_dir():
                raise SystemExit(f"missing sample directory {src}/samples/{i}")
        if (dst / name).exists():
            raise SystemExit(f"destination already exists: {dst / name}")

    covered = sorted(i for _, start, end in splits for i in range(start, end))
    if covered != list(range(len(metadata))):
        raise SystemExit("split ranges must cover the dataset exactly once")

    for name, start, end in splits:
        out = dst / name
        (out / "samples").mkdir(parents=True)
        for i in range(start, end):
            shutil.move(str(src / "samples" / str(i)),
                        str(out / "samples" / str(i)))
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata[start:end], f, indent=4)
        print(f"{name}: {end - start} samples (indices {start}:{end})")

    shutil.rmtree(src)


if __name__ == "__main__":
    main()
