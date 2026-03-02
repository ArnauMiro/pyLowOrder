#!/usr/bin/env python
"""
Split CYLINDER.h5 into train/test files by snapshot index.

Behavior:
- Copies the full input file twice.
- Replaces only DATASET/FIELDS/*/value datasets:
  - train: first 12 snapshots
  - test: last 4 snapshots
- Leaves all other groups/datasets unchanged (/MESH, /PARTITIONS, etc.).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py


def _replace_value_dataset(field_group: h5py.Group, start: int, stop: int) -> None:
    if "value" not in field_group:
        return

    old = field_group["value"]
    if old.ndim < 2:
        raise RuntimeError(
            f"Expected at least 2D dataset for snapshots, got shape={old.shape} in {old.name}"
        )

    data = old[:, start:stop]
    old_attrs = dict(old.attrs.items())

    # Preserve compression/chunk options when possible.
    create_kwargs = {}
    if old.compression is not None:
        create_kwargs["compression"] = old.compression
        create_kwargs["compression_opts"] = old.compression_opts
    if old.shuffle is not None:
        create_kwargs["shuffle"] = old.shuffle
    if old.fletcher32 is not None:
        create_kwargs["fletcher32"] = old.fletcher32

    del field_group["value"]
    new = field_group.create_dataset("value", data=data, dtype=data.dtype, **create_kwargs)
    for k, v in old_attrs.items():
        new.attrs[k] = v


def _split_file_values(path: Path, train: bool, n_train: int, n_test: int) -> None:
    with h5py.File(path, "r+") as f:
        fields = f["DATASET"]["FIELDS"]
        for field_name in fields.keys():
            g = fields[field_name]
            if train:
                _replace_value_dataset(g, 0, n_train)
            else:
                _replace_value_dataset(g, n_train, n_train + n_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CYLINDER_TRAIN.h5 and CYLINDER_TEST.h5.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Testsuite/DATA/CYLINDER.h5"),
        help="Input HDF5 file (default: Testsuite/DATA/CYLINDER.h5)",
    )
    parser.add_argument("--n-train", type=int, default=12, help="Number of training snapshots (default: 12)")
    parser.add_argument("--n-test", type=int, default=4, help="Number of test snapshots (default: 4)")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    train_path = input_path.with_name("CYLINDER_TRAIN.h5")
    test_path = input_path.with_name("CYLINDER_TEST.h5")

    shutil.copy2(input_path, train_path)
    shutil.copy2(input_path, test_path)

    _split_file_values(train_path, train=True, n_train=args.n_train, n_test=args.n_test)
    _split_file_values(test_path, train=False, n_train=args.n_train, n_test=args.n_test)

    print(f"Input : {input_path}")
    print(f"Train : {train_path} ({args.n_train} snapshots)")
    print(f"Test  : {test_path} ({args.n_test} snapshots)")


if __name__ == "__main__":
    main()
