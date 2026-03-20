#!/usr/bin/env python
"""
Split selected Testsuite HDF5 files into train/test files by snapshot index.

Behavior:
- Copies each input file twice.
- Replaces DATASET/FIELDS/*/value datasets:
  - train: first 70% snapshots
  - test: remaining 30% snapshots
- Replaces DATASET/VARIABLES/*/value datasets with the same split.
- Leaves all other groups/datasets unchanged (/MESH, /PARTITIONS, etc.).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py


def _replace_field_value_dataset(field_group: h5py.Group, start: int, stop: int) -> None:
    if "value" not in field_group:
        return

    old = field_group["value"]
    if old.ndim < 2:
        raise RuntimeError(
            f"Expected at least 2D dataset for snapshots, got shape={old.shape} in {old.name}"
        )

    data = old[:, start:stop]
    old_attrs = dict(old.attrs.items())

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

    # Keep loader metadata consistent: vars[0] stores snapshot-axis length.
    if "vars" in field_group and field_group["vars"].shape[0] >= 1:
        vars_arr = field_group["vars"][:]
        vars_arr[0] = data.shape[1]
        field_group["vars"][:] = vars_arr


def _replace_variable_value_dataset(var_group: h5py.Group, start: int, stop: int, n_total: int) -> None:
    if "value" not in var_group:
        return

    old = var_group["value"]
    if old.ndim == 0:
        return

    old_attrs = dict(old.attrs.items())

    if old.ndim == 1:
        data = old[start:stop]
    else:
        # Heuristic: snapshots usually lie on axis 0 for VARIABLES (e.g. time),
        # but support axis -1 when needed.
        if old.shape[0] == n_total:
            sl = [slice(None)] * old.ndim
            sl[0] = slice(start, stop)
            data = old[tuple(sl)]
        elif old.shape[-1] == n_total:
            sl = [slice(None)] * old.ndim
            sl[-1] = slice(start, stop)
            data = old[tuple(sl)]
        else:
            # Unknown convention: keep dataset unchanged.
            return

    create_kwargs = {}
    if old.compression is not None:
        create_kwargs["compression"] = old.compression
        create_kwargs["compression_opts"] = old.compression_opts
    if old.shuffle is not None:
        create_kwargs["shuffle"] = old.shuffle
    if old.fletcher32 is not None:
        create_kwargs["fletcher32"] = old.fletcher32

    del var_group["value"]
    new = var_group.create_dataset("value", data=data, dtype=data.dtype, **create_kwargs)
    for k, v in old_attrs.items():
        new.attrs[k] = v


def _infer_num_snapshots(path: Path) -> int:
    with h5py.File(path, "r") as f:
        fields = f["DATASET"]["FIELDS"]
        for field_name in fields.keys():
            value = fields[field_name].get("value")
            if value is not None:
                if value.ndim < 2:
                    raise RuntimeError(
                        f"Expected at least 2D dataset for snapshots, got shape={value.shape} in {value.name}"
                    )
                return int(value.shape[1])
    raise RuntimeError(f"No DATASET/FIELDS/*/value dataset found in {path}")


def _split_file_values(path: Path, train: bool, n_train: int, n_test: int, n_total: int) -> None:
    with h5py.File(path, "r+") as f:
        start, stop = (0, n_train) if train else (n_train, n_train + n_test)

        fields = f["DATASET"]["FIELDS"]
        for field_name in fields.keys():
            _replace_field_value_dataset(fields[field_name], start, stop)

        variables = f["DATASET"].get("VARIABLES")
        if variables is not None:
            for var_name in variables.keys():
                _replace_variable_value_dataset(variables[var_name], start, stop, n_total)


parser = argparse.ArgumentParser(description="Create *_TRAIN.h5 and *_TEST.h5 for selected Testsuite files.")
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("../DATA"),
    help="Directory with testsuite HDF5 files (default: ../DATA)",
)
parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio in [0,1] (default: 0.7)")
args = parser.parse_args()

data_dir = args.data_dir.resolve()
fnames = ["CHANNEL.h5", "CYLINDER.h5", "JET.h5", "TENSOR280.h5"]

for fname in fnames:
    input_path = data_dir / fname
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    train_path = input_path.with_name(f"{input_path.stem}_TRAIN.h5")
    test_path = input_path.with_name(f"{input_path.stem}_TEST.h5")

    shutil.copy2(input_path, train_path)
    shutil.copy2(input_path, test_path)

    n_total = _infer_num_snapshots(input_path)
    n_train = int(n_total * args.train_ratio)
    if n_total > 1:
        n_train = max(1, min(n_total - 1, n_train))
    else:
        n_train = n_total
    n_test = n_total - n_train

    _split_file_values(train_path, train=True, n_train=n_train, n_test=n_test, n_total=n_total)
    _split_file_values(test_path, train=False, n_train=n_train, n_test=n_test, n_total=n_total)

    print(f"Input : {input_path}")
    print(f"Train : {train_path} ({n_train} snapshots, {args.train_ratio:.0%})")
    print(f"Test  : {test_path} ({n_test} snapshots, {(1.0 - args.train_ratio):.0%})")
