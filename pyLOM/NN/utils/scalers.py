#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines.
#
# Last rev:

# Built-in modules
import os
import json
from typing import List, Union, Protocol

# Third-party libraries
import numpy as np
import torch

# Local modules
from ...utils.errors import raiseError




class ScalerProtocol(Protocol):
    r'''
    Abstract protocol for scalers. Must include:
        - fit: Fit the scaler to the data.
        - transform: Transform the data using the fitted scaler.
        - fit_transform: Fit the scaler to the data and transform it.
    '''
    def fit(self, X: np.ndarray, y=None) -> "ScalerProtocol":
        r""" Fit the scaler to the data. """
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        r""" Transform the data using the fitted scaler. """
        ...

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        r""" Adjust the scaler to the data and transform it. """
        ...


class MinMaxScaler:
    r"""
    Min-max scaling to scale variables to a desired range.

    By default (blocks=None), each column is treated as an independent variable
    (fully backward-compatible with previous behavior).

    If `blocks` is provided, it defines groups of columns (blocks) that are scaled
    together sharing the same (min, max). This is ideal when each "variable"
    is actually a vector quantity with multiple components.

    Args:
        feature_range (Tuple): Desired range of transformed data. Default: (0, 1).
        column (bool): Scale over column or row space. Default: False.
        blocks (Optional[List[Union[slice, Sequence[int]]]]): Column groupings.
            - None (default): each column is its own variable (old behavior).
            - List of slices or list of index lists. E.g.:
                blocks=[slice(0,1), slice(1,3), slice(3,6)]
              or   blocks=[[0], [1,2], [3,4,5]]
    """

    def __init__(self, feature_range=(0, 1), column=False, blocks=None):
        self.feature_range = feature_range
        self._is_fitted = False
        self._column    = column
        self.blocks     = blocks  # NEW

    @property
    def is_fitted(self):
        return self._is_fitted

    # ---------- internal helpers (robustos y retrocompatibles) ----------
    def _ensure_2d(self, x):
        """Return x as 2D array/tensor of shape (n_samples, n_features)."""
        if isinstance(x, torch.Tensor):
            if x.ndim == 1:
                return x.unsqueeze(1)
            return x
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        return x

    def _split_into_blocks(self, X2d):
        """
        Split a 2D np.ndarray/torch.Tensor into a list of 2D blocks by self.blocks.
        If self.blocks is None, split by single columns (old behavior).
        """
        is_tensor = isinstance(X2d, torch.Tensor)

        # Build list of (take_fn, block) safely for np/torch
        def take_cols(a, cols):
            if is_tensor:
                return a[:, cols] if isinstance(cols, list) else a[:, cols]
            else:
                return a[:, cols]

        nfeat = X2d.shape[1]
        if self.blocks is None:
            # one column per variable (old behavior)
            blocks = [take_cols(X2d, i if is_tensor else slice(i, i+1)) for i in range(nfeat)]
        else:
            norm_blocks = []
            for b in self.blocks:
                if isinstance(b, slice):
                    norm_blocks.append(take_cols(X2d, b))
                else:
                    # list/tuple of indices
                    # torch advanced indexing needs a tensor of column indices
                    if is_tensor and not isinstance(b, torch.Tensor):
                        cols = torch.tensor(b, dtype=torch.long, device=X2d.device)
                        norm_blocks.append(X2d.index_select(dim=1, index=cols))
                    else:
                        norm_blocks.append(take_cols(X2d, b))
            blocks = norm_blocks
        return blocks

    def _stack_blocks(self, blocks, as_tensor=False):
        """
        Stack list of 2D blocks back to a single 2D array/tensor with original column order.
        Assumes blocks cover all columns in order given by `blocks`/default.
        """
        if as_tensor:
            return torch.hstack(blocks)
        else:
            return np.hstack(blocks)

    # ------------------------------ API ------------------------------
    def fit(self, variables: Union[List[Union[np.ndarray, torch.tensor]], np.ndarray, torch.tensor]):
        """
        Compute the min and max per variable.
        - If `blocks is None`:
            - Backward compatible: if `variables` is array/tensor, split by columns.
            - If `variables` is a list, each item is treated as an independent variable.
        - If `blocks` provided and `variables` is array/tensor:
            - Group by the blocks for min/max.
        """
        # Normalize inputs to a single 2D matrix for easier handling
        is_array  = isinstance(variables, np.ndarray)
        is_tensor = isinstance(variables, torch.Tensor)

        if is_array or is_tensor:
            X = self._ensure_2d(variables)
            blocks = self._split_into_blocks(X)
        else:
            # List path (backward-compatible): each element can be (n, d_i)
            blocks = []
            for v in variables:
                v2d = self._ensure_2d(v)
                blocks.append(v2d)

        min_max_values = []
        for block in blocks:
            # Compute global min/max over the whole block (shared for all its columns)
            bmin = float(block.min())
            bmax = float(block.max())
            min_max_values.append({"min": bmin, "max": bmax})
        self.variable_scaling_params = min_max_values
        self._is_fitted = True

        # Keep metadata to reconstruct shapes
        # When fitting from a single 2D matrix, we remember how to split/stack later.
        # If a list is passed, we know we should return a list on transform if a list was passed.
        self._fit_from_list = not (is_array or is_tensor)

    def transform(
        self, variables: Union[List[Union[np.ndarray, torch.tensor]], np.ndarray, torch.tensor]
    ):
        """
        Scale variables using min-max.
        Returns:
            - If input was np.ndarray/torch.Tensor: returns same type (2D), preserving shape.
            - If input was list: returns list of same length with each block scaled.
        """
        if not self._is_fitted:
            raiseError("Scaler must be fitted before transform")

        def _scale_block(block, params, feature_range):
            min_val, max_val = params["min"], params["max"]
            data_range = max_val - min_val
            data_range = 1 if data_range == 0 else data_range
            out = (block - min_val) / data_range
            out = out * (feature_range[1] - feature_range[0]) + feature_range[0]
            return out

        is_array  = isinstance(variables, np.ndarray)
        is_tensor = isinstance(variables, torch.Tensor)

        # --- Case 1: array/tensor input ---
        if is_array or is_tensor:
            X = self._ensure_2d(variables)
            blocks = self._split_into_blocks(X)

            if is_tensor:
                scaled_blocks = [
                    _scale_block(block, p, self.feature_range) for block, p in zip(blocks, self.variable_scaling_params)
                ]
                out = self._stack_blocks(scaled_blocks, as_tensor=True)
                return out.T if self._column else out
            else:
                scaled_blocks = [
                    _scale_block(block.astype(float), p, self.feature_range) for block, p in zip(blocks, self.variable_scaling_params)
                ]
                out = self._stack_blocks(scaled_blocks, as_tensor=False)
                return out.T if self._column else out

        # --- Case 2: list input (backward-compatible) ---
        scaled_list = []
        for i, v in enumerate(variables):
            v2d = self._ensure_2d(v)
            scaled_v = _scale_block(v2d, self.variable_scaling_params[i], self.feature_range)
            scaled_list.append(scaled_v)
        return scaled_list

    def fit_transform(self, variables):
        self.fit(variables)
        return self.transform(variables)

    def inverse_transform(self, variables):
        """
        Inverse transformation. Mirrors `transform` rules on input/output typing.
        """
        if not self._is_fitted:
            raiseError("Scaler must be fitted before inverse_transform")

        def _inv_block(block, params, feature_range):
            min_val, max_val = params["min"], params["max"]
            data_range = max_val - min_val
            data_range = 1 if data_range == 0 else data_range
            out = (block - feature_range[0]) / (feature_range[1] - feature_range[0])
            out = out * data_range + min_val
            return out

        is_array  = isinstance(variables, np.ndarray)
        is_tensor = isinstance(variables, torch.Tensor)

        if is_array or is_tensor:
            X = self._ensure_2d(variables)
            blocks = self._split_into_blocks(X)

            if len(blocks) != len(self.variable_scaling_params):
                raiseError(
                    f"Number of variables to inverse transform ({len(blocks)}) does not match the number fitted ({len(self.variable_scaling_params)})"
                )

            if is_tensor:
                inv_blocks = [
                    _inv_block(block, p, self.feature_range) for block, p in zip(blocks, self.variable_scaling_params)
                ]
                out = self._stack_blocks(inv_blocks, as_tensor=True)
                return out.T if self._column else out
            else:
                inv_blocks = [
                    _inv_block(block.astype(float), p, self.feature_range) for block, p in zip(blocks, self.variable_scaling_params)
                ]
                out = self._stack_blocks(inv_blocks, as_tensor=False)
                return out.T if self._column else out

        # list path
        inv_list = []
        if len(variables) != len(self.variable_scaling_params):
            raiseError(
                f"Number of variables to inverse transform ({len(variables)}) does not match the number fitted ({len(self.variable_scaling_params)})"
            )
        for v, p in zip(variables, self.variable_scaling_params):
            v2d = self._ensure_2d(v)
            inv_list.append(_inv_block(v2d, p, self.feature_range))
        return inv_list

    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raiseError("Scaler must be fitted before it can be saved")
        # Normalize blocks to a JSON-serializable form
        def _serialize_blocks(blocks):
            if blocks is None:
                return None
            serial = []
            for b in blocks:
                if isinstance(b, slice):
                    serial.append({"type": "slice", "start": b.start, "stop": b.stop, "step": b.step})
                else:
                    serial.append({"type": "list", "indices": list(b)})
            return serial

        save_dict = {
            "feature_range": self.feature_range,
            "variable_scaling_params": self.variable_scaling_params,
            "column": self._column,
            "blocks": _serialize_blocks(self.blocks),
        }
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=4)

    @staticmethod
    def load(filepath: str) -> 'MinMaxScaler':
        if not os.path.exists(filepath):
            raiseError(f"No file found at {filepath}")
        with open(filepath, 'r') as f:
            loaded = json.load(f)

        # Reconstruct blocks
        def _deserialize_blocks(serial):
            if serial is None:
                return None
            out = []
            for item in serial:
                if item["type"] == "slice":
                    out.append(slice(item["start"], item["stop"], item["step"]))
                else:
                    out.append(list(item["indices"]))
            return out

        scaler = MinMaxScaler(
            feature_range=tuple(loaded["feature_range"]),
            column=loaded["column"],
            blocks=_deserialize_blocks(loaded.get("blocks", None))
        )
        scaler.variable_scaling_params = loaded["variable_scaling_params"]
        scaler._is_fitted = True
        return scaler
