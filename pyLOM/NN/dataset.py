#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines.
#
# Last rev:

import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch import Generator, randperm, default_generator
# Built-in modules
import copy
from itertools import product, accumulate
from typing import List, Optional, Tuple, Union, Callable, Sequence, cast
from sklearn.neighbors  import NearestNeighbors

from .utils.scalers import ScalerProtocol

from .                import DEVICE
from ..utils.cr       import cr
from ..utils.errors   import raiseWarning, raiseError
from ..dataset        import Dataset as pyLOMDataset
from ..               import pprint

# Wrapper of the activation functions
def tanh():      return nn.Tanh()
def relu():      return nn.ReLU()
def elu():       return nn.ELU()
def sigmoid():   return nn.Sigmoid()
def leakyRelu(): return nn.LeakyReLU()
def silu():      return nn.SiLU()




class Dataset(torch.utils.data.Dataset):
    r"""
    Dataset class to be used with PyTorch. It can be used with both mesh and point data.
    It is useful convert the `pyLOM.Dataset` to a PyTorch dataset and train neural networks with results from CFD simulations.

    Example:
        >>> original_dataset = pyLOM.Dataset.load(path)
        >>> input_scaler = pyLOM.NN.MinMaxScaler()
        >>> output_scaler = pyLOM.NN.MinMaxScaler()
        >>> dataset = pyLOM.NN.Dataset(
        ...     variables_out=(original_dataset["CP"],),
        ...     variables_in=original_dataset.xyz,
        ...     parameters=[[*zip(original_dataset.get_variable('AoA'), original_dataset.get_variable('Mach'))]], # to have each Mach and AoA pair just once. To have all possibnle combinations, use [original_dataset.get_variable('AoA'), original_dataset.get_variable("Mach")]
        ...     inputs_scaler=input_scaler,
        ...     outputs_scaler=output_scaler,
        ... )


    Args:
        variables_out (Tuple): Tuple of variables to be used as output. Each variable should be a 2d numpy array or torch tensor. If only one variable wants to be provided, it should be passed as a tuple with one element. E.g. ``(variable,)``.
        mesh_shape (Tuple): Shape of the mesh. If not mesh is used and the data is considered as points, leave this as default. Default is ``(1,)``.
        variables_in (np.ndarray): Input variables. Default is ``None``.
        parameters (List[List[float]]): List of parameters to be used as input. All possible combinations of these parameters, i.e. its cartesian product, will appear along with variables_in. If there is only one inner list and its elements are tuples, they will be treated as a single element for the cartesiand produt, which is useful when the combination of the parameters was predefined. Default is ``None``.
        inputs_scaler (MinMaxScaler): Scaler to scale the input variables. If the scaler is not fitted, it will be fitted. Default is ``None``.
        outputs_scaler (MinMaxScaler): Scaler to scale the output variables. If the scaler is not fitted, it will be fitted. Default is ``None``.
        snapshots_by_column (bool): If the snapshots from `variables_out` are stored by column. The snapshots on `pyLOM.Dataset`s have this format. Default is ``True``.
    """
    @cr("NN.Dataset.__init__")
    def __init__(
        self,
        variables_out: Tuple,
        mesh_shape: Tuple = (1,),
        variables_in: np.ndarray = None,
        parameters: List[List[float]] = None,
        combine_parameters_with_cartesian_prod: bool = False,
        inputs_scaler: ScalerProtocol = None,
        outputs_scaler: ScalerProtocol = None,
        snapshots_by_column: bool = True,
        squeeze_last_dim: bool = True,
        channels_last: bool = False,
    ):
        """
        See class docstring for argument semantics.
        """
        # --- metadata ---
        self.parameters = parameters
        self.num_channels = len(variables_out)
        self.mesh_shape = mesh_shape
        self.channels_last = channels_last

        # Attach scalers on the instance
        self.inputs_scaler = inputs_scaler
        self.outputs_scaler = outputs_scaler

        # --- outputs ---
        if snapshots_by_column:
            variables_out = [variable.T for variable in variables_out]

        self.variables_out = self._process_variables_out(
            variables_out, self.outputs_scaler, squeeze_last_dim, channels_last
        )

        # --- inputs / parameters ---
        if variables_in is not None:
            self.parameters = self._process_parameters(parameters, combine_parameters_with_cartesian_prod)
            self.variables_in = torch.tensor(variables_in, dtype=torch.float32)

            if self.inputs_scaler is not None:
                # Build column-wise list: [variables_in columns] + [parameters columns]
                variables_in_columns = [self.variables_in[:, i] for i in range(self.variables_in.shape[1])]
                parameters_columns = [self.parameters[:, i] for i in range(self.parameters.shape[1])] if self.parameters is not None else []
                cols = variables_in_columns + parameters_columns  # list of 2D tensors (n, 1) after scaler handling

                # Fit once if needed and transform (list-mode by protocol)
                if not self.inputs_scaler.is_fitted:
                    self.inputs_scaler.fit(cols)
                transformed = self.inputs_scaler.transform(cols)

                # Reconstruct tensors
                self.variables_in = torch.cat(transformed[: self.variables_in.shape[1]], dim=1).float()
                if self.parameters is not None:
                    self.parameters = torch.cat(transformed[self.variables_in.shape[1] :], dim=1).float()
        else:
            self.variables_in = None
            if parameters is not None:
                raiseWarning("Parameters were passed but no input variables were passed. Parameters will be ignored.")
            self.parameters = None


    def _process_variables_out(
        self,
        variables_out,
        outputs_scaler: ScalerProtocol,
        squeeze_last_dim: bool = True,
        channels_last: bool = False,
    ):
        """
        Scale and reshape output variables consistently across channels.

        - Accepts a tuple/list of per-channel arrays with original shapes like (N, *mesh_shape).
        - Flattens each channel to 2D ([-1, 1]) and, if a scaler is provided, fits once with a list
        (one block per channel) and transforms accordingly (per-channel min/max).
        - Rebuilds a torch.Tensor with shape (N, C, *mesh_shape) by default.
          If ``channels_last=True`` it returns (N, *mesh_shape, C).
        """
        # 1) Prepare flattened list (one 2D block per channel)
        np_vars = [np.asarray(v) for v in variables_out]
        flat_list = [arr.reshape(-1, 1) for arr in np_vars]

        # 2) Fit (once) and transform using list-mode (as required by the protocol)
        if outputs_scaler is not None:
            if not outputs_scaler.is_fitted:
                outputs_scaler.fit(flat_list)
            flat_scaled = outputs_scaler.transform(flat_list)
        else:
            flat_scaled = flat_list  # pass-through

        # 3) Rebuild tensors and stack along the channel axis
        stacked = []
        for scaled, orig in zip(flat_scaled, np_vars):
            s = np.asarray(scaled).reshape(orig.shape)     # back to (N, *mesh_shape)
            t = torch.as_tensor(s)
            t = t.reshape(-1, *self.mesh_shape)
            stacked.append(t)

        out = torch.stack(stacked, dim=-1 if channels_last else 1)
        if squeeze_last_dim:
            if channels_last and out.shape[-1] == 1:
                out = out.squeeze(-1)
            if (not channels_last) and out.shape[1] == 1:
                out = out.squeeze(1)
        return out.float()



    def _process_parameters(self, parameters, combine_parameters_with_cartesian_prod):
        if parameters is None:
            return None
        if not combine_parameters_with_cartesian_prod:
            # assert that all elements in the inner lists have the same length
            if not all(len(parameters[0]) == len(inner_list) for inner_list in parameters):
                raiseError("All parameter lists must have the same length")
            parameters = torch.tensor(list(zip(*parameters)))
            
        else:
            if len(parameters) == 1:
                parameters = torch.tensor(parameters[0])
            else:
                parameters = torch.tensor(list(product(*parameters)))
        return parameters.float()

    @property
    def shape(self):
        return self.variables_out.shape
    
    @property
    def num_parameters(self):
        return self.parameters.shape[1] if self.parameters is not None else 0
    
    @property
    def has_input_scaler(self) -> bool:
        """True if an input scaler is attached and fitted."""
        return self.inputs_scaler is not None and self.inputs_scaler.is_fitted

    @property
    def has_output_scaler(self) -> bool:
        """True if an output scaler is attached and fitted."""
        return self.outputs_scaler is not None and self.outputs_scaler.is_fitted


    def __len__(self):
        return len(self.variables_out)

    def __getitem__(self, idx: Union[int, slice]):
        """
        Return the input data and the output data for a given index as a tuple. If there is no input data, only the output data will be returned.
        If parameters are used, the parameters will be concatenated with the input data at the end.

        Args:
            idx (int, slice): Index of the data to be returned.
        """            
        if self.variables_in is None:
            return self.variables_out[idx]
        
        if isinstance(idx, slice):
            idx = torch.arange(
                idx.start if idx.start is not None else 0, 
                idx.stop if idx.stop is not None else len(self),
                idx.step if idx.step is not None else 1
            )
        else:
            idx = torch.tensor(idx) if not isinstance(idx, torch.Tensor) else idx
        
        variables_in_idx = idx % len(self.variables_in)
        parameters_idx = idx // len(self.variables_in)
        input_data = torch.hstack([self.variables_in[variables_in_idx], self.parameters[parameters_idx]]) if self.parameters is not None else self.variables_in[variables_in_idx]
        return input_data, self.variables_out[idx]


    def __setitem__(self, idx: Union[int, slice], value: Tuple):
        """
        Set the input data and the output data for a given index. If there is no input data, only the output data will be set.
        If parameters are used, the parameters will be concatenated with the input data at the end.

        Args:
            idx (int, slice): Index of the data to be set.
            value (Tuple): Tuple with the input data and the output data. If there is no input data, the tuple should have only one element.
        """
        if self.variables_in is None:
            self.variables_out[idx] = value
        else:
            if len(value) != 2:
                raiseError("Invalid value to set. Must be a tuple with two elements when variables_in takes some value")
            if isinstance(idx, slice):
                idx = torch.arange(
                    idx.start if idx.start is not None else 0, 
                    idx.stop if idx.stop is not None else len(self),
                    idx.step if idx.step is not None else 1
                )
            else:
                idx = torch.tensor(idx) if not isinstance(idx, torch.Tensor) else idx
            input_data, variables_out = value
            variables_in_idx = idx % len(self.variables_in)
            parameters_idx = idx // len(self.variables_in)

            self.variables_in[variables_in_idx] = input_data[:, :self.variables_in.shape[1]]
            if self.parameters is not None:
                self.parameters[parameters_idx] = input_data[:, self.variables_in.shape[1]:]
            self.variables_out[idx] = variables_out

        
    def __add__(self, other):
        """
        Concatenate two datasets. The datasets must have the same number of input coordinates and parameters.

        Args:
            other (Dataset): Dataset to be concatenated with.
        """
        if not isinstance(other, Dataset):
            raiseError(f"Cannot add Dataset with {type(other)}")
        if self.variables_in is not None:
            if other.variables_in is None or other.variables_in.shape[1] != self.variables_in.shape[1]:
                raiseError(f"Cannot add datasets with different number of columns on input coordinates, got {other.variables_in.shape[1]} and {self.variables_in.shape[1]}")
            if self.parameters is not None:
                if other.parameters is None or other.parameters.shape[1] != self.parameters.shape[1]:
                    raiseError("Cannot add datasets with different number of parameters")
                if other.variables_in is None or other.variables_in.shape[0] != self.variables_in.shape[0]:
                    raiseError(f"Cannot add datasets with different number of input coordinates, got {other.variables_in.shape[0]} and {self.variables_in.shape[0]}")
                self.parameters = torch.cat([self.parameters, other.parameters], dim=0)
            else:
                self.variables_in = torch.cat([self.variables_in, other.variables_in], dim=0)

        variables_out = torch.cat([self.variables_out, other.variables_out], dim=0)
        self.variables_out = variables_out
        return self
    
    def concatenate(self, other):
        """
        Alias for the `__add__` method.

        Args:
            other (Dataset): Dataset to be concatenated with.
        """
        return self.__add__(other)

    def _crop2D(self, nh, nw):
        self.variables_out = self.variables_out[:, :, :nh, :nw]
        self.mesh_shape    = (nh, nw)

    def _crop3D(self, nh, nw, nd):
        self.variables_out = self.variables_out[:, :, :nh, :nw, :nd]
        self.mesh_shape    = (nh, nw, nd)

    def crop(self, *args):
        """
        Crop the dataset to a desired shape. The cropping currently works for 2D and 3D meshes.

        Args:
            args (Tuple): Desired shape of the mesh. If the mesh is 2D, the shape should be a tuple with two elements. If the mesh is 3D, the shape should be a tuple with three elements.
        """
        if len(args) == 2: 
            self._crop2D(*args)
        elif len(args) == 3:
            self._crop3D(*args)
        else:
            raiseError(f'Invalid number of dimensions {len(args)} for mesh {self.mesh_shape}')

    def _pad2D(self, n0h, n0w):
        nh, nw = self.mesh_shape
        self.variables_out = F.pad(self.variables_out, (0, n0w-nw, 0, n0h-nh), mode='constant', value=0)
        self.mesh_shape    = (n0h,n0w)

    def _pad3D(self, n0h, n0w, n0d):
        nh, nw, nd = self.mesh_shape
        self.variables_out = F.pad(self.variables_out, (0, n0d-nd, 0, n0w-nw, 0, n0h-nh), mode='constant', value=0)
        self.mesh_shape    = (n0h, n0w, n0d)

    def pad(self, *args):
        """
        Pad the dataset to a desired shape. The padding currently works for 2D and 3D meshes.

        Args:
            args (Tuple): Desired shape of the mesh. If the mesh is 2D, the shape should be a tuple with two elements. If the mesh is 3D, the shape should be a tuple with three elements.
        """
        if len(args) == 2: 
            self._pad2D(*args)
        elif len(args) == 3:
            self._pad3D(*args)
        else:
            raiseError(f'Invalid number of dimensions {len(args)} for mesh {self.mesh_shape}')

    def get_splits(
        self,
        sizes: Sequence[int],
        shuffle: bool = True,
        return_views: bool = True,
        generator: Optional[Generator] = default_generator,
    ):
        """
        Randomly split the dataset into non-overlapping new datasets of given lengths.

        If a list of fractions that sum up to 1 is given,
        the lengths will be computed automatically as
        floor(frac * len(self)) for each fraction provided.

        After computing the lengths, if there are any remainders, 1 count will be
        distributed in round-robin fashion to the lengths
        until there are no remainders left.

        Optionally fix the generator for reproducible results.

        Args:
            sizes (sequence): lengths or fractions of splits to be produced.
            shuffle (bool): whether to shuffle the data before splitting. Default: ``True``.
            return_views (bool): If True, the splits will be returned as views of the original dataset in form of torch.utils.data.Subset.
            If False, the splits will be returned as new Dataset instances. Default: ``True``.
            Warning: If return_views is True, the original dataset must be kept alive, otherwise the views will point to invalid memory.
            Be careful when using `return_views=False` with `variables_in` and `parameters`, the memory usage will be higher.
            generator (Generator): Generator used for the random permutation. Default: ``default_generator``.
        
        Returns:
            List[Dataset | Subset]: List with the splits of the dataset.
        """
        if np.isclose(sum(sizes), 1) and sum(sizes) <= 1:
            subset_lengths: List[int] = []
            for i, frac in enumerate(sizes):
                if frac < 0 or frac > 1:
                    raise ValueError(f"Fraction at index {i} is not between 0 and 1")
                n_items_in_split = int(
                    np.floor(len(self) * frac)  # type: ignore[arg-type]
                )
                subset_lengths.append(n_items_in_split)
            remainder = len(self) - sum(subset_lengths)  # type: ignore[arg-type]
            # add 1 to all the lengths in round-robin fashion until the remainder is 0
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths)
                subset_lengths[idx_to_add_at] += 1
            sizes = subset_lengths
            for i, length in enumerate(sizes):
                if length == 0:
                    raiseWarning(
                        f"Length of split at index {i} is 0. "
                        f"This might result in an empty dataset."
                    )

        # Cannot verify that dataset is Sized
        if sum(sizes) != len(self):  # type: ignore[arg-type]
            raiseError(
                "Sum of input lengths does not equal the length of the dataset!"
            )
        if shuffle:
            indices = randperm(sum(sizes), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
        else:
            indices = list(range(sum(sizes)))
        sizes = cast(Sequence[int], sizes)

        split_datasets = []
        for offset, length in zip(accumulate(sizes), sizes):
            new_indices = indices[offset - length : offset]
            split_datasets.append(self._create_subset_from_indices(new_indices, return_views))
        return split_datasets

    def get_splits_by_parameters(
        self, 
        sizes: Sequence[int], 
        shuffle: bool = True, 
        return_views: bool = True,
        generator: Optional[Generator] = default_generator
    ):
        """
        Split the dataset into non-overlapping new datasets with diferent sets of parameters of given length.

        If a list of fractions that sum up to 1 is given,
        the lengths will be computed automatically as
        floor(frac * len(self.parameters)) for each fraction provided.

        After computing the lengths, if there are any remainders, 1 count will be
        distributed in round-robin fashion to the lengths
        until there are no remainders left.

        Optionally fix the generator for reproducible results.

        Args:
            sizes (sequence): lengths or fractions of splits to be produced
            shuffle (bool): whether to shuffle the data before splitting. Default: ``True``
            return_views (bool): If True, the splits will be returned as views of the original dataset in form of torch.utils.data.Subset.
            If False, the splits will be returned as new Dataset instances. Default: ``True``.
            Warning: If return_views is True, the original dataset must be kept alive, otherwise the views will point to invalid memory.
            generator (Generator): Generator used for the random permutation. Default: ``default_generator``.

        Returns:
            List[Dataset | Subset]: List with the splits of the dataset.
        """
        if self.parameters is None:
            raiseError("This Dataset does not have parameters to split by")
        if np.isclose(sum(sizes), 1) and sum(sizes) <= 1:
            subset_lengths: List[int] = []
            for i, frac in enumerate(sizes):
                if frac < 0 or frac > 1:
                    raise ValueError(f"Fraction at index {i} is not between 0 and 1")
                n_items_in_split = int(
                    np.floor(len(self.parameters) * frac)  # type: ignore[arg-type]
                )
                subset_lengths.append(n_items_in_split)
            remainder = len(self.parameters) - sum(subset_lengths)
            # add 1 to all the lengths in round-robin fashion until the remainder is 0
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths)
                subset_lengths[idx_to_add_at] += 1
            sizes = subset_lengths
            for i, length in enumerate(sizes):
                if length == 0:
                    raiseWarning(
                        f"Length of split at index {i} is 0. "
                        f"This might result in an empty dataset."
                    )

        # Cannot verify that dataset is Sized
        if sum(sizes) != len(self.parameters):
            raiseError(
                "Sum of input lengths does not equal the length of the number of parameters!"
            )
        if shuffle:
            indices = randperm(sum(sizes), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
        else:
            indices = list(range(sum(sizes)))
        sizes = cast(Sequence[int], sizes)
        split_datasets = []
        for offset, length in zip(accumulate(sizes), sizes):
            new_indices = []
            for index in indices[offset - length : offset]:
                vars_in_extended_indices = list(range(index * len(self.variables_in), (index + 1) * len(self.variables_in)))
                new_indices.extend(vars_in_extended_indices)
            if not return_views:
                # Get the split data
                if self.variables_in is not None:
                    split_variables_in = self.variables_in
                    split_parameters = self.parameters[indices[offset - length : offset]]
                else:
                    split_variables_in = None
                    split_parameters = None
                    
                # Get the split output variables
                split_variables_out = self.variables_out[new_indices]

                # Create a new dataset instance with the split data
                split_dataset = Dataset(
                    variables_out=tuple([split_variables_out[:, i].reshape(-1, *self.mesh_shape).numpy()
                                    for i in range(self.num_channels)]),
                    mesh_shape=self.mesh_shape,
                    variables_in=split_variables_in.numpy() if split_variables_in is not None else None,
                    # Parameters must be transposed to pass only as many list as the number of variables
                    parameters=split_parameters.T.tolist() if split_parameters is not None else None,
                    combine_parameters_with_cartesian_prod=False,
                    snapshots_by_column=False  # Since the data is already reshaped
                )
                split_datasets.append(split_dataset)
            else:
                split_datasets.append(Subset(self, new_indices))

        return split_datasets
    
    @classmethod
    def load(
        cls,
        file_path,
        field_names: List[str],
        variables_names: List[str] = ['all'],
        add_variables: bool = False,
        add_mesh_coordinates: bool = True,
        **kwargs,
    ):
        """
        Create a Dataset from a saved `pyLOM.Dataset` in one of its formats.

        Args:
            file_path (str): Path to the HDF5 file.
            variables_out_names (List[str]): Names of the fields to be used as output. E.g. ``["CP"]``.
            add_variables (bool): Whether to add the variables as input variables. Default is ``False``.
            variables_names (List[str]): Names of the variables from pyLOM.Dataset.varnames to be used as input. If ``["all"]`` is passed, all variables will be used. Default is ``["all"]``.
            kwargs: Additional arguments to be passed to the pyLOM.NN.Dataset constructor.

        Returns:
            Dataset: Dataset created from the saved `pyLOM.Dataset`.

        Example:
            >>> dataset = pyLOM.NN.Dataset.load(
            ...     file_path,
            ...     field_names=["CP"],
            ...     add_variables=False,
            ...     add_mesh_coordinates=True,
            ...     inputs_scaler=inputs_scaler,
            ...     outputs_scaler=outputs_scaler,
            ... )
        """
        original_dataset = pyLOMDataset.load(file_path)
        if (len(variables_names) >= 0 and "combine_parameters_with_cartesian_prod" in kwargs):
            raiseWarning(
                """Be careful when using combine_parameters_with_cartesian_prod and passing variables_names.
                You need to make sure that the length of cartesian product multiplied by the length 
                of the mesh coordinates is equal to the length of the fields on field_names."""
            )
        if variables_names == ["all"]:
            if not add_mesh_coordinates:
                raiseError("Cannot use all variables without adding mesh coordinates")
            if len(original_dataset.varnames) == 0:
                raiseError("No variabele found in the dataset")
            variables_names = original_dataset.varnames

        variables_in = None
        parameters = None
        if add_variables:
            variables_cols = [original_dataset.get_variable(var_name) for var_name in variables_names]
            variables_in = np.stack(variables_cols, axis=1) if len(variables_cols) > 0 else None
            if add_mesh_coordinates:
                if variables_in is None:
                    variables_in = original_dataset.xyz
                else:
                    variables_in = np.hstack([variables_in, original_dataset.xyz])
        else:
            parameters = [original_dataset.get_variable(var_name) for var_name in variables_names]
            parameters = parameters if len(parameters) > 0 else None
            if add_mesh_coordinates:
                variables_in = original_dataset.xyz
            else:
                variables_in = None
        
        variables_out = tuple(
            [original_dataset[var_name] for var_name in field_names]
        )

        # if add_variables:
        #     print("stacking variables_out")
        #     variables_out = np.stack(variables_out, axis=2) if len(variables_out) > 0 else None
        #     variables_out_tuple = (variables_out,)
        # else:
        #     variables_out_tuple = variables_out

        variables_out_tuple = variables_out
        # for i, var_name in enumerate(field_names):
        #     print(f"Loaded output variable '{var_name}' with shape {original_dataset[var_name].shape}")
        #     print(f"Variable {i}-th in variables_out shape: {variables_out_tuple[i].shape}")
        #     print(f"Variable {i}-th in variables_out_tuple shape: {variables_out_tuple[i].shape}")
        return cls(
            variables_out=variables_out_tuple,
            parameters=parameters,
            variables_in=variables_in,
            **kwargs,
        )
    
    @cr("NN.Dataset.map")
    def map(self, function: Callable, fn_kwargs: dict = {}, batched: bool = False, batch_size: int = 1000):
        '''
        Apply a function to the dataset.

        Args:
            function (Callable): Function to be applied to the dataset with one of the following signatures:
                
                - function (inputs: torch.Tensor, outputs: torch.Tensor, \*\*kwargs) -> Tuple[torch.Tensor, torch.Tensor] if variables_in exists. Here `inputs` is the input data and `outputs` is the output data that __getitem__ returns, so `inputs` will include the parameters if they exist. 
                - function (outputs: torch.Tensor, \*\*kwargs) -> torch.Tensor if variables_in does not exist.
                
                If batched is False, the tensors will have only one element in the first dimension.
            fn_kwargs (Dict): Additional keyword arguments to be passed to the function.
            batched (bool): If True, the function will be applied to the dataset in batches. Default is ``False``.
            batch_size (int): Size of the batch. Default is ``1000``.

        Returns:
            Dataset: A reference to the dataset with th e function applied.
        '''

        if not batched:
            batch_size = 1
        for i in range(0, len(self), batch_size):
            batch = self[i:i + batch_size]
            if self.variables_in is not None:
                inputs, outputs = batch
                fn_outputs = function(inputs, outputs, **fn_kwargs)
            else:
                fn_outputs = function(batch, **fn_kwargs)
            self[i:i + batch_size] = fn_outputs

        return self

    @cr("NN.Dataset.filter")
    def filter(
        self,
        function: Callable,
        fn_kwargs: dict = {},
        batched: bool = False,
        batch_size: int = 1000,
        return_views: bool = True,
    ):
        """
        Filter the dataset using a function.

        Args:
            function (Callable): Function to be applied to the dataset with one of the following signatures:

                - function (inputs: torch.Tensor, outputs: torch.Tensor, \*\*kwargs) -> bool if variables_in exists. Here `inputs` is the input data and `outputs` is the output data that __getitem__ returns, so `inputs` will include the parameters if they exist. 
                - function (outputs: torch.Tensor, \*\*kwargs) -> bool if variables_in does not exist.

                If batched is True, the function should return a list of booleans.
            fn_kwargs (Dict): Additional keyword arguments to be passed to the function.
            batched (bool): If True, the function will be applied to the dataset in batches. Default is ``False``.
            batch_size (int): Size of the batch. Default is ``1000``.
            return_views (bool): If True, the filtered dataset will be returned as a view of the original dataset in form of torch.utils.data.Subset. Default is ``True``.
            Warning: If return_views is True, the original dataset must be kept alive, otherwise the views will point to invalid memory.
            Be careful when using `return_views=False` with `variables_in` and `parameters`, the memory usage will be higher.
            
        Returns:
            Subset | Dataset: Subset of the dataset that passed the filter.
        """

        if not batched:
            batch_size = 1
        indices = []
        for i in range(0, len(self), batch_size):
            batch = self[i:i + batch_size]
            if self.variables_in is not None:
                inputs, outputs = batch
                function_outputs = function(inputs, outputs, **fn_kwargs)
            else:
                outputs = batch
                function_outputs = function(outputs, **fn_kwargs)
            indices.extend(filter(lambda x: function_outputs[x], range(i, i + batch_size)))
        
        return self._create_subset_from_indices(indices, return_views)

    def _create_subset_from_indices(self, indices, return_views):
        if not return_views:
            data = self[indices]
            if self.variables_in is None:
                dataset = Dataset(
                    variables_out=tuple([data[:, i].reshape(-1, *self.mesh_shape).numpy()
                                for i in range(self.num_channels)]),
                    mesh_shape=self.mesh_shape,
                    variables_in=None,
                    combine_parameters_with_cartesian_prod=False,
                    snapshots_by_column=False  # Since the data is already reshaped
                )
            else:
                dataset = Dataset(
                    variables_out=tuple([data[1][:, i].reshape(-1, *self.mesh_shape).numpy()
                                for i in range(self.num_channels)]),
                    mesh_shape=self.mesh_shape,
                    variables_in=data[0].numpy(),
                    combine_parameters_with_cartesian_prod=False,
                    snapshots_by_column=False  # Since the data is already reshaped
                )
            return dataset
        else:
            return Subset(self, indices)
        
    def remove_column(self, column_idx: int, from_variables_out: bool):
        """
        Remove a column from the dataset.

        Args:
            column_idx (int): Index of the column to be removed.
            from_variables_out (bool): If True, the column will be removed from the output variables. If False, the column will be removed from the input variables, if `column_idx` is greater than the number of columns on variables_in, the column will be removed from parameters.
        """
        if from_variables_out:
            if column_idx >= self.variables_out.shape[1]:
                raiseError(f"Invalid column index {column_idx}, there are only {self.variables_out.shape[1]} columns in variables_out")
            self.variables_out = torch.cat(
                [self.variables_out[:, :column_idx], self.variables_out[:, column_idx + 1 :]], dim=1
            )
        elif column_idx < self.variables_in.shape[1]:
            self.variables_in = torch.cat(
                [self.variables_in[:, :column_idx], self.variables_in[:, column_idx + 1 :]], dim=1
            )
        elif column_idx < self.parameters.shape[1] + self.variables_in.shape[1]:
            self.parameters = torch.cat(
                [self.parameters[:, :column_idx - self.variables_in.shape[1]], self.parameters[:, column_idx - self.variables_in.shape[1] + 1 :]], dim=1
            )
        else:
            raiseError(f"Invalid column index {column_idx}, there are only {self.variables_in.shape[1]} columns in variables_in and {self.parameters.shape[1]} columns in parameters")
    
    def print_stats(self, dataset_name="Dataset"):
        '''
        Auxiliary function to print the statistics of the dataset.
        '''
        x, y = self[:]
        inputs_min = x.min(dim=0).values
        inputs_max = x.max(dim=0).values
        outputs_min = y.min(dim=0).values
        outputs_max = y.max(dim=0).values

        # Format the values to 2 decimal places
        inputs_min_str = ', '.join([f'{val:.2f}' for val in inputs_min])
        inputs_max_str = ', '.join([f'{val:.2f}' for val in inputs_max])
        outputs_min_str = f'{outputs_min[0]:.2f}'
        outputs_max_str = f'{outputs_max[0]:.2f}'

        pprint(
            0,
            f'{dataset_name} \n'
            f'  Inputs shape : {x.shape} \n'
            f'  Inputs min   : {inputs_min_str} \n'
            f'  Inputs max   : {inputs_max_str} \n'
            f'  Outputs shape: {y.shape} \n'
            f'  Outputs min  : {outputs_min_str} \n'
            f'  Outputs max  : {outputs_max_str} \n'
        )

    def inverse_scale_inputs(self, x: Union[np.ndarray, torch.Tensor], scaler: ScalerProtocol = None):
        """
        Inverse-scale a 2D inputs matrix using the provided scaler or self.inputs_scaler.
        Accepts numpy or torch and returns the same container type.
        """
        sc = scaler or self.inputs_scaler
        if sc is None or not sc.is_fitted:
            return x

        was_numpy = isinstance(x, np.ndarray)
        X = torch.as_tensor(x)
        if X.ndim != 2:
            X = X.reshape(-1, X.shape[-1])

        X_inv = sc.inverse_transform(X)               # protocol guarantees this
        X_inv_t = torch.as_tensor(X_inv, dtype=X.dtype)
        return X_inv_t.detach().cpu().numpy() if was_numpy else X_inv_t

    def filter_by_spatial_mask(self, mask, *, inplace: bool = False, keep_scalers: bool = False) -> "Dataset":
        """Return a dataset filtered along the spatial/mesh dimension.

        The mask must flag the mesh cells/points to keep. When ``variables_in`` is present,
        the mask length must match the number of input coordinates. Otherwise it must match
        ``np.prod(mesh_shape)``.

        Args:
            mask: Boolean-like 1D array/tensor selecting the spatial positions to keep.
            inplace: If True, modify the current dataset and return it. Defaults to False.
            keep_scalers: If False, drop attached scalers in the filtered dataset so they can
                be re-fitted on the reduced data. Defaults to False.
        """
        target = self if inplace else copy.deepcopy(self)

        mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
        mask_flat = mask_tensor.reshape(-1)
        if mask_flat.numel() == 0 or mask_flat.sum() == 0:
            raiseError("Spatial mask is empty; nothing to keep.")

        spatial_size = int(np.prod(target.mesh_shape))

        def _filter_outputs(size_to_match: int):
            mask_for_out = mask_flat.to(target.variables_out.device)
            rest_shape = target.variables_out.shape[1 + len(target.mesh_shape) :]
            reshaped = target.variables_out.reshape(
                target.variables_out.shape[0], size_to_match, *rest_shape
            )
            filtered = reshaped[:, mask_for_out, ...]
            kept = int(mask_flat.sum().item())
            target.variables_out = filtered.reshape(
                target.variables_out.shape[0], kept, *rest_shape
            )
            target.mesh_shape = (kept,) if len(target.mesh_shape) == 1 else target.mesh_shape

        if target.variables_in is None:
            if mask_flat.numel() != spatial_size:
                raiseError(
                    f"Mask length {mask_flat.numel()} does not match mesh size {spatial_size}."
                )
            _filter_outputs(spatial_size)
        else:
            n_points = target.variables_in.shape[0]

            if mask_flat.numel() == spatial_size and spatial_size != n_points:
                _filter_outputs(spatial_size)
            elif mask_flat.numel() == n_points:
                mask_for_in = mask_flat.to(target.variables_in.device)
                target.variables_in = target.variables_in[mask_for_in]
                if spatial_size == n_points:
                    target.mesh_shape = (int(mask_flat.sum().item()),)

                total_samples = target.variables_out.shape[0]
                rest_shape = target.variables_out.shape[1:]
                if total_samples % n_points != 0:
                    raiseError(
                        "Cannot reshape outputs with the provided mask; dataset length is not a multiple of the number of input points."
                    )
                blocks = total_samples // n_points
                mask_for_out = mask_flat.to(target.variables_out.device)
                reshaped = target.variables_out.reshape(blocks, n_points, *rest_shape)
                filtered = reshaped[:, mask_for_out, ...]
                target.variables_out = filtered.reshape(-1, *rest_shape)
            else:
                raiseError(
                    f"Mask length {mask_flat.numel()} does not match mesh size {spatial_size} or number of input points {n_points}."
                )

        if not keep_scalers:
            target.inputs_scaler = None
            target.outputs_scaler = None

        return target

    def inverse_scale_outputs(
        self,
        y: Union[np.ndarray, torch.Tensor],
        scaler: ScalerProtocol = None,
        mesh_shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Inverse-scale an output tensor using the provided scaler or self.outputs_scaler.
        Accepts channel-first (N, C, *mesh_shape), channel-last (N, *mesh_shape, C),
        or no-explicit-channel (N, *mesh_shape), and returns the same container type.
        """
        sc = scaler or self.outputs_scaler
        if sc is None or not sc.is_fitted:
            return y

        mshape = mesh_shape or self.mesh_shape
        was_numpy = isinstance(y, np.ndarray)
        t = torch.as_tensor(y)
        ndim_expected_no_channel = len(mshape) + 1
        ndim_expected_with_channel = len(mshape) + 2

        if t.ndim == ndim_expected_no_channel:
            # (N, *mesh_shape)
            flat_list = [t.reshape(-1, 1)]
            inv_list = sc.inverse_transform(flat_list)
            out = torch.as_tensor(inv_list[0]).reshape(t.shape)
        elif t.ndim == ndim_expected_with_channel:
            # (N, C, *mesh_shape) or (N, *mesh_shape, C)
            channels_last = bool(getattr(self, "channels_last", False))
            if channels_last:
                C = t.shape[-1]
                flat_list = [t[..., i].reshape(-1, 1) for i in range(C)]
                inv_list = sc.inverse_transform(flat_list)
                restored = [torch.as_tensor(inv).reshape(t.shape[:-1]) for inv in inv_list]
                out = torch.stack(restored, dim=-1)
            else:
                C = t.shape[1]
                flat_list = [t[:, i, ...].reshape(-1, 1) for i in range(C)]
                inv_list = sc.inverse_transform(flat_list)
                restored = [torch.as_tensor(inv).reshape((t.shape[0],) + tuple(t.shape[2:])) for inv in inv_list]
                out = torch.stack(restored, dim=1)
        else:
            raiseError(
                f"Unsupported output shape {tuple(t.shape)} for mesh_shape {mshape}. "
                "Expected (N, *mesh_shape), (N, C, *mesh_shape), or (N, *mesh_shape, C)."
            )

        return out.detach().cpu().numpy() if was_numpy else out

    
    def save_scalers(self, prefix: str):
        """
        Persist attached scalers to JSON files if they provide a 'save' method and are fitted.
        Files produced: f"{prefix}_inputs_scaler.json" and f"{prefix}_outputs_scaler.json".
        """
        if self.inputs_scaler is not None and hasattr(self.inputs_scaler, "save") and self.inputs_scaler.is_fitted:
            self.inputs_scaler.save(f"{prefix}_inputs_scaler.json")
        if self.outputs_scaler is not None and hasattr(self.outputs_scaler, "save") and self.outputs_scaler.is_fitted:
            self.outputs_scaler.save(f"{prefix}_outputs_scaler.json")
class NeighborhoodDataset(Dataset):
    def __init__(
        self,
        *args,
        n_neighbors: int = 6,
        geom_dim: int = None,
        neighbors: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.variables_in is None:
            raise ValueError("NeighborhoodDataset requires input variables (coordinates).")

        input_dim = self.variables_in.shape[1]

        if geom_dim is None:
            if input_dim < 2:
                raise ValueError("At least 2 input columns are required for geometry.")
            self.geom_dim = min(3, input_dim)
        else:
            if geom_dim not in (2, 3):
                raise ValueError("geom_dim must be 2 or 3.")
            if input_dim < geom_dim:
                raise ValueError(
                    f"geom_dim={geom_dim} but only {input_dim} input columns available."
                )
            self.geom_dim = geom_dim

        self.n_neighbors = n_neighbors
        self.neighbors = neighbors if neighbors is not None else self._build_neighbors()
        self.use_neighbors = False

    def __getitem__(self, idx: Union[int, slice]):
        if self.use_neighbors:
            return self.get_with_neighbors(idx)
        else:
            return super().__getitem__(idx)
        
    def _build_neighbors(self):
        """
        Precompute geometric neighbors using only geometric coordinates.
        """
        coords = self.variables_in[:, :self.geom_dim].cpu().numpy()

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm="ball_tree")
        nbrs.fit(coords)

        _, indices = nbrs.kneighbors(coords)
        return torch.tensor(indices[:, 1:], dtype=torch.long)

    def _build_new(self, **kwargs):
        return NeighborhoodDataset(
            n_neighbors=self.n_neighbors,
            geom_dim=self.geom_dim,
            neighbors=self.neighbors,
            **kwargs,
        )

    def train(self):
        self.use_neighbors = True

    def eval(self):
        self.use_neighbors = False

    def get_with_neighbors(self, idx: Union[int, slice, List[int], torch.Tensor]):

        # Normalize idx
        if isinstance(idx, slice):
            idx = torch.arange(
                idx.start if idx.start is not None else 0,
                idx.stop if idx.stop is not None else len(self),
                idx.step if idx.step is not None else 1,
                dtype=torch.long,
            )
        elif isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)
        elif isinstance(idx, (list, tuple)):
            idx = torch.tensor(idx, dtype=torch.long)
        elif isinstance(idx, torch.Tensor):
            idx = idx.long()
        else:
            idx = torch.tensor(idx, dtype=torch.long)

        if idx.dim() == 0:
            idx = idx.unsqueeze(0)

        single_index = idx.numel() == 1

        # Obtain center data (vectorized)
        center_input, center_output = super().__getitem__(idx)

        # Obtain neighbor data (vectorized)
        batch_size = idx.shape[0]
        mesh_length = len(self.neighbors)
        neighbor_idx = idx % mesh_length
        parameters_idx = idx // mesh_length

        neighbor_indices = (self.neighbors[neighbor_idx] + mesh_length * parameters_idx.unsqueeze(-1))
        n_neighbors = neighbor_indices.shape[1]
        flat_neighbor_indices = neighbor_indices.reshape(-1)

        neighbor_inputs_flat, neighbor_outputs_flat = super().__getitem__(
            flat_neighbor_indices
        )

        # Reshape back
        input_dim = neighbor_inputs_flat.shape[-1]
        output_dim = neighbor_outputs_flat.shape[-1]

        neighbor_inputs = neighbor_inputs_flat.view(
            batch_size, n_neighbors, input_dim
        )
        neighbor_outputs = neighbor_outputs_flat.view(
            batch_size, n_neighbors, output_dim
        )

        # Single index case
        if single_index:
            center_input = center_input.squeeze(0)
            center_output = center_output.squeeze(0)
            neighbor_inputs = neighbor_inputs.squeeze(0)
            neighbor_outputs = neighbor_outputs.squeeze(0)

        return (
            center_input,
            neighbor_inputs,
            center_output,
            neighbor_outputs,
        )