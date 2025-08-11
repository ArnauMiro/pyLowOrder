#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines.
#
# Last rev:

# Built-in modules
from itertools import product, accumulate
from typing import List, Optional, Tuple, Union, Callable, Sequence, cast

# Third-party libraries
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch import Generator, randperm, default_generator

# Local modules

from ..dataset import Dataset as pyLOMDataset
from ..utils.cr import cr
from ..utils.errors import raiseWarning, raiseError









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
        inputs_scaler=None,
        outputs_scaler=None,
        snapshots_by_column=True,
        squeeze_last_dim: bool = True, 
    ):
        self.parameters = parameters
        self.num_channels = len(variables_out)
        self.mesh_shape = mesh_shape
        if snapshots_by_column:
            variables_out = [variable.T for variable in variables_out]
        if outputs_scaler is not None:
            if not outputs_scaler.is_fitted:
                outputs_scaler.fit(variables_out)
            variables_out = outputs_scaler.transform(variables_out)
        self.variables_out = self._process_variables_out(variables_out, squeeze_last_dim)
        if variables_in is not None:
            self.parameters = self._process_parameters(parameters, combine_parameters_with_cartesian_prod)
            self.variables_in = torch.tensor(variables_in, dtype=torch.float32)
            if inputs_scaler is not None:
                variables_in_columns = [self.variables_in[:, i] for i in range(self.variables_in.shape[1])]
                parameters_columns = [self.parameters[:, i] for i in range(self.parameters.shape[1])] if self.parameters is not None else []
                if not inputs_scaler.is_fitted:
                    inputs_scaler.fit(variables_in_columns + parameters_columns)
                input_data_transformed = inputs_scaler.transform(variables_in_columns + parameters_columns)
                self.variables_in = torch.stack(input_data_transformed[:self.variables_in.shape[1]], dim=1).float()
                if self.parameters is not None:
                    self.parameters = torch.stack(input_data_transformed[self.variables_in.shape[1]:], dim=1)
        else:
            self.variables_in = None
            if parameters is not None:
                raiseWarning("Parameters were passed but no input variables were passed. Parameters will be ignored.")
            self.parameters = None

    def _process_variables_out(self, variables_out, squeeze_last_dim=True):
        variables_out_stacked = []
        for variable in variables_out:
            variable = torch.tensor(variable)
            variable = variable.reshape(-1, *self.mesh_shape)
            variables_out_stacked.append(variable)
        variables_out_stacked = torch.stack(variables_out_stacked, dim=1)

        if squeeze_last_dim:  # If the last dimension is 1, squeeze it
            if variables_out_stacked.shape[-1] == 1:  # (N, C, 1) -> (N, C)
                variables_out_stacked = variables_out_stacked.squeeze(-1)
        return variables_out_stacked.float()

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
            # print("adding variables")
            variables_in = [original_dataset.get_variable(var_name) for var_name in variables_names]
            # print("stacking")
            variables_in = np.stack(variables_in, axis=1) if len(variables_in) > 0 else None
            if add_mesh_coordinates:
                variables_in.append(original_dataset.xyz)
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
        if add_variables:
            variables_out = np.stack(variables_out, axis=1) if len(variables_out) > 0 else None

        return cls(
            variables_out=variables_out,
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
            

