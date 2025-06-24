#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines.
#
# Last rev: 02/10/2024
import os, random, json, numpy as np, torch
import torch.nn.functional as F
from torch_geometric.data import Data

from torch.utils.data import Subset
from torch            import Generator, randperm, default_generator
from itertools        import product, accumulate
from typing           import List, Optional, Tuple, cast, Sequence, Union, Callable, Dict
import warnings

from .                import DEVICE
from ..               import io
from ..utils.cr       import cr
from ..utils.errors   import raiseWarning, raiseError
from ..dataset        import Dataset as pyLOMDataset
from ..mesh           import Mesh
from ..vmmath.geometric import edge_to_cells, wall_normals

class MinMaxScaler:
    r"""
    Min-max scaling to scale variables to a desired range. The formula is given by:

    .. math::

        X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}} * (feature\_range_{max} - feature\_range_{min}) + feature\_range_{min}

    Args:
        feature_range (Tuple): Desired range of transformed data. Default is ``(0, 1)``.
        column (bool, optional): Scale over the column space or the row space (default ``False``)
    """

    def __init__(self, feature_range=(0, 1), column=False):
        self.feature_range = feature_range
        self._is_fitted = False
        self._column    = column

    @property
    def is_fitted(self):
        return self._is_fitted
    
    def fit(self, variables: Union[List[Union[np.ndarray, torch.tensor]], np.ndarray, torch.tensor]):
        """
        Compute the min and max values of each variable.
        Args:
            variables (List): List of variables to be fitted. The variables should be numpy arrays or torch tensors.
            A numpy array or torch tensor can be passed directly and each column will be considered as a variable to be scaled.
        """
        variables = self._cast_variables(variables)
        min_max_values = []
        for variable in variables:
            min_val = float(variable.min())
            max_val = float(variable.max())
            min_max_values.append({"min": min_val, "max": max_val})
        self.variable_scaling_params = min_max_values
        self._is_fitted = True

    def transform(
        self, variables: Union[List[Union[np.ndarray, torch.tensor]], np.ndarray, torch.tensor]
    ) -> List[Union[np.ndarray, torch.tensor]]:
        """
        Scale variables to the range defined on `feature_range` using min-max scaling.
        Args:
            variables (List): List of variables to be scaled. The variables should be numpy arrays or torch tensors.
            A numpy array or torch tensor can be passed directly and each column will be considered as a variable to be scaled.
        Returns:
            scaled_variables: List of scaled variables.
        """
        is_array = isinstance(variables, (np.ndarray))
        is_tensor = isinstance(variables, (torch.Tensor))
        variables = self._cast_variables(variables)
        scaled_variables = []
        for i, variable in enumerate(variables):
            min_val = self.variable_scaling_params[i]["min"]
            max_val = self.variable_scaling_params[i]["max"]
            data_range = max_val - min_val
            data_range = 1 if data_range == 0 else data_range
            scaled_variable = (variable - min_val) / data_range
            # scale the variable to the desired feature_range
            scaled_variable = (
                scaled_variable * (self.feature_range[1] - self.feature_range[0])
                + self.feature_range[0]
            )
            scaled_variables.append(scaled_variable)

        if is_array:
            scaled_variables = np.hstack(scaled_variables)
        elif is_tensor:
            scaled_variables = torch.hstack(scaled_variables)

        return scaled_variables.T if self._column else scaled_variables
    
    def fit_transform(self, variables: List[Union[np.ndarray, torch.tensor]]) -> List[Union[np.ndarray, torch.tensor]]:
        """
        Fit and transform the variables using min-max scaling.
        Args:
            variables (List): List of variables to be fitted and scaled. The variables should be numpy arrays or torch tensors.
            A numpy array or torch tensor can be passed directly and each column will be considered as a variable to be scaled.
        Returns:
            scaled_variables: List of scaled variables.
        """
        self.fit(variables)
        return self.transform(variables)

    def inverse_transform(self, variables: List[Union[np.ndarray, torch.tensor]]) -> List[Union[np.ndarray, torch.tensor]]:
        """
        Inverse scale variables that have been scaled using min-max scaling.
        Args:
            variables (List): List of variables to be inverse scaled. The variables should be numpy arrays or torch tensors.
            A numpy array or torch tensor can be passed directly and each column will be considered as a variable to be scaled.
        Returns:
            inverse_scaled_variables: List of inverse scaled variables.
        """
        is_array = isinstance(variables, (np.ndarray))
        is_tensor = isinstance(variables, (torch.Tensor))
        variables = self._cast_variables(variables)
        if len(variables) != len(self.variable_scaling_params):
            raiseError(
                f"Number of variables to inverse transform ({len(variables)}) does not match the number of variables fitted ({len(self.variable_scaling_params)})"
            )
        inverse_scaled_variables = []
        for variable, scaling_params in zip(variables, self.variable_scaling_params):
            min_val = scaling_params["min"]
            max_val = scaling_params["max"]
            inverse_scaled_variable = (variable - self.feature_range[0]) / (
                self.feature_range[1] - self.feature_range[0]
            )
            data_range = max_val - min_val
            data_range = 1 if data_range == 0 else data_range
            inverse_scaled_variable = inverse_scaled_variable * data_range + min_val
            inverse_scaled_variables.append(inverse_scaled_variable)
        if is_array:
            inverse_scaled_variables = np.hstack(inverse_scaled_variables)
        elif is_tensor:
            inverse_scaled_variables = torch.hstack(inverse_scaled_variables)
        return inverse_scaled_variables.T if self._column else inverse_scaled_variables

    def save(self, filepath: str) -> None:
        """
        Save the fitted scaler parameters to a JSON file.
        
        Args:
            filepath (str): Path where the scaler parameters will be saved
        """
        if not self.is_fitted:
            raiseError("Scaler must be fitted before it can be saved")
        
        save_dict = {
            "feature_range": self.feature_range,
            "variable_scaling_params": self.variable_scaling_params,
            "column": self._column
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=4)

    @staticmethod
    def load(filepath: str) -> 'MinMaxScaler':
        """
        Load a saved MinMaxScaler from a JSON file.
        
        Args:
            filepath (str): Path to the saved scaler parameters
            
        Returns:
            MinMaxScaler: A new MinMaxScaler instance with the loaded parameters

        """
        if not os.path.exists(filepath):
            raiseError(f"No file found at {filepath}")
        
        with open(filepath, 'r') as f:
            loaded_dict = json.load(f)
        
        scaler = MinMaxScaler(feature_range=tuple(loaded_dict["feature_range"]),column=loaded_dict["column"])
        
        # Restore the scaling parameters
        scaler.variable_scaling_params = loaded_dict["variable_scaling_params"]
        scaler._is_fitted = True
        
        return scaler   

    def _cast_variables(self, variables):
        variables = variables.T if self._column else variables
        if isinstance(variables, (torch.Tensor)):
            variables = [variables[:, i].unsqueeze(1) for i in range(variables.shape[1])]
        elif isinstance(variables, (np.ndarray)):
            variables = [np.expand_dims(variables[:, i], axis=1) for i in range(variables.shape[1])]
        return variables


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
        self.variables_out = self._process_variables_out(variables_out)
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

    def _process_variables_out(self, variables_out):
        variables_out_stacked = []
        for variable in variables_out:
            variable = torch.tensor(variable)
            variable = variable.reshape(-1, *self.mesh_shape)
            variables_out_stacked.append(variable)
        variables_out_stacked = torch.stack(variables_out_stacked, dim=1)

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
        add_mesh_coordinates: bool = True,
        **kwargs,
    ):
        """
        Create a Dataset from a saved `pyLOM.Dataset` in one of its formats.

        Args:
            file_path (str): Path to the HDF5 file.
            variables_out_names (List[str]): Names of the fields to be used as output. E.g. ``["CP"]``.
            add_variables (bool): Whether to add the variables as input variables. Default is ``True``.
            variables_names (List[str]): Names of the variables from pyLOM.Dataset.varnames to be used as input. If ``["all"]`` is passed, all variables will be used. Default is ``["all"]``.
            kwargs: Additional arguments to be passed to the pyLOM.NN.Dataset constructor.

        Returns:
            Dataset: Dataset created from the saved `pyLOM.Dataset`.

        Example:
            >>> dataset = pyLOM.NN.Dataset.load(
            ...     file_path,
            ...     field_names=["CP"],
            ...     add_variables=True,
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

        parameters = [original_dataset.get_variable(var_name) for var_name in variables_names]

        return cls(
            variables_out=tuple(
                [original_dataset[var_name] for var_name in field_names]
            ),
            parameters=parameters if len(parameters) > 0 else None,
            variables_in=original_dataset.xyz if add_mesh_coordinates else None,
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

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
 
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
 

def create_results_folder(RESUDIR: str, verbose: bool=True):
    r"""
    Create a folder to store the results of the neural network training.

    Args:
        RESUDIR (str): Path to the folder to be created.
        verbose (bool): If True, print messages to the console. Default is ``True``.
    """    
    if not os.path.exists(RESUDIR):
        os.makedirs(RESUDIR)
        if verbose: 
            print(f"Folder created: {RESUDIR}")
    elif verbose:
        print(f"Folder already exists: {RESUDIR}")


def select_device(device: str = DEVICE):
    r"""
    Select the device to be used for the training.

    Args:
        device (str): Device to be used. Default is cuda if available, otherwise cpu.
    """
    torch.device(device)
    return device


class betaLinearScheduler:
    r"""
    Linear scheduler for beta parameter in the loss function of the Autoencoders.

    Args:
        start_value (float): initial value of beta
        end_value (float): final value of beta
        warmup (int): number of epochs to reach final value
    """

    def __init__(self, start_value, end_value, start_epoch, warmup):
        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.warmup = warmup

    def getBeta(self, epoch):
        r"""
        Get the value of beta for a given epoch.

        Args:
            epoch (int): current epoch
        """
        if epoch < self.start_epoch:
            return 0
        else:
            if epoch < self.warmup:
                beta = self.start_value + (self.end_value - self.start_value) * (
                    epoch - self.start_epoch
                ) / (self.warmup - self.start_epoch)
                return beta
            else:
                return self.end_value


class Graph(Data):
    def __init__(
        self,
        edge_index: torch.Tensor,
        node_attrs: Dict[str, torch.Tensor],
        edge_attrs: Dict[str, torch.Tensor],
        device: Union[str, torch.device] = None,
        # **custom_attrs: Any # Not yet implemented.
    ):
        r'''
        Initialize the Graph object. Node and edge attributes are stacked separately along dimension 1 for use in GNNs.

        Args:
            edge_index (torch.Tensor): COO format edge index (shape [2, num_edges]).
            node_attrs (Dict[str, torch.Tensor]): Dict of node attributes [num_nodes, attr_dim].
            edge_attrs (Dict[str, torch.Tensor]): Dict of edge attributes [num_edges, attr_dim].
            device (Union[str, torch.device], optional): Torch device. Defaults to CUDA if available.
        '''
        if device is None:
            device = torch.device(DEVICE)  # Default to global DEVICE constant
        elif isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA not available. Falling back to CPU.")
            device = torch.device('cpu')

        # Ensure everything is on the correct device
        edge_index = edge_index.to(device)
        node_attrs = {k: v.to(device) for k, v in node_attrs.items()}
        edge_attrs = {k: v.to(device) for k, v in edge_attrs.items()}
        # custom_attrs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in custom_attrs.items()} # Not yet implemented.

        # Concatenate node/edge attributes
        x = torch.cat(list(node_attrs.values()), dim=1) if node_attrs else None
        edge_attr = torch.cat(list(edge_attrs.values()), dim=1) if edge_attrs else None

        # Build kwargs for Data
        data_kwargs = {
            'edge_index': edge_index,
            'x': x,
            'edge_attr': edge_attr,
            'num_nodes': next(iter(node_attrs.values())).shape[0] if node_attrs else None
            # **custom_attrs,
        }

        super().__init__(**data_kwargs)

        # store raw attribute dicts (for i/o operations)
        self.node_attrs_dict = node_attrs
        self.edge_attrs_dict = edge_attrs
        # self.custom_attrs = custom_attrs # Not yet implemented.

        self.validate()

    def validate(self) -> None:
        assert self.edge_index.shape[0] == 2
        assert self.edge_index.dtype == torch.long

        for k, v in self.node_attrs_dict.items():
            assert v.shape[0] == self.num_nodes, f"Node attribute '{k}' has inconsistent length: {v.shape[0]} vs expected {self.num_nodes}"
        for k, v in self.edge_attrs_dict.items():
            assert v.shape[0] == self.edge_index.shape[1], f"Edge attribute '{k}' has inconsistent length: {v.shape[0]} vs expected {self.edge_index.shape[1]}"
        
        if self.edge_attr is not None:
            expected_dim = sum(v.shape[1] if v.ndim > 1 else 1 for v in self.edge_attrs_dict.values())
            assert self.edge_attr.shape[1] == expected_dim
        
        assert all(isinstance(k, str) and k.strip() for k in self.node_attrs_dict), "All node attribute keys must be non-empty strings."
        assert all(isinstance(k, str) and k.strip() for k in self.edge_attrs_dict), "All edge attribute keys must be non-empty strings."
        
        for name, tensor in self.node_attrs_dict.items():
            assert torch.isfinite(tensor).all(), f"Node attribute '{name}' contains NaN or inf."
        for name, tensor in self.edge_attrs_dict.items():
            assert torch.isfinite(tensor).all(), f"Edge attribute '{name}' contains NaN or inf."


    @classmethod
    def from_pyLOM_mesh(cls,
                        mesh: Mesh,
                        device: Optional[Union[str, torch.device]] = None,
                        # **custom_attrs: Dict[str, Any]  # Not yet implemented.
                        ) -> "Graph":
        r"""
        Create a Graph object from a pyLOM Mesh object. This method computes the node attributes and edge index/attributes from the mesh.
        Args:
            mesh (Mesh): A mesh object in pyLOM format.
            device (Optional[Union[str, torch.device]]): The device to use for the graph.
        Returns:
            Graph: A Graph object with the computed node attributes and edge index/attributes.
        """
        node_attrs_dict = cls._compute_node_attrs(mesh)  # Get the node attributes
        edge_index, edge_attrs_dict = cls._compute_edge_index_and_attrs(mesh)  # Get the edge attributes

        graph = cls(
            edge_index=edge_index,
            node_attrs = node_attrs_dict,
            edge_attrs=edge_attrs_dict,
            device=device,
            # **custom_attrs # Not yet implemented.
            )

        return graph

    @cr('Graph.save')
    def save(self,fname,**kwargs) -> None:
        '''
        Store the graph in a h5 file, pyLOM style.
        '''
        # Set default parameters
        if not 'mode' in kwargs.keys():        kwargs['mode']        = 'w' if not os.path.exists(fname) else 'a'
        # Append or save
        edge_index = self.edge_index.cpu().numpy()
        node_save_dict = self._to_pyLOM_format(self.node_attrs_dict)
        edge_save_dict = self._to_pyLOM_format(self.edge_attrs_dict)
        if not kwargs.pop('append',False):
            io.h5_save_graph_serial(fname,edge_index,node_save_dict,edge_save_dict,**kwargs)
        else:
            io.h5_append_graph_serial(fname,edge_index,node_save_dict,edge_save_dict,**kwargs)

    @classmethod
    def load(
        cls,
        fname: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Graph":
        r'''
        Load a graph from a h5 file, pyLOM style.
        '''
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Graph file {fname} not found.")
        if not fname.endswith('.h5'):
            raise ValueError(f"Graph file {fname} must be a .h5 file.")

        edge_index,node_attrs_dict,edge_attrs_dict = io.h5_load_graph_serial(fname)

        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        node_attrs = {key: torch.tensor(value['value'], dtype=torch.float32) for key, value in node_attrs_dict.items()}
        edge_attrs = {key: torch.tensor(value['value'], dtype=torch.float32) for key, value in edge_attrs_dict.items()}  

        return cls(edge_index=edge_index,
                   node_attrs=node_attrs,
                   edge_attrs=edge_attrs,
                   device=device,
                   )

    def save_pt(self, fname: str) -> None:
        r'''
        Save the graph to a PyTorch .pt file.
        Args:
            fname (str): The filename to save the graph to.
        '''
        if not fname.endswith('.pt'):
            raise ValueError(f"Graph file {fname} must be a .pt file.")
        outdir = os.path.dirname(fname)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        torch.save(self, fname)

    @classmethod
    def load_pt(cls, fname: str, device: Union[str, torch.device]) -> "Graph":
        r'''
        Load a graph from a PyTorch .pt file.
        Args:
            fname (str): The filename to load the graph from.
        Returns:
            Graph: The loaded graph object.
        '''
        if not fname.endswith('.pt'):
            raise ValueError(f"Graph file {fname} must be a .pt file.")
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Graph file {fname} not found.")
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please use CPU instead.")
        return torch.load(fname, map_location=str(device))


    @staticmethod
    def _to_pyLOM_format(attrs_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Union[int, np.ndarray]]]:
        r'''
        Converts a dictionary of numpy arrays to a dictionary of dictionaries with the shape and value keys.
        This is used to save the node and edge attributes in pyLOM format.
        '''
        save_dict = {}
        for key, value in attrs_dict.items():
            save_dict[key] = {
                'dim': value.shape[1] if len(value.shape) > 1 else 1,
                'value': value.cpu().numpy()
            }
        return save_dict

    @staticmethod
    def _compute_node_attrs(mesh: Mesh) -> Dict[str, torch.Tensor]:
        r'''Computes the node attributes of Graph as described in
            Hines, D., & Bekemeyer, P. (2023). Graph neural networks for the prediction of aircraft surface pressure distributions.
            Aerospace Science and Technology, 137, 108268.
            https://doi.org/10.1016/j.ast.2023.108268

        Args:
            mesh (Mesh): A RANS mesh in pyLOM format.
        Returns:
            Dict[str, torch.Tensor]: Node attributes of the graph.
        '''
        # Get the cell centers
        xyzc = mesh.xyzc
        # Get the surface normals
        surface_normals = mesh.normal
        
        node_attrs_dict = {'xyz': torch.tensor(xyzc, dtype=torch.float32), 'normals': torch.tensor(surface_normals, dtype=torch.float32)}

        return node_attrs_dict


    @staticmethod
    def _compute_edge_index_and_attrs(mesh: Mesh) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r'''Computes the edge index and attributes of Graph as described in
            Hines, D., & Bekemeyer, P. (2023). Graph neural networks for the prediction of aircraft surface pressure distributions.
            Aerospace Science and Technology, 137, 108268.
            https://doi.org/10.1016/j.ast.2023.108268
        Args:
            mesh (Mesh): A RANS mesh in pyLOM format.
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Edge index and attributes of the graph.
        '''
        # Check whether the cells are 2D
        if not np.all(np.isin(mesh.eltype, [2, 3, 4, 5])):
            raise ValueError("The mesh must contain only 2D cells in order to compute the wall normals.")
        

        # Dictionary that maps each edge to the cells that share it
        edge_dict = edge_to_cells(mesh.connectivity)
        # List storing directed edges in the dual graph
        edge_list = []
        # List to store the wall normals.
        wall_normals_list = []

        # Iterate over each cell
        for i, cell_id in enumerate(range(mesh.ncells)):
            cell_normal = mesh.normal[cell_id]
            cell_nodes = mesh.connectivity[cell_id]
            nodes_xyz = mesh.xyz[cell_nodes]  # Get the nodes of the cell

            cell_edges, cell_wall_normals = wall_normals(cell_nodes, nodes_xyz, cell_normal)  # Compute the edge normals of the cell
            
            # Directed dual edges: tuples of the form (cell_id, neighbor_id)
            dual_edges = [
                (cell_id, (edge_dict[edge] - {cell_id}).pop()) if len(edge_dict[edge]) == 2 else None # If the edge is not a boundary edge, get the neighbor cell
                for edge in cell_edges
            ]

            edge_list.extend(dual_edges)
            wall_normals_list.extend(cell_wall_normals)

            if i%1e5 == 0:
                print(f"Processing mesh. {i} cells out of {mesh.ncells} processed.")

        # Remove the wall normals and dual edges at the boundary walls
        edge_list, wall_normals_list = zip(*[
                (x, y) for x, y in zip(edge_list, wall_normals_list) if x is not None
            ])

        edge_index_np = np.array(edge_list, dtype=np.int64).T  # Convert to numpy array and transpose
        wall_normals_tensor = torch.tensor(wall_normals_list, dtype=torch.float32)  # Convert to torch tensor

        # Compute the rest of the edge_attributes
        # Get the cell centers
        xyzc = mesh.xyzc
        # Get the edge coordinates
        c_i = xyzc[edge_index_np[0, :]]
        c_j = xyzc[edge_index_np[1, :]]
        d_ij = c_j - c_i
        # Transform to spherical coordinates
        r = np.linalg.norm(d_ij, axis=1)
        theta = np.arccos(d_ij[:, 2] / r)
        phi = np.arctan2(d_ij[:, 1], d_ij[:, 0])

        r = torch.from_numpy(r).float()
        theta = torch.from_numpy(theta).float()
        phi = torch.from_numpy(phi).float()
        
        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)
        edge_attrs_dict = {'edges_spherical': torch.stack((r, theta, phi), dim=1),
                           'wall_normals': wall_normals_tensor}

        return edge_index, edge_attrs_dict

    # def filter(self,
    #     node_mask: Optional[Union[list, torch.Tensor, np.ndarray]]=None,
    #     node_indices: Optional[Union[list, torch.Tensor, np.ndarray]]=None
    # ):
    #     r'''
    #     Filter graph by providing either a boolean mask or a list of node indices to keep.

    #     Args:
    #         node_mask: Boolean mask to filter nodes.
    #         node_indices: List of node indices to keep.
    #     '''
        

    #     if node_mask is None and node_indices is None:
    #         raise ValueError("Either node_mask or node_indices must be provided.")
    #     elif node_mask is not None and node_indices is not None:
    #         raise ValueError("Only one of node_mask or node_indices must be provided.")
    #     elif node_mask is not None:
    #         node_mask = torch.tensor(node_mask, dtype=torch.bool)
    #     elif node_indices is not None:
    #         node_mask = torch.zeros(self.x.shape[0], dtype=torch.bool)
    #         node_mask[node_indices] = True

    #     for attr in self.node_attrs():
    #         if getattr(self, attr) is not None:
    #             setattr(self, attr, getattr(self, attr)[node_mask])

    #     self.edge_attr = self.edge_attr[torch.logical_and(node_mask[self.edge_index[0]], node_mask[self.edge_index[1]])]
    #     self.edge_index = self.edge_index[:, torch.logical_and(node_mask[self.edge_index[0]], node_mask[self.edge_index[1]])]
    #     self.edge_index -= torch.min(self.edge_index)