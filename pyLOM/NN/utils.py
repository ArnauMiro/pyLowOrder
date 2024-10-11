#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines.
#
# Last rev: 02/10/2024

import os, numpy as np, torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from torch.utils.data import Subset
from torch            import Generator, randperm, default_generator
from functools        import reduce
from itertools        import product, accumulate
from operator         import mul
from typing           import List, Optional, Tuple, cast, Sequence, Union

from .                import DEVICE
from ..utils.cr       import cr
from ..utils.errors   import raiseWarning, raiseError


class MinMaxScaler:
    """
    Min-max scaling to scale variables to a desired range. The formula is given by:
    .. math::
        X_{scaled} = (X - X_{min}) / (X_{max} - X_{min}) * (feature-range_{max} - feature-range_{min}) + feature-range_{min}


    Args:
            feature_range (Tuple): Desired range of transformed data. Default is ``(0, 1)``.
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self._is_fitted = False

    @property
    def is_fitted(self):
        return self._is_fitted
    
    def fit(self, variables: List[Union[np.ndarray, torch.tensor]]):
        """
        Compute the min and max values of each variable.
        Args:
                variables: List of variables to be fitted. The variables should be 2d numpy arrays or torch tensors.
        """
        min_max_values = []
        for variable in variables:
            min_val = variable.min()
            max_val = variable.max()
            min_max_values.append({"min": min_val, "max": max_val})
        self.variable_scaling_params = min_max_values
        self._is_fitted = True

    def transform(
        self, variables: List[Union[np.ndarray, torch.tensor]]
    ) -> List[Union[np.ndarray, torch.tensor]]:
        """
        Scale variables to the range defined on `feature_range` using min-max scaling.
        Args:
                variables: List of variables to be scaled. The variables should be 2d numpy arrays or torch tensors.
        Returns:
                scaled_variables: List of scaled variables.
        """

        scaled_variables = []
        for i, variable in enumerate(variables):
            min_val = self.variable_scaling_params[i]["min"]
            max_val = self.variable_scaling_params[i]["max"]
            scaled_variable = (variable - min_val) / (max_val - min_val)
            # scale the variable to the desired feature_range
            scaled_variable = (
                scaled_variable * (self.feature_range[1] - self.feature_range[0])
                + self.feature_range[0]
            )
            scaled_variables.append(scaled_variable)

        return scaled_variables
    
    def fit_transform(self, variables: List[Union[np.ndarray, torch.tensor]]) -> List[Union[np.ndarray, torch.tensor]]:
        """
        Fit and transform the variables using min-max scaling.
        Args:
                variables: List of variables to be fitted and scaled. The variables should be 2d numpy arrays or torch tensors.
        Returns:
                scaled_variables: List of scaled variables.
        """
        self.fit(variables)
        return self.transform(variables)

    def inverse_transform(self, variables: List[np.ndarray]) -> List[np.ndarray]:
        """
        Inverse scale variables that have been scaled using min-max scaling.
        Args:
                variables: List of variables to be inverse scaled. The variables should be 2d numpy arrays.
                variable_scaling_params: List of dictionaries containing the min and max values of each variable.
        Returns:
                inverse_scaled_variables: List of inverse scaled variables.
        """

        inverse_scaled_variables = []
        for variable, scaling_params in zip(variables, self.variable_scaling_params):
            min_val = scaling_params["min"]
            max_val = scaling_params["max"]
            inverse_scaled_variable = (variable - self.feature_range[0]) / (
                self.feature_range[1] - self.feature_range[0]
            )
            inverse_scaled_variable = inverse_scaled_variable * (max_val - min_val) + min_val
            inverse_scaled_variables.append(inverse_scaled_variable)
        return inverse_scaled_variables


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
            parameters (List[List[Union[float, Tuple]]]): List of parameters to be used as input. All possible combinations of these parameters, i.e. its cartesian product, will appear along with variables_in. If there is only one inner list and its elements are tuples, they will be treated as a single element for the cartesiand produt, which is useful when the combination of the parameters was predefined. Default is ``None``.
            inputs_scaler (MinMaxScaler): Scaler to scale the input variables. Default is ``None``.
            outputs_scaler (MinMaxScaler): Scaler to scale the output variables. Default is ``None``.
            snapshots_by_column (bool): If the snapshots from `variables_out` are stored by column. Default is ``True``.
    """

    def __init__(
        self,
        variables_out: Tuple,
        mesh_shape: Tuple = (1,),
        variables_in: np.ndarray = None,
        parameters: List[List[Union[float, Tuple]]] = None,
        inputs_scaler=None,
        outputs_scaler=None,
        snapshots_by_column=True
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
            self.variables_in = self._process_variables_in(variables_in, parameters)
            if inputs_scaler is not None:
                if not inputs_scaler.is_fitted:
                    inputs_scaler.fit([self.variables_in[:, i] for i in range(self.variables_in.shape[1])])
                self.variables_in = inputs_scaler.transform([self.variables_in[:, i] for i in range(self.variables_in.shape[1])])
                self.variables_in = torch.stack(self.variables_in, dim=1)
        else:
            self.variables_in = None

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

    def _process_variables_in(self, variables_in, parameters):
        if parameters is None:
            return torch.tensor(variables_in, dtype=torch.float32)
        
        variables_in = torch.tensor(variables_in, dtype=torch.float32)
        # parameters is a list of lists of floats. Each contains the values that will be repeated for each input coordinate
        # in some sense, it is like a cartesian product of the parameters with the input coordinates
        if len(parameters) == 1:
            cartesian_product = torch.tensor(parameters[0])
        else:
            cartesian_product = torch.tensor(list(product(*parameters)))

        # repeat the variables_in for each element in the cartesian product
        variables_in_repeated = variables_in.repeat(len(cartesian_product), 1)
        # to repeat the cartesian product for each element in variables_in, we need to repeat each element in the cartesian product for the initial length of variables_in
        # parameters_repeated = []
        # for product_element in cartesian_product:
            # parameters_repeated.append(product_element.repeat(len(variables_in), 1))
        # cartesian_product = torch.cat(parameters_repeated, dim=0)
        cartesian_product = cartesian_product.repeat_interleave(variables_in.size(0), dim=0)
        return torch.cat([variables_in_repeated, cartesian_product], dim=1).float()

    def __len__(self):
        return len(self.variables_out)

    def __getitem__(self, idx):
        if self.variables_in is None:
            return self.variables_out[idx]
        return self.variables_in[idx], self.variables_out[idx]
    
    def __add__(self, other):
        if not isinstance(other, Dataset):
            raiseError(f"Cannot add Dataset with {type(other)}")
        if self.variables_in is None:
            variables_in = None
        else:
            variables_in = torch.cat([self.variables_in, other.variables_in], dim=0)
        variables_out = torch.cat([self.variables_out, other.variables_out], dim=0)
        self.variables_in = variables_in
        self.variables_out = variables_out
        return self

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
        self.variables_out = F.pad(self.variables_out, (0, n0w-nw, 0, n0h-nh, 0, n0d-nd), mode='constant', value=0)
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
        sizes,
        random=True,
        generator: Optional[Generator] = default_generator,
    ):
        """
        Randomly split a dataset into non-overlapping new datasets of given lengths.

        If a list of fractions that sum up to 1 is given,
        the lengths will be computed automatically as
        floor(frac * len(dataset)) for each fraction provided.

        After computing the lengths, if there are any remainders, 1 count will be
        distributed in round-robin fashion to the lengths
        until there are no remainders left.

        Optionally fix the generator for reproducible results.

        Args:
                sizes (sequence): lengths or fractions of splits to be produced
                generator (Generator): Generator used for the random permutation.
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
            raise ValueError(
                "Sum of input lengths does not equal the length of the input dataset!"
            )
        if random:
            indices = randperm(sum(sizes), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
        else:
            indices = list(range(sum(sizes)))
        sizes = cast(Sequence[int], sizes)
        return [
            Subset(self, indices[offset - length : offset])
            for offset, length in zip(accumulate(sizes), sizes)
        ]


def create_results_folder(RESUDIR,echo=True):
    if not os.path.exists(RESUDIR):
        os.makedirs(RESUDIR)
        if echo: print(f"Folder created: {RESUDIR}")
    else:
        if echo: print(f"Folder already exists: {RESUDIR}")


def select_device(device=DEVICE):
    torch.device(device)
    return device


class betaLinearScheduler:
    """Beta schedule, linear growth to max value
    Args:
       start_value (float): initial value of beta
       end_value (float): final value of beta
       warmup (int): number of epochs to reach final value"""

    def __init__(self, start_value, end_value, start_epoch, warmup):
        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.warmup = warmup

    def getBeta(self, epoch):
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