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
from typing import List, Union

# Third-party libraries
import numpy as np
import torch

# Local modules
from ...utils.errors import raiseError






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