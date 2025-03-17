#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# GPR Module
#
# Last rev: 19/02/2025

import numpy as np, GPy

# Multi Fidelity GPR Model
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.kernels                import LinearMultiFidelityKernel
from emukit.multi_fidelity.models                 import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers     import GPyMultiOutputWrapper

from ..utils  import cr_nvtx as cr, raiseError, pprint


# KernelSelector: instantiates kernels with bounds and provides information
class KernelSelector:
    """
    A kernel selector with a limited list of available kernels.
    Allows retrieving the parameters of each kernel without needing to instantiate it.
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.available_kernels = {
            "RBF": GPy.kern.RBF,
            "Matern32": GPy.kern.Matern32,
            "Matern52": GPy.kern.Matern52,
            "Exponential": GPy.kern.Exponential,
            "Linear": GPy.kern.Linear,
            "RatQuad": GPy.kern.RatQuad,
            "Poly": GPy.kern.Poly,
            "PeriodicExponential": GPy.kern.PeriodicExponential,
            "StdPeriodic": GPy.kern.StdPeriodic,
            "White": GPy.kern.White,
        }
        # Dynamically add functions to create each kernel.
        for name in self.available_kernels:
            setattr(self, name, self._make_kernel_func(name))

    def _make_kernel_func(self, kernel_name: str):
        """
        Creates a function that generates a kernel with the specified name.
        Allows passing parameters and bounds.
        """

        def kernel_func(**kwargs):
            limits = kwargs.pop("limits", None)
            kernel_class = self.available_kernels[kernel_name]
            kernel = kernel_class(self.input_dim, **kwargs)
            if limits:
                for param, (lower, upper) in limits.items():
                    if hasattr(kernel, param):
                        kernel[param].constrain_bounded(lower, upper)
                    else:
                        raiseError(
                            f"The kernel '{kernel_name}' does not have the parameter '{param}'."
                        )
            return kernel

        return kernel_func

    def get_kernel_parameters(self, kernel_name: str):
        """
        Returns the parameter names of the kernel and indicates if it supports ARD.
        """
        kernel_class = self.available_kernels.get(kernel_name)
        if kernel_class is None:
            raiseError(f"Kernel class '{kernel_name}' not found.")
        try:
            kernel_instance = kernel_class(self.input_dim)
        except TypeError:
            raiseError(
                f"Could not instantiate '{kernel_name}' with input_dim={self.input_dim}."
            )
        param_names = list(kernel_instance.parameter_names())
        has_ard = hasattr(kernel_instance, "ARD") or (
            "lengthscale" in param_names and kernel_instance.lengthscale.size > 1
        )
        return {"parameters": param_names, "ARD": has_ard}

    def __dir__(self):
        """
        Allows dir(obj) to list the available kernels.
        """
        return list(self.available_kernels.keys()) + super().__dir__()


# Base class with common utilities
class GPRBase:
    def __init__(self):
        self.model = None
        self.input_dim = None

    @staticmethod
    def ensure_column_matrix(arr):
        """
        Ensures that the array is in column matrix shape.
        """
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr


# Single Fidelity GPR Model
class SF_GPR(GPRBase):
    """
    Class for Single Fidelity Gaussian Process Regression (GPR) using GPy.

    The interface is similar to sklearn:
      - fit(X_train, y_train, kernel, noise_var, num_restarts, verbose)
      - predict(X_test) returns a dictionary with 'mean' and 'std'

    It ensures that the data are in column matrix shape.
    """

    def __init__(self, input_dim=None):
        """
        If input_dim is provided, a KernelSelector is created immediately.
        """
        super().__init__()
        self._kernel_selector = None
        self._kernel = None
        if input_dim is not None:
            self.input_dim = input_dim
            self._kernel_selector = KernelSelector(input_dim)

    @property
    def kernel(self):
        """
        Allows access to the KernelSelector to create kernels (e.g., model.kernel.RBF(...)).
        """
        if self._kernel_selector is None:
            raiseError(
                "Provide input_dim in the constructor before accessing the kernel."
            )
        return self._kernel_selector

    @cr('SF_GPR.fit')
    def fit(
        self, X_train, y_train, kernel, noise_var=None, num_restarts=5, verbose=True
    ):
        """
        Fits the GPR model:
          - X_train, y_train: training data (converted to column matrix).
          - kernel: expects a kernel already created.
          - noise_var: if provided (not None), the Gaussian noise variance is fixed;
                       otherwise, it is left free for optimization.
          - num_restarts and verbose: parameters for optimization.
        """
        self.train_features = self.ensure_column_matrix(X_train)
        self.train_labels = self.ensure_column_matrix(y_train)
        # If input_dim was not provided in the constructor, infer it from the data
        if self.input_dim is None:
            self.input_dim = self.train_features.shape[1]
        if self._kernel_selector is None:
            self._kernel_selector = KernelSelector(self.input_dim)
        self._kernel = kernel
        self.model = GPy.models.GPRegression(
            self.train_features, self.train_labels, self._kernel
        )
        if noise_var is not None:
            self.model.Gaussian_noise.fix(noise_var)
        self.model.optimize_restarts(num_restarts, verbose=verbose)
        return self

    def predict(self, X_test):
        """
        Makes predictions with the model using the test data X_test.
        Ensures that X_test is in column matrix shape and returns a dictionary with 'mean' and 'std'.
        """
        if self.model is None:
            raiseError("Fit the model before predicting.")
        X_test = self.ensure_column_matrix(X_test)
        mean, var = self.model.predict(X_test)
        return {"mean": mean, "std": np.sqrt(var)}

    def display_model(self):
        """
        Displays the model summary.
        """
        pprint(0,self.model)


class MF_GPR(GPRBase):
    """
    Model for multi-fidelity Gaussian Process Regression.

    Interface similar to sklearn:
      - fit(train_features_list, train_labels_list, kernels, noise_vars, num_restarts, verbose)
      - predict(predict_features_list) returns a dictionary with predictions for each fidelity.

    It ensures that each array is in column matrix shape.
    """

    def __init__(self, input_dim=None):
        super().__init__()
        self.train_features_list = None
        self.train_labels_list = None
        self._kernel_selector = None
        if input_dim is not None:
            self.input_dim = input_dim
            self._kernel_selector = KernelSelector(input_dim)
        self.n_fidelities = None
        self.kernel_MF = None
        self.wrapper = None

    @property
    def kernel(self):
        """
        Allows access to the KernelSelector to create kernels for each fidelity.
        """
        if self._kernel_selector is None:
            raiseError(
                "Provide input_dim in the constructor before accessing the kernel."
            )
        return self._kernel_selector

    @cr('MF_GPR.fit')
    def fit(
        self,
        train_features_list,
        train_labels_list,
        kernels,
        noise_vars=None,
        num_restarts=5,
        verbose=True,
    ):
        """
        Fits the multi-fidelity model:
          - train_features_list, train_labels_list: lists of training arrays (one per fidelity)
          - kernels: list of kernels (one per fidelity), which are combined internally
          - noise_vars: float or list of floats to fix the noise; if None, noise is free for optimization.
          - num_restarts and verbose: parameters for optimization.
        """
        self.train_features_list = [
            self.ensure_column_matrix(x) for x in train_features_list
        ]
        self.train_labels_list = [
            self.ensure_column_matrix(y) for y in train_labels_list
        ]
        self.input_dim = self.train_features_list[0].shape[1]
        self.n_fidelities = len(self.train_features_list)
        self.X_MF_upper, self.Y_MF_upper = convert_xy_lists_to_arrays(
            self.train_features_list, self.train_labels_list
        )
        if self._kernel_selector is None:
            self._kernel_selector = KernelSelector(self.input_dim)
        if not isinstance(kernels, list) or len(kernels) != self.n_fidelities:
            raiseError(
                "Provide a list of kernels equal in length to the number of fidelities."
            )
        self.kernel_MF = LinearMultiFidelityKernel(kernels)
        self.model = GPyLinearMultiFidelityModel(
            self.X_MF_upper,
            self.Y_MF_upper,
            self.kernel_MF,
            n_fidelities=self.n_fidelities,
        )
        self.wrapper = GPyMultiOutputWrapper(
            self.model,
            self.n_fidelities,
            n_optimization_restarts=num_restarts,
            verbose_optimization=verbose,
        )
        if noise_vars is not None:
            self._fix_noise(noise_vars)
        self.wrapper.optimize()
        return self

    def _fix_noise(self, noise_vars):
        """
        Internal method to fix the Gaussian noise for each fidelity.
        noise_vars: float or list of floats.
        """
        noise_params = [
            p
            for p in self.wrapper.gpy_model.mixed_noise.parameters
            if "Gaussian_noise" in p.name
        ]
        if isinstance(noise_vars, (int, float)):
            noise_vars = [noise_vars] * len(noise_params)
        if len(noise_vars) != len(noise_params):
            raiseError(
                f"Expected {len(noise_params)} noise variances, but got {len(noise_vars)}."
            )
        for param, noise_var in zip(noise_params, noise_vars):
            param.fix(noise_var)

    def predict(self, predict_features_list):
        """
        Performs prediction for each fidelity using the provided test data.
        Expects a list of arrays (one per fidelity); ensures that each array is in column matrix shape.
        Returns a dictionary with predictions for each fidelity (keys 'fidelity_1', etc.).
        """
        if not isinstance(predict_features_list, list):
            raiseError(
                "predict_features_list must be a list of arrays (one per fidelity)."
            )
        predict_features_list = [
            self.ensure_column_matrix(x) for x in predict_features_list
        ]
        x_mf_predict = convert_x_list_to_array(predict_features_list)
        results = {}
        start = 0
        for i, x in enumerate(predict_features_list):
            n_points = x.shape[0]
            Xi = x_mf_predict[start : start + n_points]
            mean, var = self.wrapper.predict(Xi)
            results[f"fidelity_{i+1}"] = {"mean": mean, "std": np.sqrt(var)}
            start += n_points
        return results

    def display_model(self):
        """
        Displays the multi-fidelity model summary.
        """
        pprint(0,self.wrapper.gpy_model)

