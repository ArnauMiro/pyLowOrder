#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# GPR Module
#
# Last rev: 24/04/2025

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
    A selector for GPy kernel classes that allows dynamic instantiation
    with optional bounds and inspection of kernel parameters.

    Attributes:
        input_dim (int): The dimensionality of the input space.
        available_kernels (Dict[str, Type[GPy.kern.Kern]]): Mapping
            of kernel names to their GPy classes.
    """

    def __init__(self, input_dim: int):
        """
        Initializes the KernelSelector.

        Args:
            input_dim (int): Dimensionality of the input features for kernels.
        """
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
        Produces a kernel function that accepts constructor parameters and optional bounds.

        Args:
            kernel_name (str): Key matching one of available_kernels.

        Returns:
            Callable[..., GPy.kern.Kern]: A function that creates the kernel.
        """

        def kernel_func(**kwargs):
            """
            Instantiates the kernel with given parameters and applies bounds.

            Keyword Args:
                Any valid constructor args for the GPy kernel class.
                limits (Dict[str, Tuple[float, float]], optional):
                    Bounds for kernel parameters, mapping parameter name to (min, max).

            Returns:
                GPy.kern.Kern: Configured kernel instance.

            Raises:
                RuntimeError: If specified limit refers to non-existent parameter.
            """
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
        Retrieves the list of parameter names for a kernel and whether it supports ARD.

        Args:
            kernel_name (str): Name of the kernel to inspect.

        Returns:
            Dict[str, Any]: {
                'parameters': List[str],  # Names of trainable parameters
                'ARD': bool              # True if kernel uses Automatic Relevance Determination
            }

        Raises:
            RuntimeError: If kernel_name is invalid or instantiation fails.
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
        Extends dir() to list available kernel names as attributes.

        Returns:
            List[str]: List of attribute names including kernel factory methods.
        """
        return list(self.available_kernels.keys()) + super().__dir__()


# Base class with common utilities
class GPRBase:
    """
    Base class providing common utilities for Gaussian Process Regression models.

    Attributes:
        model (Optional[Any]): Underlying GPy or Emukit model, once fitted.
        input_dim (Optional[int]): Feature dimensionality.
    """
    def __init__(self):
        self.model = None
        self.input_dim = None

    @staticmethod
    def ensure_column_matrix(arr):
        """
        Converts a 1D or row-oriented array into a column matrix.

        Args:
            arr (array-like): Input data array.

        Returns:
            numpy.ndarray: Array reshaped to (n_samples, n_features) format.
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
    Single-fidelity Gaussian process regression model.

    Interface:
      - fit(X, y, kernel, noise_var, num_restarts, verbose)
      - predict(X) → {'mean': array, 'std': array}

    Automatically reshapes input arrays to column matrices for consistency.
    """

    def __init__(self, input_dim=None):
        """
        Initializes the SF_GPR model.

        Args:
            input_dim (int, optional): Dimensionality of input features.
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
        Access the KernelSelector to build kernels before fitting.

        Returns:
            KernelSelector: Factory for kernel instantiation.

        Raises:
            RuntimeError: If input_dim was not provided at construction.
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
        Fits the GP regression model to training data.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training targets.
            kernel (GPy.kern.Kern): Pre-configured kernel instance.
            noise_var (float, optional): Fixed Gaussian noise variance.
            num_restarts (int): Number of restarts for optimizer.
            verbose (bool): Whether to print optimization progress.

        Returns:
            SF_GPR: The fitted model instance.
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
        Predicts mean and standard deviation on new data.

        Args:
            X_test (array-like): Test feature set.

        Returns:
            Dict[str, numpy.ndarray]: {
                'mean': Predicted means,
                'std': Predicted standard deviations
            }

        Raises:
            RuntimeError: If called before fitting.
        """
        if self.model is None:
            raiseError("Fit the model before predicting.")
        X_test = self.ensure_column_matrix(X_test)
        mean, var = self.model.predict(X_test)
        return {"mean": mean, "std": np.sqrt(var)}

    def display_model(self):
        """
        Prints the summary of the underlying GPy model.
        """
        pprint(0,self.model)


class MF_GPR(GPRBase):
    """
    Multi-fidelity Gaussian process regression model.

    Interface:
      - fit(feature_list, label_list, kernels, noise_vars, num_restarts, verbose)
      - predict(feature_list) → {
            'fidelity_1': {'mean': array, 'std': array},
            'fidelity_2': {'mean': array, 'std': array},
            ...
        }

    Automatically reshapes input arrays to column matrices for consistency.
    """

    def __init__(self, input_dim=None):
        """
        Initializes the multi-fidelity GPR model.

        Args:
            input_dim (int, optional): Dimensionality of each fidelity feature space.
        """
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
        Access the KernelSelector to build kernels for each fidelity.

        Returns:
            KernelSelector: Factory for multi-fidelity kernels.

        Raises:
            RuntimeError: If input_dim was not provided at construction.
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
        Fits the linear multi-fidelity model to lists of data arrays.

        Args:
            train_features_list (List[array-like]): Features per fidelity.
            train_labels_list (List[array-like]): Targets per fidelity.
            kernels (List[GPy.kern.Kern]): One kernel per fidelity.
            noise_vars (float or List[float], optional): Noise variances.
            num_restarts (int): Number of optimization restarts.
            verbose (bool): Print optimization logs.

        Returns:
            MF_GPR: The fitted multi-fidelity model.

        Raises:
            RuntimeError: If kernels list length mismatches number of fidelities.
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
        Fixes Gaussian noise parameters in the wrapped GPy model.

        Args:
            noise_vars (float or List[float]): Desired noise variances.

        Raises:
            RuntimeError: If length of noise_vars mismatches number of noise parameters.
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
        Predicts outputs at each fidelity level for new data.

        Args:
            predict_features_list (List[array-like]): Test arrays per fidelity.

        Returns:
            Dict[str, Dict[str, numpy.ndarray]]: Mapping
            fidelity_i → {'mean': ..., 'std': ...}

        Raises:
            RuntimeError: If input is not a list of arrays.
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
        Prints the summary of the underlying multi-fidelity GPy model.
        """
        pprint(0,self.wrapper.gpy_model)

