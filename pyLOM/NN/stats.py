import numpy as np
from .. import pprint
from ..utils import raiseError

class RegressionEvaluator():
    r"""
    Evaluator class for regression tasks. 
    Includes methods to calculate the mean squared error (MSE), mean absolute error (MAE), 
    mean relative error (MRE), quantiles of the absolute errors, L2 error, and R-squared.

    Args:
        tolerance (float): Tolerance level to consider values close to zero for MRE calculation (default: ``1e-4``).
    """

    def __init__(
        self,
        tolerance: float = 1e-4,
    ) -> None:
        self.tolerance = tolerance

    @property
    def tolerance(self) -> float:
        return self._tolerance
    
    @tolerance.setter
    def tolerance(self, value: float) -> None:
        if value < 0:
            raise raiseError("Tolerance must be a positive value.")
        if value == 0:
            raise raiseError("Tolerance cannot be zero.")
        if value > 1e-2:
            raise raiseError("Tolerance should be less than 1e-2.")
        self._tolerance = value

    def mean_squared_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute the mean squared error (MSE) between the true values and the predicted values.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def mean_absolute_error(self, y_true, y_pred):
        """
        Compute the mean absolute error (MAE) between the true values and the predicted values.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The mean absolute error.
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def mean_relative_error(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """
        Compute the mean relative error (MRE) between the true values and the predicted values,
        adding a tolerance level to consider values close to zero.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.
            tolerance (float): Tolerance level to consider values close to zero. Default is 1e-4.

        Returns:
            float: The mean relative error excluding cases where y_true is close to zero.
        """
        relative_errors = np.abs((y_true - y_pred) / (y_true + self.tolerance))
        return np.mean(relative_errors) * 100
    
    def ae_q(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        quantile: int,
    ) -> float:
        """
        Calculate the quantile of the absolute errors between the true and predicted values.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.
            quantile (int): The quantile to calculate. Must be between 0 and 100.

        Returns:
            float: The quantile of the absolute errors.
        """

        absolute_errors = np.abs(y_true - y_pred)
        return np.percentile(absolute_errors, quantile)    

    def l2_error(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """
        Calculate the L2 error between the true and predicted values.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The L2 error.
        """

        return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
    
    def R2(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Calculate the R-squared (coefficient of determination) for a set of true and predicted values.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The R-squared value.
        """
        y_mean = np.mean(y_true)
        total_sum_of_squares = np.sum((y_true - y_mean) ** 2)
        residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
        r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r_squared

    def print_metrics(self):
        """
        Print the calculated regression metrics.
        """
        if self._metrics is None:
            raise raiseError("No metrics have been calculated yet.")
        
        pprint(0, "\nRegression evaluator metrics:")
        
        for key, value in self._metrics.items():
            if key == "mre":
                pprint(0, f"{key}: {value:.4f}%")
            elif key == "r2":
                pprint(0, f"{key}: {value:.4f}")
            else:
                if value < 1e-3 or value > 1e3:
                    pprint(0, f"{key}: {value:.4e}")
                else:
                    pprint(0, f"{key}: {value:.4f}")


    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """
        Calculate multiple regression metrics between the true and predicted values.

        Args:
            y_true (numpy.ndarray): An array-like object containing the true values.
            y_pred (numpy.ndarray): An array-like object containing the predicted values.

        Returns:
            dict: A dictionary containing the calculated regression metrics.
        """
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
        except AttributeError:
            raise raiseError(f"could not create numpy arrays from object with type {type(y_true)}")
        
        mse = self.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = self.mean_absolute_error(y_true, y_pred)
        mre = self.mean_relative_error(y_true, y_pred)
        aq_95 = self.ae_q(y_true, y_pred, 95)
        aq_99 = self.ae_q(y_true, y_pred, 99)
        r2 = self.R2(y_true, y_pred)
        l2_error = self.l2_error(y_true, y_pred)
        self._metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae, 
            "mre": mre, 
            "ae_95": aq_95,
            "ae_99": aq_99,
            "r2": r2,
            "l2_error": l2_error 
        }
        return self._metrics