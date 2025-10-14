import numpy as np

from typing import Optional, Dict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    precision_recall_curve, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss
)

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
    

class ClassificationEvaluator:
    r"""
    Evaluator class for binary classification tasks (labels in {0, 1}).
    Works with probabilities or hard labels.
    Learns an operating threshold from validation by maximizing a chosen metric
    ("f1", "accuracy", "balanced_accuracy", "youden").
    Reports common classification metrics, including confusion-matrix entries.

    Args:
        threshold_metric (str): Metric to maximize when selecting threshold on validation (default: ``f1``).
        pos_label (int): Positive class label (default: ``1``).
    """

    def __init__(
        self,
        threshold_metric: str = "f1",
        pos_label: int = 1,
    ) -> None:
        self.threshold_metric = threshold_metric
        self.pos_label = pos_label
        self.threshold_: float = 0.5
        self._metrics: Optional[Dict[str, float]] = None

    @property
    def threshold_metric(self) -> str:
        return self._threshold_metric

    @threshold_metric.setter
    def threshold_metric(self, value: str) -> None:
        allowed = {"f1", "accuracy", "balanced_accuracy", "youden"}
        if value not in allowed:
            raiseError(f"threshold_metric must be one of {allowed}")
        self._threshold_metric = value

    @property
    def pos_label(self) -> int:
        return self._pos_label

    @pos_label.setter
    def pos_label(self, value: int) -> None:
        if value not in (0, 1):
            raiseError("pos_label must be 0 or 1 for binary classification.")
        self._pos_label = int(value)

    def _choose_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> float:
        """
        Pick threshold that maximizes the configured metric on validation.
        Uses precision_recall_curve thresholds (plus a couple of extras).
        """
        # Ensure 1D probabilities
        y_prob = self._to_1d(y_prob)
        y_true = self._to_1d(y_true).astype(int)

        # precision_recall_curve gives thresholds spanning (0,1) where decisions change
        try:
            p, r, th = precision_recall_curve(y_true, y_prob, pos_label=self.pos_label)
            candidates = list(th) + [0.5, 0.0, 1.0]
        except Exception:
            # If something goes wrong, fall back to a small grid
            candidates = [0.0, 0.25, 0.5, 0.75, 1.0]

        best_val = -np.inf
        best_th = 0.5

        for t in candidates:
            y_pred = (y_prob >= t).astype(int)
            if self.threshold_metric == "f1":
                val = f1_score(y_true, y_pred, pos_label=self.pos_label)
            elif self.threshold_metric == "accuracy":
                val = accuracy_score(y_true, y_pred)
            elif self.threshold_metric == "balanced_accuracy":
                val = balanced_accuracy_score(y_true, y_pred)
            elif self.threshold_metric == "youden":
                tn, fp, fn, tp = self._confmat_counts(y_true, y_pred)
                # sensitivity (TPR) + specificity (TNR) - 1
                sens = tp / max(1, tp + fn)
                spec = tn / max(1, tn + fp)
                val = sens + spec - 1.0
            else:
                val = -np.inf

            if val > best_val:
                best_val, best_th = val, float(t)

        return best_th

    @staticmethod
    def _confmat_counts(y_true: np.ndarray, y_pred: np.ndarray):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return tn, fp, fn, tp

    @staticmethod
    def _to_1d(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 2:
            if a.shape[1] == 2:
                # assume [:, 1] is P(y=1)
                a = a[:, 1]
            elif a.shape[1] == 1:
                a = a[:, 0]
        return a.reshape(-1)

    def print_metrics(self) -> None:
        if self._metrics is None:
            raiseError("No metrics have been calculated yet.")
        pprint(0, "\nClassification evaluator metrics:")
        ordered = [
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "roc_auc",
            "pr_auc",
            "mcc",
            "log_loss",
            "brier",
            "tp", "fp", "tn", "fn",
        ]
        for k in ordered:
            if k in self._metrics:
                v = self._metrics[k]
                if isinstance(v, float):
                    if np.isnan(v):
                        pprint(0, f"{k}: NaN")
                    elif v < 1e-3 or v > 1e3:
                        pprint(0, f"{k}: {v:.4e}")
                    else:
                        pprint(0, f"{k}: {v:.4f}")
                else:
                    pprint(0, f"{k}: {v}")

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        is_probability: Optional[bool] = None,
        pick_threshold: bool = True,
        given_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Evaluate classification metrics.

        Args:
            y_true: Array of true labels (0/1).
            y_pred: Either probabilities P(y=1|x) or hard labels {0,1}.
                    Shapes accepted: (N,), (N,1), or (N,2) with column 1 = P(y=1).
            is_probability: If None, auto-detect (non-binary -> probs).
            pick_threshold: If True and input are probabilities, choose threshold on-the-fly
                            by maximizing `threshold_metric`.
            given_threshold: If provided, use this threshold (overrides learned one).

        Returns:
            dict with metrics and chosen threshold.
        """
        # Ensure arrays
        try:
            y_true = self._to_1d(np.array(y_true)).astype(int)
            y_in = np.array(y_pred)
        except Exception:
            raiseError(f"could not create numpy arrays from types {type(y_true)} / {type(y_pred)}")

        # Detect if y_pred are probabilities or labels
        if is_probability is None:
            # If values are only {0,1}, treat as labels; otherwise probabilities
            uniques = np.unique(y_in)
            is_probability = not np.array_equal(uniques, np.array([0, 1])) and not np.array_equal(uniques, np.array([0])) and not np.array_equal(uniques, np.array([1]))

        # Prepare labels and probs
        if is_probability:
            y_prob = self._to_1d(y_in)
            # Choose threshold if requested
            thr = float(self.threshold_)
            if given_threshold is not None:
                thr = float(given_threshold)
            elif pick_threshold:
                thr = self._choose_threshold(y_true, y_prob)
            self.threshold_ = thr
            y_hat = (y_prob >= thr).astype(int)
        else:
            y_hat = self._to_1d(y_in).astype(int)
            # If labels provided and a threshold was passed, ignore it (labels already hard)
            y_prob = None

        # Confusion counts
        tn, fp, fn, tp = self._confmat_counts(y_true, y_hat)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_hat)
        bal_acc = (tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2.0  # robust balanced accuracy
        precision = precision_score(y_true, y_hat, zero_division=0, pos_label=self.pos_label)
        recall = recall_score(y_true, y_hat, zero_division=0, pos_label=self.pos_label)
        specificity = tn / max(1, tn + fp)
        f1 = f1_score(y_true, y_hat, zero_division=0, pos_label=self.pos_label)
        mcc = matthews_corrcoef(y_true, y_hat) if (tp+tn+fp+fn) > 0 else np.nan

        # Prob-based metrics (may be NaN if not computable)
        if y_prob is not None:
            # Handle degenerate cases where AUC/AP are undefined
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
            except Exception:
                roc_auc = np.nan
            try:
                pr_auc = average_precision_score(y_true, y_prob)
            except Exception:
                pr_auc = np.nan
            try:
                ce = log_loss(y_true, y_prob, labels=[0, 1])
            except Exception:
                ce = np.nan
            try:
                brier = brier_score_loss(y_true, y_prob, pos_label=self.pos_label)
            except Exception:
                brier = np.nan
            threshold_used = self.threshold_
        else:
            roc_auc = pr_auc = ce = brier = np.nan
            threshold_used = self.threshold_

        self._metrics = {
            "threshold": float(threshold_used),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(bal_acc),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if roc_auc == roc_auc else np.nan,
            "pr_auc": float(pr_auc) if pr_auc == pr_auc else np.nan,
            "mcc": float(mcc) if mcc == mcc else np.nan,
            "log_loss": float(ce) if ce == ce else np.nan,
            "brier": float(brier) if brier == brier else np.nan,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        }
        return self._metrics