#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Multiclass Classifier class.
#
# Last rev: 24/06/2026

import numpy as np, os, pickle, torch, xgboost as xgb

from typing             import Dict, List, Tuple
from torch.utils.data   import DataLoader
from sklearn.metrics    import log_loss
from sklearn.utils      import compute_class_weight

from ..                 import DEVICE, PIN_MEMORY, set_seed
from ..optimizer        import OptunaOptimizer
from ...                import pprint, cr
from ...utils.errors    import raiseWarning, raiseError

try:
    from optuna.exceptions import TrialPruned
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


class MulticlassClassifier:
    r"""
    Gradient-boosted decision trees multiclass classifier (XGBoost).
    The model is based on `xgboost.XGBClassifier <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier>`_ with ``objective="multi:softprob"``.

    Args:
        input_size (int): Number of input features.
        n_classes (int): Number of target classes (must be >= 3).
        learning_rate (float, optional): Boosting learning rate (default: 0.05).
        n_estimators (int, optional): Number of boosting rounds (default: 1000).
        early_stopping_rounds (int, optional): Early stopping rounds (default: 100).
        max_depth (int, optional): Maximum tree depth for base learners (default: 6).
        subsample (float, optional): Subsample ratio of the training instances (default: 0.9).
        colsample_bytree (float, optional): Subsample ratio of columns when constructing each tree (default: 0.9).
        reg_lambda (float, optional): L2 regularization term on weights (default: 1.0).
        min_child_weight (float, optional): Minimum sum of instance weight needed in a child (default: 1.0).
        tree_method (str, optional): Tree construction algorithm (default: "hist"). Use "gpu_hist" if GPU build of XGBoost is available.
        enable_categorical (bool, optional): Whether to enable categorical features (default: False). Set True only if dtype is categorical and using hist/gpu_hist.
        class_weight (str or None, optional): Class weighting strategy. Use ``"balanced"`` to weight samples inversely proportional to class frequencies, or ``None`` to disable (default: None).
        seed (int, optional): Random seed for reproducibility (default: 42).
        model_name (str, optional): Name of the model (default: "xgb_multi").
        device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).
        verbose (bool, optional): Whether to print model parameters (default: True).
    """

    def __init__(
        self,
        input_size:             int,
        n_classes:              int,
        *,
        learning_rate:          float = 0.05,
        n_estimators:           int = 1000,
        early_stopping_rounds:  int = 100,
        max_depth:              int = 6,
        subsample:              float = 0.9,
        colsample_bytree:       float = 0.9,
        reg_lambda:             float = 1.0,
        min_child_weight:       float = 1.0,
        tree_method:            str = "hist",           # use "gpu_hist" if GPU build available
        enable_categorical:     bool = False,           # set True only if dtype is categorical and using hist/gpu_hist
        class_weight:           str = None,             # "balanced" or None
        seed:                   int = 42,
        model_name:             str = "xgb_multi",
        device:                 torch.device = DEVICE,
        verbose:                bool = True,
        **kwargs:               Dict,
    ):
        if n_classes < 3:
            raiseError("n_classes must be >= 3. For binary classification use BinaryClassifier.")

        if tree_method not in ["auto", "exact", "approx", "hist", "gpu_hist"]:
            raiseError(f"Invalid tree_method: {tree_method}. Must be one of ['auto', 'exact', 'approx', 'hist', 'gpu_hist'].")

        if class_weight not in [None, "balanced"]:
            raiseError(f"Invalid class_weight: {class_weight}. Must be None or 'balanced'.")

        self.input_size             = input_size
        self.n_classes              = n_classes
        self.output_size            = n_classes
        self.learning_rate          = learning_rate
        self.n_estimators           = n_estimators
        self.early_stopping_rounds  = early_stopping_rounds
        self.max_depth              = max_depth
        self.subsample              = subsample
        self.colsample_bytree       = colsample_bytree
        self.reg_lambda             = reg_lambda
        self.min_child_weight       = min_child_weight
        self.tree_method            = tree_method
        self.enable_categorical     = enable_categorical
        self.class_weight           = class_weight
        self.seed                   = seed
        self.model_name             = model_name
        self.mname                  = f"{self._model_name}_{self.n_estimators:05d}"
        self.device                 = device

        if seed is not None:
            set_seed(seed)
            self.random_state = seed

        self.model = None
        self.checkpoint = None

        if verbose:
            pprint(0, f"Creating model: {self._model_name}")
            keys_print = [
                "input_size", "n_classes", "output_size", "learning_rate",
                "n_estimators", "max_depth", "subsample", "colsample_bytree",
                "reg_lambda", "min_child_weight", "tree_method",
                "enable_categorical", "class_weight", "early_stopping_rounds",
                "random_state", "model_name",
            ]
            for k in keys_print:
                pprint(0, f"\t{k}: {getattr(self, k)}")
            pprint(0, "\ttotal_size (trainable parameters): [tree-based, N/A]\n")

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        if not isinstance(value, str):
            raiseError("model_name must be a string")
        value = value.strip()
        if not value:
            raiseError("model_name cannot be empty")
        self._model_name = value

    @staticmethod
    def _dataset_to_numpy(dataset: torch.utils.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Load the whole dataset into NumPy arrays (X, y)."""
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
        X = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).reshape(-1).astype(np.int64)
        return X, y

    @staticmethod
    def _compute_sample_weights(y: np.ndarray, strategy: str = "balanced") -> np.ndarray:
        classes = np.unique(y)
        weights = compute_class_weight(class_weight=strategy, classes=classes, y=y)
        weight_map = dict(zip(classes, weights))
        return np.array([weight_map[label] for label in y], dtype=np.float32)

    def _count_xgb_leaf_values(
        self,
        include_intercept: bool = False,
        only_used_trees: bool = True,
    ) -> int:
        booster = self.model.get_booster()
        df = booster.trees_to_dataframe()

        if only_used_trees:
            best_it = getattr(self.model, "best_iteration", None)
            if best_it is not None:
                df = df[df["Tree"] <= int(best_it) * self.n_classes]

        n_leaves = int((df["Feature"] == "Leaf").sum())
        return n_leaves + (1 if include_intercept else 0)

    @cr('MulticlassClassifier.fit')
    def fit(
        self,
        train_dataset:  torch.utils.data.Dataset,
        eval_dataset:   torch.utils.data.Dataset = None,
        *,
        batch_size:     int = 32,
        save_logs_path: str = None,
        verbose:        bool = True,
        **kwargs,
    ) -> Dict[str, List[float]]:
        r""""
        Fit the MulticlassClassifier model. If eval_dataset is provided, uses it for early stopping.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset to fit the model.
            eval_dataset (torch.utils.data.Dataset, optional): Evaluation dataset for early stopping.
            batch_size (int, optional): Batch size for DataLoader (default: 32).
            save_logs_path (str, optional): Directory to save training logs (default: None).
            verbose (bool, optional): Whether to print training results (default: True).

        Returns:
            Dict[str, List[float]]: Dictionary containing training and validation losses.
                - "train_loss": List of training mlogloss values.
                - "test_loss": List of validation mlogloss values (if eval_dataset provided).
                - "check": List with a single boolean indicating successful training.
        """

        dataloader_params = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": PIN_MEMORY,
        }

        if not hasattr(self, "train_dataloader"):
            for key in dataloader_params.keys():
                if key in kwargs:
                    dataloader_params[key] = kwargs[key]
            self.train_dataloader = DataLoader(train_dataset, **dataloader_params)

        if not hasattr(self, "eval_dataloader") and eval_dataset is not None:
            for key in dataloader_params.keys():
                if key in kwargs:
                    dataloader_params[key] = kwargs[key]
            self.eval_dataloader = DataLoader(eval_dataset, **dataloader_params)

        X_tr, y_tr = self._dataset_to_numpy(train_dataset)
        if eval_dataset is not None:
            X_va, y_va = self._dataset_to_numpy(eval_dataset)
            eval_set = [(X_va, y_va)]
        else:
            eval_set = None

        sample_weight = (
            self._compute_sample_weights(y_tr, strategy=self.class_weight)
            if self.class_weight is not None
            else None
        )

        self.model = xgb.XGBClassifier(
            objective           = "multi:softprob",
            eval_metric         = "mlogloss",
            num_class           = self.n_classes,
            learning_rate       = self.learning_rate,
            n_estimators        = self.n_estimators,
            max_depth           = self.max_depth,
            subsample           = self.subsample,
            colsample_bytree    = self.colsample_bytree,
            reg_lambda          = self.reg_lambda,
            min_child_weight    = self.min_child_weight,
            tree_method         = self.tree_method,
            enable_categorical  = self.enable_categorical,
            random_state        = self.random_state,
            n_jobs              = 0,
            verbosity           = 0,
        )

        self.model.fit(
            X_tr, y_tr,
            sample_weight=sample_weight,
            eval_set=eval_set if eval_set is not None else None,
            verbose=False,
            early_stopping_rounds=self.early_stopping_rounds if eval_set is not None else None,
        )

        self.n_param_like_ = self._count_xgb_leaf_values(include_intercept=False, only_used_trees=True)

        train_losses = []
        test_losses  = []

        if eval_set is not None and hasattr(self.model, "evals_result"):
            ev = self.model.evals_result()
            val_losses = ev.get("validation_0", {}).get("mlogloss", [])
            test_losses = list(map(float, val_losses))
            if verbose:
                if len(test_losses) > 0:
                    pprint(0, f"\tFinal Val MLLogLoss: {test_losses[-1]:.4e}")
        else:
            prob_tr = self.model.predict_proba(X_tr)
            tr_ll = log_loss(y_tr, prob_tr)
            train_losses.append(tr_ll)
            if verbose:
                pprint(0, f"\tTrain MLLogLoss: {tr_ll:.4e}")

        results = {
            "train_loss": np.array(train_losses, dtype=np.float64),
            "test_loss":  np.array(test_losses,  dtype=np.float64),
            "check": [True],
        }

        if save_logs_path is not None:
            pprint(0, f"\nPrinting losses on path: {save_logs_path}")
            fn = os.path.join(save_logs_path, f"training_results_{self._model_name}.npy")
            np.save(fn, results)

        return results

    @cr('MulticlassClassifier.predict')
    def predict(
        self,
        X: torch.utils.data.Dataset,
        return_targets: bool = False,
        **kwargs,
    ):
        r"""
        Predict class probabilities for the input data.

        Args:
            X (torch.utils.data.Dataset): The dataset whose label class is to be predicted using the input data.
            return_targets (bool, optional): If True, also return the ground-truth labels (default: False).

        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]:
                - ``all_prob``: Array of shape ``(N, n_classes)`` with class probabilities.
                - ``all_targets``: Array of shape ``(N,)`` with integer class labels (only returned when ``return_targets=True``).
        """
        dataloader_params = {
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": PIN_MEMORY,
        }
        for key in list(dataloader_params.keys()):
            if key in kwargs:
                dataloader_params[key] = kwargs[key]

        predict_dataloader = DataLoader(X, **dataloader_params)
        total_rows = len(predict_dataloader.dataset)
        all_prob = np.empty((total_rows, self.n_classes), dtype=np.float32)
        all_targets = np.empty((total_rows,), dtype=np.int64)

        start_idx = 0
        for batch in predict_dataloader:
            x = batch[0].detach().cpu().numpy().astype(np.float32)
            prob = self.model.predict_proba(x)
            bsz = x.shape[0]
            end_idx = start_idx + bsz
            all_prob[start_idx:end_idx, :] = prob
            if return_targets:
                y = batch[1].detach().cpu().numpy().reshape(-1)
                all_targets[start_idx:end_idx] = y
            start_idx = end_idx

        return (all_prob, all_targets) if return_targets else all_prob

    def _define_checkpoint(self):
        return {
            "input_size":            self.input_size,
            "n_classes":             self.n_classes,
            "learning_rate":         self.learning_rate,
            "n_estimators":          self.n_estimators,
            "max_depth":             self.max_depth,
            "subsample":             self.subsample,
            "colsample_bytree":      self.colsample_bytree,
            "reg_lambda":            self.reg_lambda,
            "min_child_weight":      self.min_child_weight,
            "tree_method":           self.tree_method,
            "enable_categorical":    self.enable_categorical,
            "class_weight":          self.class_weight,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed":                  self.seed,
            "device":                self.device,
            "model_name":            self._model_name,
            "xgb_pickle":            pickle.dumps(self.model),
        }

    def save(self, path: str):
        self.checkpoint = self._define_checkpoint()
        if os.path.isdir(path):
            filename = "/" + str(self.mname) + ".pth"
            path = path + filename
        torch.save(self.checkpoint, path)

    @classmethod
    def load(cls, path: str, device: torch.device = DEVICE, verbose: bool = True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raiseWarning("The model has been loaded with weights_only set to False. According with torch documentation, this is not recommended if you do not trust the source of your saved model, as it could lead to arbitrary code execution.")
        checkpoint["device"] = device

        model = cls(
            input_size              = checkpoint["input_size"],
            n_classes               = checkpoint["n_classes"],
            learning_rate           = checkpoint["learning_rate"],
            n_estimators            = checkpoint["n_estimators"],
            max_depth               = checkpoint["max_depth"],
            subsample               = checkpoint["subsample"],
            colsample_bytree        = checkpoint["colsample_bytree"],
            reg_lambda              = checkpoint["reg_lambda"],
            min_child_weight        = checkpoint["min_child_weight"],
            tree_method             = checkpoint["tree_method"],
            enable_categorical      = checkpoint["enable_categorical"],
            class_weight            = checkpoint["class_weight"],
            early_stopping_rounds   = checkpoint["early_stopping_rounds"],
            seed                    = checkpoint["seed"],
            model_name              = checkpoint["model_name"],
            device                  = checkpoint["device"],
            verbose                 = verbose,
        )
        model.model = pickle.loads(checkpoint["xgb_pickle"])
        model.checkpoint = checkpoint
        return model

    @classmethod
    @cr('MulticlassClassifier.create_optimized_model')
    def create_optimized_model(
        cls,
        train_dataset:    torch.utils.data.Dataset,
        eval_dataset:     torch.utils.data.Dataset,
        optuna_optimizer: OptunaOptimizer,
        n_classes:        int,
        **kwargs,
    ) -> Tuple["MulticlassClassifier", Dict]:
        r"""
        Create an optimized model using Optuna. The model is trained on the training dataset and evaluated on the validation dataset.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            optuna_optimizer (OptunaOptimizer): The optimizer to use for optimization.
            n_classes (int): Number of target classes.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[MulticlassClassifier, Dict]: The optimized model and the optimization parameters.
        """
        if not _OPTUNA_AVAILABLE:
            raiseError("Optuna is required for create_optimized_model. Install it with: pip install optuna")
            
        optimization_params = optuna_optimizer.optimization_params
        input_dim = train_dataset[0][0].shape[0]

        def suggest_value(name, space, trial):
            if isinstance(space, (tuple, list)):
                use_log = (space[1] / max(1e-12, space[0])) >= 1000 if isinstance(space[0], (int, float)) else False
                if isinstance(space[0], int):
                    return trial.suggest_int(name, int(space[0]), int(space[1]), log=use_log)
                elif isinstance(space[0], float):
                    return trial.suggest_float(name, float(space[0]), float(space[1]), log=use_log)
            else:
                return space

        def optimization_function(trial):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = None

            try: 
                training_params = {}
                for k, spec in optimization_params.items():
                    training_params[k] = suggest_value(k, spec, trial)
                training_params["save_logs_path"] = None

                model = cls(input_dim, n_classes, verbose=False, **training_params)
                if optuna_optimizer.pruner is not None:
                    n_estimators = training_params["n_estimators"]
                    training_params["n_estimators"] = 1
                    for estimator in range(n_estimators):
                        model.fit(train_dataset, verbose=False, **training_params)
                        y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                        loss_val = log_loss(y_true, y_pred)
                        trial.report(loss_val, estimator)
                        if trial.should_prune():
                            raise TrialPruned()
                else:
                    model.fit(train_dataset, verbose=False, **training_params)
                    y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                    loss_val = log_loss(y_true, y_pred)

                return loss_val

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    pprint(0, f"Trial {trial.number} failed due to out of memory error. Pruning the trial.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise TrialPruned()
                raise

            finally:
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        best_params = optuna_optimizer.optimize(objective_function=optimization_function)

        # Update params with best ones
        OptunaOptimizer.apply_to(optimization_params, optimized_params=best_params)

        return cls(input_dim, n_classes, **optimization_params), optimization_params