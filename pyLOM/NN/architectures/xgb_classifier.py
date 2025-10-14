import numpy as np, os, pickle, torch, xgboost as xgb

from typing             import Dict, List, Tuple, Callable
from torch.utils.data   import DataLoader
from sklearn.metrics    import log_loss
from ..optimizer        import OptunaOptimizer, TrialPruned
from ..                 import DEVICE, set_seed # pyLOM/NN/__init__.py
from ...                import pprint, cr # pyLOM/__init__.py
from ...utils.errors    import raiseWarning, raiseError


class XGBClassifier:
    r"""
    Gradient-boosted decision trees binary classifier (XGBoost).
    The model is based on `xgboost.XGBClassifier <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier>`_.

    Args:
        input_size (int): Number of input features.
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
        seed (int, optional): Random seed for reproducibility (default: 42).
        model_name (str, optional): Name of the model (default: "xgb").
        device (torch.device, optional): Device to use (default: DEVICE).
        verbose (bool, optional): Whether to print model parameters (default: True).
    """

    def __init__(
        self,
        input_size: int,
        *,
        learning_rate: float = 0.05,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 100,
        max_depth: int = 6,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        tree_method: str = "hist",          # use "gpu_hist" if GPU build available
        enable_categorical: bool = False,   # set True only if dtype is categorical and using hist/gpu_hist
        seed: int = 42,
        model_name: str = "xgb",
        device: torch.device = DEVICE,
        verbose: bool = True,
        **kwargs: Dict,
    ):
        
        self.input_size = input_size
        self.output_size = 1
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.tree_method = tree_method
        self.enable_categorical = enable_categorical
        self.seed = seed
        self.model_name = model_name
        self.mname = f"{self._model_name}_{self.n_estimators:05d}"
        self.device = device

        if seed is not None:
            set_seed(seed)
            self.random_state = seed

        self.model = None
        self.checkpoint = None

        if verbose:
            pprint(0, f"Creating model: {self._model_name}")
            keys_print = [
                "input_size", "output_size", "learning_rate", "n_estimators",
                "max_depth", "subsample", "colsample_bytree", "reg_lambda",
                "min_child_weight", "tree_method", "enable_categorical",
                "early_stopping_rounds", "random_state", "model_name"
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
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, pin_memory=True)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
        X = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).reshape(-1).astype(np.int64)
        return X, y

    @staticmethod
    def _compute_scale_pos_weight(y: np.ndarray) -> float:
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        return float(neg / max(1, pos)) if pos > 0 else 1.0

    @cr('XGBClassifier.fit')
    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset = None,
        *,
        batch_size: int = 32,
        save_logs_path: str = None,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, List[float]]:
        r""""
        Fit the XGBClassifier model. If eval_dataset is provided, uses it for early stopping.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset to fit the model.
            eval_dataset (torch.utils.data.Dataset, optional): Evaluation dataset for early stopping.
            batch_size (int, optional): Batch size for DataLoader (default: 32).
            save_logs_path (str, optional): Directory to save training logs (default: None).
            verbose (bool, optional): Whether to print training results (default: True).

        Returns:
            Dict[str, List[float]]: Dictionary containing training and validation losses.
                - "train_loss": List of training logloss values.
                - "test_loss": List of validation logloss values (if eval_dataset provided).
                - "check": List with a single boolean indicating successful training.
        """

        dataloader_params = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
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

        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=self._compute_scale_pos_weight(y_tr),
            tree_method=self.tree_method,
            enable_categorical=self.enable_categorical,
            random_state=self.random_state,
            n_jobs=0,
            verbosity=0,
        )

        self.model.fit(
            X_tr, y_tr,
            eval_set=eval_set if eval_set is not None else None,
            verbose=False,
            # early_stopping_rounds=self.early_stopping_rounds if eval_set is not None else None,
        )

        train_losses = []
        test_losses = []

        if eval_set is not None and hasattr(self.model, "evals_result"):
            ev = self.model.evals_result()
            val_losses = ev.get("validation_0", {}).get("logloss", [])
            test_losses = list(map(float, val_losses))
            if verbose:
                if len(test_losses) > 0:
                    pprint(0, f"\tFinal Val LogLoss: {test_losses[-1]:.4e}")
        else:
            prob_tr = self.model.predict_proba(X_tr)[:, 1]
            tr_ll = log_loss(y_tr, prob_tr, labels=[0,1])
            train_losses.append(tr_ll)
            if verbose:
                pprint(0, f"\tTrain LogLoss: {tr_ll:.4e}")

        results = {
            "train_loss": np.array(train_losses, dtype=np.float64),
            "test_loss": np.array(test_losses, dtype=np.float64),
            "check": [True],
        }

        if save_logs_path is not None:
            pprint(0, f"\nPrinting losses on path: {save_logs_path}")
            fn = os.path.join(save_logs_path, f"training_results_{self._model_name}.npy")
            np.save(fn, results)

        return results

    @cr('XGBClassifier.predict')
    def predict(
        self,
        X: torch.utils.data.Dataset,
        return_targets: bool = False,
        **kwargs,
    ):
        r"""
        Predict the target values for the inputs data. 
        """
        dataloader_params = {
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": True,
        }
        for key in list(dataloader_params.keys()):
            if key in kwargs:
                dataloader_params[key] = kwargs[key]

        predict_dataloader = DataLoader(X, **dataloader_params)
        total_rows = len(predict_dataloader.dataset)
        all_prob = np.empty((total_rows, self.output_size), dtype=np.float32)
        all_targets = np.empty((total_rows, self.output_size), dtype=np.float32)

        start_idx = 0
        for batch in predict_dataloader:
            x = batch[0].detach().cpu().numpy().astype(np.float32)
            prob = self.model.predict_proba(x)[:, 1:2]
            bsz = x.shape[0]
            end_idx = start_idx + bsz
            all_prob[start_idx:end_idx, :] = prob
            if return_targets:
                y = batch[1].detach().cpu().numpy().reshape(-1, 1)
                all_targets[start_idx:end_idx, :] = y
            start_idx = end_idx

        return (all_prob, all_targets) if return_targets else all_prob

    def _define_checkpoint(self):
        return {
            "input_size": self.input_size,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "tree_method": self.tree_method,
            "enable_categorical": self.enable_categorical,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "device": self.device,
            "model_name": self._model_name,
            "xgb_pickle": pickle.dumps(self.model),
        }

    def save(self, path: str):
        self.checkpoint = self._define_checkpoint()
        if os.path.isdir(path):
            filename = "/" + str(self.mname) + ".pth"
            path = path + filename
        torch.save(self.checkpoint, path)

    @classmethod
    def load(cls, path: str, device: torch.device = DEVICE):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raiseWarning("The model has been loaded with weights_only set to False. According with torch documentation, this is not recommended if you do not trust the source of your saved model, as it could lead to arbitrary code execution.")
        checkpoint["device"] = device

        model = cls(
            input_size=checkpoint["input_size"],
            learning_rate=checkpoint["learning_rate"],
            n_estimators=checkpoint["n_estimators"],
            max_depth=checkpoint["max_depth"],
            subsample=checkpoint["subsample"],
            colsample_bytree=checkpoint["colsample_bytree"],
            reg_lambda=checkpoint["reg_lambda"],
            min_child_weight=checkpoint["min_child_weight"],
            tree_method=checkpoint["tree_method"],
            enable_categorical=checkpoint["enable_categorical"],
            early_stopping_rounds=checkpoint["early_stopping_rounds"],
            seed=checkpoint["seed"],
            model_name=checkpoint["model_name"],
            device=checkpoint["device"],
            verbose=False,
        )
        model.model = pickle.loads(checkpoint["xgb_pickle"])
        model.checkpoint = checkpoint
        return model

    @classmethod
    @cr('XGBClassifier.create_optimized_model')
    def create_optimized_model(
        cls,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        optuna_optimizer: OptunaOptimizer,
        **kwargs,
    ) -> Tuple["XGBClassifier", Dict]:
        r"""
        Create an optimized model using Optuna. The model is trained on the training dataset and evaluated on the validation dataset.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            optuna_optimizer (OptunaOptimizer): The optimizer to use for optimization.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[XGBClassifierModel, Dict]: The optimized model and the optimization parameters.
        """
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
            training_params = {}
            for k, spec in optimization_params.items():
                training_params[k] = suggest_value(k, spec, trial)
            training_params["save_logs_path"] = None

            model = cls(input_dim, verbose=False, **training_params)
            if optuna_optimizer.pruner is not None:
                n_estimators = training_params["n_estimators"]
                training_params["n_estimators"] = 1
                for estimator in range(n_estimators):
                    model.fit(train_dataset, verbose=False, **training_params)
                    y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                    loss_val = log_loss(y_true, y_pred, labels=[0, 1])
                    trial.report(loss_val, estimator)
                    if trial.should_prune():
                        raise TrialPruned()
            else:
                model.fit(train_dataset, verbose=False, **training_params)
                y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                loss_val = log_loss(y_true, y_pred, labels=[0, 1])

            return loss_val

        best_params = optuna_optimizer.optimize(objective_function=optimization_function)

        # Update params with best ones
        for param in best_params.keys():
            if param in optimization_params:
                optimization_params[param] = best_params[param]

        return cls(input_dim, **optimization_params), optimization_params
