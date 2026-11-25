#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# MLP (Multilayer Perceptron) class.
#
# Last rev: 06/05/2026

import os, math, inspect, numbers, numpy as np, torch, torch.nn as nn

from typing                 import Dict, List, Tuple, Callable

from ..                     import DEVICE, PIN_MEMORY, set_seed
from ..optimizer            import OptunaOptimizer, run_trial
from .kan                   import _create_dataloader, _EVAL_BATCH_SIZE
from ..utils                import suggest_params
from ..utils.optuna_utils   import trial_params as _trial_params, _flatten_params
from ..loss                 import BaseLossFunction, TorchLossAdapter
from ..utils.callbacks      import EarlyStopper
from ...                    import pprint, cr
from ...utils.errors        import raiseWarning, raiseError


class MLP(nn.Module):
    r"""
    Multi-layer perceptron model for regression tasks. The model is based on the PyTorch library `torch.nn` 
    (detailed documentation can be found at https://pytorch.org/docs/stable/nn.html).

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        n_layers (int): Number of hidden layers.
        hidden_size (int): Number of neurons in each hidden layer.
        p_dropouts (float, optional): Dropout probability for the hidden layers (default: ``0.0``).
        activation (torch.nn.Module, optional): Activation function to use (default: ``torch.nn.functional.relu``).
        device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).
        initialization (Callable, optional): Initialization function for the weights (default: ``torch.nn.init.xavier_uniform_``).
        initialization_kwargs (Dict, optional): Additional keyword arguments for the initialization function (default: ``{}``).
        seed (int, optional): Seed for reproducibility (default: ``None``).
        model_name (str, optional): Name of the model used as a base for the model name (default: ``"mlp"``).
        verbose (bool, optional): If ``True``, prints the model parameters and total size (default: ``True``).
        kwargs: Additional keyword arguments.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int,
        p_dropouts: float = 0.0,
        activation: torch.nn.Module = torch.nn.functional.relu,
        device: torch.device = DEVICE,
        initialization: Callable = torch.nn.init.xavier_uniform_,
        initialization_kwargs: Dict = {},
        seed: int = None,
        model_name: str = "mlp",
        verbose: bool = True,
        **kwargs: Dict,
    ):
        super().__init__()
        if seed is not None:
            set_seed(seed)
        
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.p_dropouts = p_dropouts
        self.activation = activation
        self.device = device
        self.initialization = initialization
        self.initialization_kwargs = initialization_kwargs
        self.seed = seed
        self.model_name = model_name

        self.layers = nn.ModuleList()
        for i in range(self.n_layers + 1):
            in_size = self.input_size if i == 0 else self.hidden_size
            out_size = self.hidden_size
            self.layers.append(nn.Linear(in_size, out_size))
            if self.p_dropouts > 0:
                self.layers.append(nn.Dropout(self.p_dropouts))
        self.oupt = nn.Linear(self.hidden_size, self.output_size)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.initialization(layer.weight, **self.initialization_kwargs)
                nn.init.zeros_(layer.bias)
        self.initialization(self.oupt.weight, **self.initialization_kwargs)
        nn.init.zeros_(self.oupt.bias)

        self.to(self.device)
        if verbose:
            pprint(0, f"Creating model: {self._model_name}")
            keys_print = [
                "input_size",
                "output_size",
                "n_layers",
                "hidden_size",
                "p_dropouts",
                "activation",
                "device",
                "initialization",
                "initialization_kwargs",
                "seed",
                "model_name",
            ]
            for key in keys_print:
                value = getattr(self, key)
                if callable(value):
                    value = value.__name__
                pprint(0, f"\t{key}: {value}")
            pprint(
                0,
                f"\ttotal_size (trainable parameters): {self._count_n_parameters()}\n"
            )
    
    def forward(self, x):
        for layer in self.layers:
            z = self.activation(layer(x))
            x = z
        z = self.oupt(x)
        return z
    
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

    def _count_n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def unpack_batch(self, batch):
        if len(batch) == 2:
            x, y = batch
            return {"x": x,"x_neighbors": None, "y": y, "y_neighbors": None}
        elif len(batch) == 4:
            x, x_neighbors, y, y_neighbors = batch
            return {"x": x, "x_neighbors": x_neighbors, "y": y, "y_neighbors": y_neighbors}
        else:
            raiseError("Unexpected batch format")
    
    @cr('MLP.fit')
    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        loss_fn: BaseLossFunction | torch.nn.Module = torch.nn.MSELoss(),
        loss_fn_kwargs: dict = {},
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler = None,
        optimizer_kwargs: dict = {},
        scheduler_kwargs: dict = {},
        dataloader_kwargs: dict = {},
        save_logs_path: str = None,
        print_rate_batch: int = 0,
        print_rate_epoch: int = 1,
        save_best: bool = False,
        early_stopping_patience: int = None,
        **kwargs,
    )-> Dict[str, List[float]]:
        r"""
        Fit the model to the training data. If eval_set is provided, the model will be evaluated on this set after each epoch. Calling ``fit`` again continues the training with the same optimizer and scheduler.
        
        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset to fit the model.
            eval_dataset (torch.utils.data.Dataset, optional): Evaluation dataset to evaluate the model after each epoch (default: ``None``).
            epochs (int, optional): Number of epochs to train the model (default: ``100``).
            batch_size (int, optional): Batch size for training (default: ``32``).
            lr (float, optional): Learning rate for the optimizer (default: ``0.001``).
            loss_fn (torch.nn.Module, optional): Loss function class to train with (default: ``torch.nn.MSELoss()``).
            loss_fn_kwargs (dict, optional): Additional keyword arguments to pass to the loss function (default: ``{}``).
            optimizer_class (torch.optim.Optimizer, optional): Optimizer class to use (default: ``torch.optim.Adam``).
            scheduler_class (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler class to use. If ``None``, no scheduler will be used (default: ``None``).
            optimizer_kwargs (dict, optional): Additional keyword arguments to pass to the optimizer (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer.
            scheduler_kwargs (dict, optional): Additional keyword arguments to pass to the scheduler (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler._LRScheduler.
            dataloader_kwargs (dict, optional): Additional keyword arguments to pass to the dataloader (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader. Overrides the following defaults: ``batch_size`` (taken from the ``batch_size`` argument),``shuffle=True``, ``num_workers=0``, ``pin_memory=PIN_MEMORY`` (default: ``False``).
            save_logs_path (str, optional): Path to save the training results. If ``None``, no results will be saved (default: ``None``).
            print_rate_batch (int, optional): Print loss every ``print_rate_batch`` batches (default: ``1``). If set to ``0``, no print will be done.
            print_rate_epoch (int, optional): Print loss every ``print_rate_epoch`` epochs (default: ``1``). If set to ``0``, no print will be done.
            save_best (bool, optional): Whether to save the model of the epoch with the lowest evaluation loss of this call to ``fit``, as ``best_model_<model_name>.pth`` in the folder of ``save_logs_path`` (the folder itself, or the one of the ``.npy`` file). The file is written each time the evaluation loss improves, with the optimizer and scheduler states of that epoch. The model in memory is the one of the last epoch (default: ``False``).
            early_stopping_patience (int, optional): Stop the training when the evaluation loss has not improved for this number of epochs, with ``pyLOM.NN.EarlyStopper(patience=early_stopping_patience)`` (an epoch whose evaluation loss equals the best one exactly only counts if the training loss increased). ``None`` disables it (default: ``None``).

            The model is evaluated on ``eval_dataset`` every epoch, in batches of at least ``8192`` samples and without shuffling; the evaluation loss is the mean squared error over all its samples.
            The training stops at the end of the first epoch whose training or evaluation loss is not finite. ``save_best`` and ``early_stopping_patience`` need ``eval_dataset``, and
            ``save_best`` also needs ``save_logs_path``: without them, they are ignored with a warning.

        Returns:
            Dict[str, List[float]]: Dictionary containing the training and evaluation results:
                - "train_loss": List of training losses for each epoch.
                - "test_loss": List of evaluation losses for each epoch (if eval_dataset is provided).
                - "lr": List of learning rates for each epoch.
                - "loss_iterations_train": List of training losses for each iteration.
                - "loss_iterations_test": List of evaluation losses for each iteration (if eval_dataset is provided).
                - "grad_norms": List of gradient norms for each iteration.
                - "check": List with a single boolean indicating successful training.
        """
        _dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": PIN_MEMORY,
            **dataloader_kwargs,
        }

        if not hasattr(self, "train_dataloader"):
            self.train_dataloader = _create_dataloader(train_dataset, **_dataloader_kwargs)
        
        if not hasattr(self, "eval_dataloader") and eval_dataset is not None:
            # The evaluation does not depend on the training batch size nor on the order of the samples
            self.eval_dataloader = _create_dataloader(
                eval_dataset,
                **{**_dataloader_kwargs, "batch_size": max(_dataloader_kwargs["batch_size"], _EVAL_BATCH_SIZE), "shuffle": False, "drop_last": False},
            )

        if early_stopping_patience is not None and not (isinstance(early_stopping_patience, numbers.Integral) and early_stopping_patience > 0):
            raiseError(f"early_stopping_patience must be a positive integer or None, got {early_stopping_patience}")
        if eval_dataset is None and (save_best or early_stopping_patience is not None):
            raiseWarning("save_best and early_stopping_patience need an eval_dataset: they are ignored")
            save_best, early_stopping_patience = False, None
        best_model_file = None
        if save_best:
            if save_logs_path is None:
                raiseWarning("save_best needs a save_logs_path: the model of the best epoch is not saved")
            else:
                best_model_dir = os.path.dirname(save_logs_path) if save_logs_path.endswith(".npy") else save_logs_path
                best_model_file = os.path.join(best_model_dir, f"best_model_{self._model_name}.pth")
                os.makedirs(best_model_dir or ".", exist_ok=True)

        if not hasattr(self, "optimizer"):
            self.optimizer = optimizer_class(
                self.parameters(),
                lr=lr,
                **optimizer_kwargs
            )
        if not hasattr(self, "scheduler"):
            if scheduler_class is not None:
                self.scheduler = scheduler_class(
                    self.optimizer,
                    **scheduler_kwargs
                )
            else:
                self.scheduler = None
        
        # States of a loaded checkpoint: they are restored once, by the first fit after load, which resumes the training
        if hasattr(self, "optimizer_state_dict"):
            self.optimizer.load_state_dict(self.optimizer_state_dict)
            del self.optimizer_state_dict
            restored_lr = self.optimizer.param_groups[0]["lr"]
            if restored_lr != lr:
                raiseWarning(f"Training resumes from the loaded optimizer state, with learning rate {restored_lr:.3e} instead of lr={lr:.3e}")

        if hasattr(self, "scheduler_state_dict"):
            if self.scheduler is not None:
                self.scheduler.load_state_dict(self.scheduler_state_dict)
            else:
                raiseWarning("The loaded scheduler state is discarded because no scheduler_class was given")
            del self.scheduler_state_dict

        train_losses = []
        test_losses = []
        loss_iterations_train = []
        loss_iterations_test = []
        current_lr_vec = []
        grad_norms = []

        total_epochs = len(train_losses) + epochs
        self.mname = f"{self._model_name}_{total_epochs:05d}"

        if isinstance(loss_fn, BaseLossFunction):
            pass
        elif isinstance(loss_fn, torch.nn.Module):
            loss_fn = TorchLossAdapter(loss_fn)
        elif callable(loss_fn) and issubclass(loss_fn, BaseLossFunction):
            loss_fn = loss_fn(**loss_fn_kwargs)
        else:
            raiseError("loss_fn must be either a BaseLossFunction, a torch.nn.Module, or a BaseLossFunction callable class")

        def compute_loss(batch, mode="train"):
            batch = self.unpack_batch(batch)
            if mode == "train":
                return loss_fn(self, batch)
            elif mode == "eval":
                preds = self(batch["x"])
                return torch.nn.functional.mse_loss(preds, batch["y"])
            else:
                raiseError(f"Unknown mode: {mode}")

        # Best evaluation loss of this call to fit (epochs numbered from 1), for save_best and the messages
        best_loss, best_epoch = math.inf, None
        stopper = EarlyStopper(patience=early_stopping_patience) if early_stopping_patience is not None else None
        prev_train_loss = math.inf

        for epoch in range(1+len(train_losses), 1+total_epochs):
            train_loss = 0.0
            self.train()
            if hasattr(self.train_dataloader.dataset, "train"):
                self.train_dataloader.dataset.train()

            def closure():
                self.optimizer.zero_grad()
                loss_val = compute_loss(batch, mode="train")
                loss_val.backward()
                loss_iterations_train.append(loss_val.item())
                return loss_val

            for b_idx, batch in enumerate(self.train_dataloader):
                batch = tuple(b.to(self.device) for b in batch)
                batch_loss = self.optimizer.step(closure).item()
                train_loss += batch_loss
                total_norm = torch.norm(torch.stack([p.grad.norm() for p in self.parameters() if p.grad is not None]))
                grad_norms.append(total_norm.item())
        
                if print_rate_batch != 0 and (b_idx % print_rate_batch) == 0:
                    pprint(
                        0,
                        f"\tBatch {b_idx+1}/{len(self.train_dataloader)}, Train Loss: {batch_loss:.4e}",
                        flush=True
                    )
                    
            train_loss = train_loss / (b_idx + 1)
            train_losses.append(train_loss)

            if self.scheduler is not None:
                self.scheduler.step()
            
            diverged = not math.isfinite(train_loss)

            test_loss = 0.0
            if eval_dataset is not None:
                self.eval()
                if hasattr(self.eval_dataloader.dataset, "eval"):
                    self.eval_dataloader.dataset.eval()
                eval_losses, eval_sizes = [], []
                with torch.no_grad():
                    for sample in self.eval_dataloader:
                        sample = tuple(s.to(self.device) for s in sample)
                        eval_losses.append(compute_loss(sample, mode="eval").detach())
                        eval_sizes.append(len(sample[0]))
                # One synchronization per evaluation; the loss is the mean over all the samples
                eval_losses = torch.stack(eval_losses).double().cpu().numpy()
                loss_iterations_test.extend(eval_losses.tolist())
                test_loss = float(np.average(eval_losses, weights=eval_sizes))
                test_losses.append(test_loss)
                diverged = diverged or not math.isfinite(test_loss)

            # Saved after the scheduler step, so that a training resumed from this file continues with the next epoch
            if eval_dataset is not None and test_loss < best_loss:
                best_loss, best_epoch = test_loss, epoch
                if best_model_file is not None:
                    self.save(best_model_file)
            stop_early = stopper is not None and stopper.early_stop(test_loss, prev_train_loss, train_loss)
            prev_train_loss = train_loss

            current_lr = self.optimizer.param_groups[0]["lr"]
            current_lr_vec.append(current_lr)

            if print_rate_epoch != 0 and (epoch % print_rate_epoch) == 0:
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / (1024**2)  # Memory usage in MB
                    memory_usage_str = f", MEM: {mem_used:.2f} MB"
                else:
                    memory_usage_str = ""
                test_log = f", Test Loss: {test_loss:.4e}" if eval_dataset is not None else ""
                pprint(
                    0,
                    f"\tEpoch {epoch}/{total_epochs}, Train Loss: {train_loss:.4e}{test_log}, "
                    f"LR: {current_lr:.2e}{memory_usage_str}",
                    flush=True
                )

            if diverged:
                pprint(0, f"Stopping training at epoch {epoch}/{total_epochs} due to a non-finite loss.", flush=True)
                break
            if stop_early:
                pprint(0, f"Stopping training at epoch {epoch}/{total_epochs}: the evaluation loss has not improved since epoch {best_epoch}.", flush=True)
                break

        if best_model_file is not None and best_epoch is not None and print_rate_epoch != 0:
            pprint(0, f"Model of the best epoch ({best_epoch}, evaluation loss {best_loss:.4e}) saved at {best_model_file}")

        if hasattr(self.train_dataloader.dataset, "eval"):
                self.train_dataloader.dataset.eval()

        results = {
            "train_loss": np.array(train_losses),
            "test_loss": np.array(test_losses),
            "lr": np.array(current_lr_vec),
            "loss_iterations_train": np.array(loss_iterations_train),
            "loss_iterations_test": np.array(loss_iterations_test),
            "grad_norms": np.array(grad_norms),
            "check": [True],
        }

        if save_logs_path is not None:
            pprint(0, f"\nPrinting losses on path: {save_logs_path}")
            if save_logs_path.endswith(".npy"):
                fn = save_logs_path
            else:
                fn = os.path.join(save_logs_path,f"training_results_{self.mname}.npy")
            np.save(fn, results)

        return results

    @cr('MLP.predict')
    def predict(
        self, 
        X: torch.utils.data.Dataset, 
        return_targets: bool = False,
        dataloader_kwargs: dict = {},
        **kwargs,
    ):
        r"""
        Predict the target values for the input data. The dataset is loaded to a DataLoader with the provided keyword arguments. 
        The model is set to evaluation mode and the predictions are made using the input data. 
        To make a prediction from a torch tensor, use the `__call__` method directly.

        Args:
            X (torch.utils.data.Dataset): The dataset whose target values are to be predicted using the input data.
            return_targets (bool, optional): If ``True``, the true target values will be returned along with the predictions (default: ``False``).
            dataloader_kwargs (dict, optional): Additional keyword arguments to pass to the dataloader (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader. Overrides the following defaults: ``batch_size=256`` ,``shuffle=False``, ``num_workers=0``, ``pin_memory=PIN_MEMORY`` (default: ``False``).
 
        Returns:
            Tuple [np.ndarray, np.ndarray]: The predictions and the true target values.
        """

        _dataloader_kwargs = {
            "batch_size": kwargs.get("batch_size", 256),
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": PIN_MEMORY,
            **dataloader_kwargs,
        }

        predict_dataloader = _create_dataloader(X, **_dataloader_kwargs)
        if hasattr(predict_dataloader.dataset, "eval"):
            predict_dataloader.dataset.eval()

        total_rows = len(predict_dataloader.dataset)
        num_columns = self.output_size
        all_predictions = np.empty((total_rows, num_columns))
        all_targets = np.empty((total_rows, num_columns))

        self.eval()
        start_idx = 0
        with torch.no_grad():
            for x, y in predict_dataloader:
                output = self(x.to(self.device))
                batch_size = x.size(0)
                end_idx = start_idx + batch_size
                all_predictions[start_idx:end_idx, :] = output.cpu().numpy()
                if return_targets:
                    all_targets[start_idx:end_idx, :] = y.cpu().numpy()
                start_idx = end_idx

        if return_targets:
            return all_predictions, all_targets
        else:
            return all_predictions

    def _define_checkpoint(self):
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "p_dropouts": self.p_dropouts,
            "activation": self.activation,
            "device": self.device,
            "initialization": self.initialization,
            "initialization_kwargs": self.initialization_kwargs,
            "seed": self.seed,
            "model_name": self._model_name,
            "state_dict": self.state_dict(),
        }

    def save(
        self, 
        path: str,
        save_only_model: bool = False
    ):
        r"""
        Save the model to a checkpoint file.

        Args:
            path (str): Path to save the model. It can be either a path to a directory or a file name. 
            save_only_model (bool, optional): If ``True``, only the model state will be saved. If ``False``, the optimizer and scheduler states will also be saved (default: ``False``).
        """
        checkpoint = self._define_checkpoint()

        if not save_only_model:
            if hasattr(self, "optimizer"):
                checkpoint["optimizer"] = self.optimizer.state_dict()
                if self.scheduler is not None:
                    checkpoint["scheduler"] = self.scheduler.state_dict()
            else:
                # A loaded model keeps the states of its checkpoint until it is trained
                if hasattr(self, "optimizer_state_dict"):
                    checkpoint["optimizer"] = self.optimizer_state_dict
                if hasattr(self, "scheduler_state_dict"):
                    checkpoint["scheduler"] = self.scheduler_state_dict
        
        if os.path.isdir(path):
            filename = "/" + str(self.mname) + ".pth"
            path = path + filename
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls, 
        path: str,
        device: torch.device = DEVICE,
        verbose: bool = True,
    ):
        r"""
        Load the model from a checkpoint file. Does not require the model to be instantiated. If the checkpoint contains the optimizer and scheduler states,
        they are restored by the next call to ``fit``, which resumes the training.

        Args:
            path (str): Path to the file to load the model from.
            device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).
            verbose (bool, optional): If ``True``, prints the model parameters and total size after loading (default: ``True``).

        Returns:
            model (MLP): The loaded model with the trained weights.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        raiseWarning("The model has been loaded with weights_only set to False. According with torch documentation, this is not recommended if you do not trust the source of your saved model, as it could lead to arbitrary code execution.")
        checkpoint['device'] = device
        model = cls(
            checkpoint["input_size"],
            checkpoint["output_size"],
            checkpoint["n_layers"],
            checkpoint["hidden_size"],
            checkpoint["p_dropouts"],
            checkpoint["activation"],
            checkpoint["device"],
            checkpoint["initialization"],
            checkpoint["initialization_kwargs"],
            # Passing the seed to the constructor would reseed the global random generators
            None,
            checkpoint["model_name"],
            verbose=verbose,
        )
        model.seed = checkpoint["seed"]

        model.load_state_dict(checkpoint["state_dict"])
        if checkpoint.get("optimizer") is not None:
            model.optimizer_state_dict = checkpoint["optimizer"]
        if checkpoint.get("scheduler") is not None:
            model.scheduler_state_dict = checkpoint["scheduler"]
        return model
    
    @classmethod
    @cr('MLP.create_optimized_model')
    def create_optimized_model(
        cls, 
        train_dataset: torch.utils.data.Dataset, 
        eval_dataset: torch.utils.data.Dataset, 
        optuna_optimizer: OptunaOptimizer,
        **kwargs,
    ) -> Tuple[nn.Module, Dict]:
        r"""
        Create an optimized model using Optuna. The model is trained on the training dataset and the metric to optimize is the loss on the evaluation dataset at the last epoch
        or, if ``save_best`` is ``True``, the lowest one of the training: the one of the model that ``save_best`` keeps. ``early_stopping_patience`` is applied as in ``fit``.
        The search space is sampled with ``pyLOM.NN.utils.suggest_params``: a pair of numbers ``(low, high)`` is a range, any other tuple or list is a list of choices
        (e.g. activation functions), a dictionary with a ``"type"`` key is an explicit specification (``"int"``, ``"float"``, ``"categorical"``, ``"pow2"``, ``"towards_one"``, ``"batch_scaled"``),
        any other dictionary is searched recursively, and any other value is fixed. ``"pow2"``, ``"towards_one"`` and ``"batch_scaled"`` are ordered values, as samplers such as QMC
        (which samples categorical parameters at random) need.

        Trials whose loss is not finite, or that run out of memory, are pruned. With ``save_best``, a trial that diverges keeps the loss of its best epoch, and is only pruned
        if no evaluation loss is finite. The number of epochs trained is stored in the user attribute ``epochs_trained`` of each trial. The parameters of any trial of the study
        can be recovered with ``trial_params``.

        If ``optuna_optimizer`` has ``seeds``, each trial trains one model per seed, and its value is the mean over the seeds of the metric of each model
        (see ``pyLOM.NN.optimizer.run_trial``).

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            optuna_optimizer (OptunaOptimizer): The optimizer to use for optimization.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple [MLP, Dict]: The optimized model and the optimization parameters.

        Example:
            >>> from pyLOM.NN import MLP, OptunaOptimizer
            >>> # Split the dataset
            >>> train_dataset, eval_dataset = dataset.get_splits([0.8, 0.2])
            >>> 
            >>> # Define the optimization parameters
            >>> optimization_params = {
            >>>     "lr": (0.00001, 0.01), # optimizable parameter
            >>>     "epochs": 50, # fixed parameter
            >>>     "n_layers": (1, 4),
            >>>     "batch_size": (128, 512),
            >>>     "hidden_size": (200, 400),
            >>>     "p_dropouts": (0.1, 0.5),
            >>>     "num_workers": 0,
            >>>     'print_rate_epoch': 5
            >>> }
            >>>
            >>> # Define the optimizer
            >>> optimizer = OptunaOptimizer(
            >>>     optimization_params=optimization_params,
            >>>     n_trials=5,
            >>>     direction="minimize",
            >>>     pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
            >>>     save_dir=None,
            >>> )
            >>>
            >>> # Create the optimized model
            >>> model, optimization_params = MLP.create_optimized_model(train_dataset, eval_dataset, optimizer)
            >>> 
            >>> # Fit the model
            >>> model.fit(train_dataset, eval_dataset, **optimization_params)
        """
        optimization_params = optuna_optimizer.optimization_params
        input_dim, output_dim = train_dataset[0][0].shape[0], train_dataset[0][1].shape[0]
        if eval_dataset is None:
            raiseError("create_optimized_model needs an eval_dataset to compute the metric to optimize")
        default_epochs = inspect.signature(cls.fit).parameters["epochs"].default

        def build_model(params):
            return cls(input_dim, output_dim, **{"verbose": False, **params})

        def optimization_function(trial) -> float:
            training_params = suggest_params(optimization_params, trial)
            return run_trial(trial, optuna_optimizer, build_model, training_params, train_dataset, eval_dataset, default_epochs)

        best_params = optuna_optimizer.optimize(objective_function=optimization_function)
        # The values of the parameters sampled through another one (e.g. pow2, through its exponent)
        best_params = cls.trial_params(optimization_params, _flatten_params(best_params))

        # Update params with best ones
        OptunaOptimizer.apply_to(optimization_params, optimized_params=best_params)

        return cls(input_dim, output_dim, **optimization_params), optimization_params

    @staticmethod
    def trial_params(optimization_params: Dict, params: Dict) -> Dict:
        r"""
        Parameters of the model and of its training in a trial of ``create_optimized_model``: the search space with the values that
        Optuna sampled in that trial. Useful to train again any trial of a study, not only the best one.

        Args:
            optimization_params (Dict): The search space given to ``create_optimized_model``.
            params (Dict): The parameters of the trial as stored by Optuna, e.g. ``study.trials[i].params``.

        Returns:
            Dict: The parameters, to build the model and to train it.

        Example:
            >>> study = optuna.load_study(study_name="optuna_study", storage="sqlite:///optuna_studies.db")
            >>> params = MLP.trial_params(optimization_params, study.best_trial.params)
            >>> model = MLP(input_size, output_size, **params)
            >>> model.fit(train_dataset, eval_dataset, **params)
        """
        return _trial_params(optimization_params, params)



class MLP_JIT(MLP):
    r"""
    Multi-layer perceptron model for regression tasks. The model is based on the PyTorch library `torch.nn` 
    (detailed documentation can be found at https://pytorch.org/docs/stable/nn.html).

    Class adapted for JIT compiling with `torch.jit.script`.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        n_layers (int): Number of hidden layers.
        hidden_size (int): Number of neurons in each hidden layer.
        p_dropouts (float, optional): Dropout probability for the hidden layers (default: ``0.0``).
        activation (torch.nn.Module, optional): Activation function to use (default: ``torch.nn.functional.relu``).
        device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).
        initialization (Callable, optional): Initialization function for the weights (default: ``torch.nn.init.xavier_uniform_``).
        initialization_kwargs (Dict, optional): Additional keyword arguments for the initialization function (default: ``{}``).
        seed (int, optional): Seed for reproducibility (default: ``None``).
        model_name (str, optional): Name of the model used as a base for the model name (default: ``"mlp"``).
        verbose (bool, optional): If ``True``, prints the model parameters and total size (default: ``True``).
        kwargs: Additional keyword arguments.
    """
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value.strip()