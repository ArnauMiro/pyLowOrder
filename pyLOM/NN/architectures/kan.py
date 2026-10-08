#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# KAN (Kolmogorov-Arnold Network) class.
#
# Last rev: 06/05/2026

import os, numpy as np, torch, torch.nn as nn, torch.optim as optim

from typing             import Dict, Tuple
from torch.utils.data   import DataLoader

from ..                 import DEVICE, PIN_MEMORY, set_seed
from ..optimizer        import OptunaOptimizer, TrialPruned
from ...                import pprint, cr
from ...utils.errors    import raiseWarning, raiseError
def _check_kwargs(kwargs, valid, caller):
    r"""
    ``KAN.__init__`` and ``KAN.fit`` receive each other's arguments in ``**kwargs``, because ``Pipeline`` and
    ``create_optimized_model`` pass the same dictionary to both. Any other name is a mistake and is not ignored.
    """
    unknown = sorted(set(kwargs) - valid)
    if unknown:
        raiseError(f"{caller} got unexpected arguments: {', '.join(unknown)}. Valid arguments are: {', '.join(sorted(valid))}")



class KAN(nn.Module):
    r"""
    KAN (Kolmogorov-Arnold Network) model for regression tasks. This model is based on https://arxiv.org/abs/2404.19756, inspired by the Kolmogorov-Arnold representation theorem.
    Each layer learns a polynomial function of each input for each output (Chebyshev or Jacobi basis), and the model is ``input -> [hidden -> dropout] x n_layers -> output``.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        n_layers (int): The number of hidden layers.
        hidden_size (int): The number of neurons in the hidden layers.
        layer_type (nn.Module): The type of layer to use in the model. It can be one of the following: ``JacobiLayer``, ``ChebyshevLayer``, or their names as strings.
        degree (int, Optional): The degree of the polynomial for the layers (default: ``5``).
        p_dropouts (float, Optional): The dropout probability (default: ``0.0``).
        device (torch.device, Optional): The device where the model is loaded (default: gpu if available).
        seed (int, Optional): The random seed for reproducibility (default: ``None``).
        model_name (str, Optional): The name of the model (default: ``"kan"``).
        verbose (bool, Optional): Whether to print the model information (default: ``True``).
        layer_kwargs (dict, Optional): Additional keyword arguments for the layers, e.g. ``{"a": 0.5, "b": 0.5}`` for ``JacobiLayer`` (default: ``None``).
        input_domain (Tuple[float, float], Optional): Range of the input data, which the first layer maps onto [-1, 1], where the polynomials are defined. Use ``(0, 1)`` for inputs scaled with ``MinMaxScaler``. If ``None``, the inputs are used as they are (default: ``None``).
        tanh_between_layers (bool, Optional): Whether to apply ``tanh`` to the inputs of the hidden and output layers, which keeps them in [-1, 1]. Without it the model is an unbounded polynomial, which can overflow in deep or high-degree networks (default: ``False``).
        init (str, Optional): Initialization of the coefficients: ``"default"``, normal with standard deviation 1/(n_in (degree + 1)), or ``"linear"``, where each layer starts close to a random linear map so that the signal does not vanish with depth (default: ``"default"``).
        **kwargs: Arguments of ``fit``, which are ignored here so that the same dictionary can be passed to both methods.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int,
        layer_type,
        degree = 5,
        p_dropouts: float = 0.0,
        device: torch.device = DEVICE,
        seed: int = None, 
        model_name: str = "kan",
        verbose: bool = True,
        layer_kwargs: Dict = None,
        input_domain: Tuple[float, float] = None,
        tanh_between_layers: bool = False,
        init: str = "default",
        **kwargs: Dict,
    ):
        super().__init__()
        _check_kwargs(kwargs, _parameter_names(KAN.fit), f"{type(self).__name__}.__init__")
        if seed is not None:
            set_seed(seed)

        layer_kwargs = dict(layer_kwargs) if layer_kwargs is not None else {}
        if "degree" in layer_kwargs:
            _warn_once("Passing the degree in layer_kwargs is deprecated, use the degree argument instead")
            degree = layer_kwargs.pop("degree")
        if isinstance(layer_type, str):
            if layer_type not in _LAYER_TYPES:
                raiseError(f"Unknown layer_type '{layer_type}', available options are: {', '.join(_LAYER_TYPES)}")
            layer_type = _LAYER_TYPES[layer_type]
        
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.layer_type = layer_type
        self.degree = degree
        self.p_dropouts = p_dropouts
        self.device = device
        self.seed = seed
        self.model_name = model_name
        self.layer_kwargs = layer_kwargs
        self.input_domain = _as_domain(input_domain)
        self.tanh_between_layers = tanh_between_layers
        self.init = init

        # Hidden layers with dropout
        hidden_layers = []
        for _ in range(self.n_layers):
            hidden_layers.append(self._create_layer(self.hidden_size, self.hidden_size, tanh=self.tanh_between_layers))
            hidden_layers.append(nn.Dropout(p=self.p_dropouts))

        self.kan_layers = nn.ModuleList(hidden_layers)

        # Input and output layers
        self.input = self._create_layer(self.input_size, self.hidden_size, domain=self.input_domain)
        self.output = self._create_layer(self.hidden_size, self.output_size, tanh=self.tanh_between_layers)

        self.to(self.device)
        if verbose:
            pprint(0, f"Creating model KAN: {self._model_name}")
            keys_print = [
                "input_size",
                "output_size",
                "n_layers",
                "hidden_size",
                "layer_type",
                "degree",
                "layer_kwargs",
                "p_dropouts",
                "input_domain",
                "tanh_between_layers",
                "init",
                "seed",
                "device",
            ]
            for key in keys_print:
                pprint(0, f"\t{key}: {getattr(self, key)}")
            pprint(0,
                f"\ttotal_size (trained params):\t{sum(p.numel() for p in self.parameters() if p.requires_grad)}"
            )

    def _create_layer(self, input_dim, output_dim, domain=None, tanh=False):
        # The options are only passed when used, so that layer classes without them keep working
        kwargs = dict(self.layer_kwargs)
        if self.init != "default":
            kwargs["init"] = self.init
        if domain is not None:
            kwargs["domain"] = domain
        if tanh:
            kwargs["tanh"] = True
        return self.layer_type(input_dim, output_dim, self.degree, **kwargs)

    def forward(self, x):
        x = self.input(x)
        for layer in self.kan_layers:
            x = layer(x)
        x = self.output(x)
        return x

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

    @cr("KAN.fit")
    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset = None,
        batch_size: int = 32,
        epochs: int = 100,
        lr: float = 0.001,
        optimizer_class=optim.Adam,
        scheduler_class: optim.lr_scheduler.LRScheduler | None = None,
        opti_kwargs={},
        scheduler_kwargs: dict = {},
        dataloader_kwargs: dict = {},
        scheduler_step_on_epoch: bool = True,
        print_eval_rate: int = 2,
        loss_fn=nn.MSELoss(),
        save_logs_path=None,
        verbose: bool = True,
        max_norm_grad=float("inf"),
        **kwargs,
    ):
        r"""
        Train the model using the provided training dataset. Calling ``fit`` again continues the training with the same optimizer and scheduler.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            batch_size (int): The batch size. (default: ``32``).
            epochs (int): The number of epochs to train the model. (default: ``100``).
            lr (float): The learning rate of the optimizer. (default: ``0.001``).
            optimizer_class (torch.optim, Optional): The optimizer to use. Available all optimizers from PyTorch except AdaDelta. (default: ``optim.Adam``).
            scheduler_class (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler class to use. If ``None``, no scheduler will be used (default: ``None``). ``ReduceLROnPlateau`` is stepped with the evaluation loss each time the model is evaluated, or with the training loss every epoch if there is no ``eval_dataset``.
            opti_kwargs (dict, optional): Additional keyword arguments to pass to the optimizer (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer.
            scheduler_kwargs (dict, optional): Additional keyword arguments to pass to the scheduler (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler._LRScheduler.
            dataloader_kwargs (dict, optional): Additional keyword arguments to pass to the dataloader (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader. Overrides the following defaults: ``batch_size`` (taken from the ``batch_size`` argument),``shuffle=True``, ``num_workers=0``, ``pin_memory=PIN_MEMORY`` (default: ``False``).
            scheduler_step_on_epoch (bool, Optional): Whether to step the scheduler every epoch (default: ``True``). If ``False``, the scheduler will be stepped every batch.
            print_eval_rate (int, Optional): The model is evaluated on ``eval_dataset`` every ``print_eval_rate`` epochs and at the last one. If set to 0, it is only evaluated at the last epoch (default: ``2``).
            loss_fn (torch.nn.Module, Optional): The loss function, called as ``loss_fn(prediction, target)`` and averaged over the batch (default: ``nn.MSELoss()``).
            save_logs_path (str, Optional): Path to save the training and evaluation losses, either a ``.npy`` file or a directory (default: ``None``). The file is overwritten, unless the call continues a previous training, whose losses are then extended.
            verbose (bool, Optional): Whether to print the training information (default: ``True``).
            max_norm_grad (float, Optional): The maximum norm of the gradients. ``inf`` or ``None`` disable the clipping (default: ``float('inf')``).

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the training and evaluation results:
                - "train_loss": Mean training loss of each epoch.
                - "test_loss": Loss on ``eval_dataset`` at each evaluation, averaged over all its samples.
                - "lr": Learning rate at the end of each epoch.
                - "loss_iterations_train": Training loss of each batch.
                - "loss_iterations_test": Evaluation loss of each evaluation batch.
                - "grad_norms": Norm of the gradients of each batch, before clipping.
                - "check": List with a single boolean indicating successful training.

            The training stops at the end of the first epoch whose training or evaluation loss is not finite.
        """
        _check_kwargs(kwargs, _parameter_names(*(c.__init__ for c in type(self).__mro__ if issubclass(c, KAN))), f"{type(self).__name__}.fit")

        _dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": PIN_MEMORY,
            **dataloader_kwargs,
        }
        train_loader = _create_dataloader(train_dataset, **_dataloader_kwargs)
        if len(train_loader) == 0:
            raiseError("The training dataset has no batches")
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = _create_dataloader(
                eval_dataset,
                **{**_dataloader_kwargs, "batch_size": max(_dataloader_kwargs["batch_size"], _EVAL_BATCH_SIZE), "shuffle": False, "drop_last": False},
            )
            if len(eval_loader) == 0:
                raiseError("The evaluation dataset is empty")

        if verbose:
            pprint(0, "")
            pprint(0, f"TRAINING MODEL {self._model_name}")
            pprint(0, "")
            pprint(0, "Conditions:")
            pprint(0, f"\tepochs:     {epochs}")
            pprint(0, f"\tbatch size: {batch_size}")
            pprint(0, f"\toptimizer class:  {optimizer_class}")
            pprint(0, f"\tscheduler:  {scheduler_class}")
            pprint(0, f"\tloss_fn:  {loss_fn}")
            pprint(0, f"\tsave_path:  {save_logs_path}")
            pprint(0, "\t")
            pprint(0, "Scheduler conditions:")
            for key, value in sorted(scheduler_kwargs.items()):
                if isinstance(value, dict):
                    pprint(0, f"\t{key}:")
                    for subkey, subvalue in sorted(value.items()):
                        pprint(0, f"\t{subkey}: {subvalue}")
                else:
                    pprint(0, f"\t{key}: {value}")
            pprint(0, "   ")
            pprint(0, "Dataloader conditions:")
            for key, value in sorted(_dataloader_kwargs.items()):
                if isinstance(value, dict):
                    pprint(0, f"\t{key}:")
                    for subkey, subvalue in sorted(value.items()):
                        pprint(0, f"\t{subkey}: {subvalue}")
                else:
                    pprint(0, f"\t{key}: {value}")
            pprint(0, "   ")
        
        # Training continues when the optimizer state comes from a previous call or from a loaded checkpoint
        continuing = hasattr(self, "optimizer") or hasattr(self, "optimizer_state_dict")
        if hasattr(self, "optimizer"):
            if type(self.optimizer) is not optimizer_class or self.optimizer.defaults.get("lr") != lr:
                raiseWarning("fit continues the previous training with its optimizer: the optimizer_class and lr arguments are ignored")
        else:
            self.optimizer = optimizer_class(
                self.parameters(),
                lr=lr,
                **opti_kwargs
            )

        if getattr(self, "scheduler", None) is None:
            if scheduler_class is not None:
                self.scheduler = scheduler_class(
                    self.optimizer,
                    **scheduler_kwargs
                )
            else:
                self.scheduler = None

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

        params = [p for p in self.parameters() if p.requires_grad]
        dtype = params[0].dtype
        clip_grads = max_norm_grad is not None and math.isfinite(max_norm_grad)
        plateau_scheduler = isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau)

        grad_norm = None
        def closure():
            nonlocal grad_norm
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            if clip_grads:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm_grad)
            else:
                grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in params if p.grad is not None]))
            return loss

        for epoch in range(epochs):
            self.train()
            batch_losses, batch_norms, batch_sizes = [], [], []
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device, dtype, non_blocking=True)
                targets = targets.to(self.device, dtype, non_blocking=True)
                batch_losses.append(self.optimizer.step(closure).detach())
                batch_norms.append(grad_norm)
                batch_sizes.append(len(inputs))
                if self.scheduler is not None and not scheduler_step_on_epoch and not plateau_scheduler:
                    self.scheduler.step()

            # The losses and gradient norms stay on the device during the epoch: one synchronization per epoch
            batch_stats = torch.stack([torch.stack(batch_losses).double(), torch.stack(batch_norms).double()]).cpu().numpy()
            loss_iterations_train.extend(batch_stats[0].tolist())
            grad_norms.extend(batch_stats[1].tolist())
            train_loss = float(np.average(batch_stats[0], weights=batch_sizes))
            train_losses.append(train_loss)
            diverged = not math.isfinite(train_loss)

            test_loss = None
            last_epoch = epoch == epochs - 1
            if eval_loader is not None and (diverged or last_epoch or (print_eval_rate > 0 and (epoch + 1) % print_eval_rate == 0)):
                self.eval()
                eval_losses, eval_sizes = [], []
                with torch.no_grad():
                    for x, y in eval_loader:
                        x = x.to(self.device, dtype, non_blocking=True)
                        y = y.to(self.device, dtype, non_blocking=True)
                        eval_losses.append(loss_fn(self(x), y))
                        eval_sizes.append(len(x))
                eval_losses = torch.stack(eval_losses).double().cpu().numpy()
                loss_iterations_test.extend(eval_losses.tolist())
                test_loss = float(np.average(eval_losses, weights=eval_sizes))
                test_losses.append(test_loss)
                diverged = diverged or not math.isfinite(test_loss)

            if self.scheduler is not None and not diverged:
                if plateau_scheduler:
                    if eval_loader is None:
                        self.scheduler.step(train_loss)
                    elif test_loss is not None:
                        self.scheduler.step(test_loss)
                elif scheduler_step_on_epoch:
                    self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            current_lr_vec.append(current_lr)

            if verbose:
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / (1024**2)  # Memory usage in MB
                    memory_usage_str = f", MEM: {mem_used:.2f} MB"
                else:
                    memory_usage_str = ""
                test_log = f", Test Loss: {test_loss:.4e}" if test_loss is not None else ""
                pprint(
                    0,
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4e}{test_log}, "
                    f"LR: {current_lr:.2e}{memory_usage_str}",
                )
            if diverged:
                if verbose:
                    pprint(0, f"Stopping training at epoch {epoch + 1} due to a non-finite loss.")
                break

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
            if save_logs_path.endswith(".npy"):
                fn = save_logs_path
            else:
                fn = os.path.join(save_logs_path, f"training_results_{self._model_name}.npy")
            os.makedirs(os.path.dirname(fn) or ".", exist_ok=True)
            if verbose:
                pprint(0, f"Printing losses on path {fn}")
            # Only a continued training extends the previous results; the file of any other run is overwritten
            if continuing and os.path.isfile(fn):
                results_old = np.load(fn, allow_pickle=True).item()
                for key in results.keys():
                    if key != 'check':
                        results[key] = np.concatenate((results_old.get(key, np.array([])), results[key]), axis=0)
                    else:
                        results[key] = list(results_old.get(key, [])) + results[key][:]
                if verbose:
                    pprint(0, "Updating previous data in file " + fn)
            np.save(fn, results)
            if verbose:
                pprint(0, f"Training results saved at {fn}")
        
        return results
            
    @cr("KAN.predict")
    def predict(
        self,
        X: torch.utils.data.Dataset,
        return_targets: bool = False,
        dataloader_kwargs: dict = {},
        **kwargs,
    ):
        r"""
        Predict the target values for the input data. The dataset is loaded into a DataLoader with the provided keyword arguments.
        The model is set to evaluation mode and the predictions are made using the input data. The output can be rescaled using
        the dataset scaler.

        Args:
            X (torch.utils.data.Dataset): The dataset whose target values are to be predicted using the input data.
            rescale_output (bool): Whether to rescale the output with the scaler of the dataset (default: ``True``).
            dataloader_kwargs (dict, optional): Additional keyword arguments to pass to the dataloader (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader. Overrides the following defaults: ``batch_size=256`` ,``shuffle=False``, ``num_workers=0``, ``pin_memory=PIN_MEMORY`` (default: ``False``).

        Returns:
            Tuple[np.ndarray, np.ndarray]: The predictions and the true target values.
        """

        _dataloader_kwargs = {
            "batch_size": kwargs.get("batch_size", 256),
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": PIN_MEMORY,
            **dataloader_kwargs,
        }

        predict_dataloader = DataLoader(X, **_dataloader_kwargs)

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
                all_targets[start_idx:end_idx, :] = y.cpu().numpy()
                start_idx = end_idx

        if return_targets:
            return all_predictions, all_targets
        else:
            return all_predictions
        
    def _define_checkpoint(self):
        checkpoint = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "layer_type": _layer_type_name(self.layer_type) or self.layer_type,
            "model_name": self._model_name,
            "p_dropouts": self.p_dropouts,
            "degree": self.degree,
            "layer_kwargs": self.layer_kwargs,
            "input_domain": self.input_domain,
            "tanh_between_layers": self.tanh_between_layers,
            "init": self.init,
            "seed": self.seed,
            "state_dict": self.state_dict(),
        }

        return checkpoint
    
    def save(self, path: str, save_only_model: bool = False):
        r"""
        Save the model to a checkpoint file.

        Args:
            path (str): Path to save the model. It can be either a path to a directory or a file name.
            If it is a directory, the model will be saved inside it as ``<model_name>.pth``.
            save_only_model (bool, Optional): Whether to only save the model, or also the optimizer and scheduler. Note that when this is true, you won't be able to resume training from checkpoint.(default: ``False``).
        """
        checkpoint = self._define_checkpoint()

        if not save_only_model:
            if hasattr(self, "optimizer"):
                checkpoint["optimizer"] = self.optimizer.state_dict()
                checkpoint["scheduler"] = self.scheduler.state_dict() if self.scheduler is not None else None
            else:
                # A loaded model keeps the states of its checkpoint until it is trained
                checkpoint["optimizer"] = getattr(self, "optimizer_state_dict", None)
                checkpoint["scheduler"] = getattr(self, "scheduler_state_dict", None)

        if os.path.isdir(path):
            path = os.path.join(path, f"{self._model_name}.pth")

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")):
        r"""
        Loads a model from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
            device (torch.device): Device where the model is loaded (default: cpu).

        Returns:
            model (KAN): The loaded KAN model with the trained weights.
        """

        pprint(0, "Loading model...")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        raiseWarning("The model has been loaded with weights_only set to False. According with torch documentation, this is not recommended if you do not trust the source of your saved model, as it could lead to arbitrary code execution.")
        state_dict = checkpoint['state_dict']
        optimizer_state = checkpoint.get('optimizer', None)
        scheduler_state = checkpoint.get('scheduler', None)
        del checkpoint['state_dict']
        del checkpoint['optimizer']
        del checkpoint['scheduler']
        model = cls(device=device, **checkpoint)

        if optimizer_state is not None:
            model.optimizer_state_dict = optimizer_state
        if scheduler_state is not None:
            model.scheduler_state_dict = scheduler_state
        model.load_state_dict(state_dict)
        pprint(0, f"Loaded KAN model: {checkpoint['model_name']}")
        keys_print = [
            "input_size",
            "output_size",
            "n_layers",
            "hidden_size",
            "layer_type",
            "p_dropouts",
            "degree"
        ]
        for key in keys_print:
            pprint(0, f"{key}: {checkpoint[key]}")

        return model

    @classmethod
    @cr("KAN.create_optimized_model")
    def create_optimized_model(
        cls, 
        train_dataset: torch.utils.data.Dataset, 
        eval_dataset: torch.utils.data.Dataset, 
        optuna_optimizer: OptunaOptimizer,
        **kwargs,
    ) -> Tuple[nn.Module, Dict]:
        r"""
        Create an optimized KAN model using Optuna. The model is trained on the training dataset and the metric to optimize is computed with the evaluation dataset.
        If the parameters from the optimizer are a tuple, the function will optimize the parameter. If the parameter is a single value, it will be fixed during optimization.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            optuna_optimizer (OptunaOptimizer): The optimizer to use for optimization.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple [KAN, Dict]: The optimized model and the optimization parameters.

        Example:
            >>> from pyLOM.NN import KAN, OptunaOptimizer
            >>> # Split the dataset
            >>> train_dataset, eval_dataset = dataset.get_splits([0.8, 0.2])
            >>>
            >>> # Define the optimization parameters
            >>> optimization_params = {
            >>>     "lr": (0.00001, 0.1),
            >>>     "batch_size": (10, 64),
            >>>     "hidden_size": (10, 40), # optimizable parameter
            >>>     "n_layers": (1, 4),
            >>>     "print_eval_rate": 2,
            >>>     "epochs": 10, # non-optimizable parameter
            >>>    "lr_kwargs":{
            >>>        "gamma": (0.95, 0.99),
            >>>        "step_size": 7000
            >>>    },
            >>>     "model_name": "kan_test_optuna",
            >>>     'device': device,
            >>>     "layer_type": (pyLOM.NN.ChebyshevLayer, pyLOM.NN.JacobiLayer),
            >>>     "layer_kwargs": {
            >>>         "degree": (3, 10),
            >>>     },
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
            >>> model, optimization_params = KAN.create_optimized_model(train_dataset, eval_dataset, optimizer)
            >>>
            >>> # Fit the model
            >>> model.fit(train_dataset, eval_dataset, **optimization_params)
        """
        optimization_params = optuna_optimizer.optimization_params
        input_dim, output_dim = train_dataset[0][0].shape[0], train_dataset[0][1].shape[0]
        
        def suggest_value(name, space, trial):
            if isinstance(space, dict):
                suggested_dict = {}
                for key, subspace in space.items():
                    full_name = f"{name}.{key}"
                    suggested_dict[key] = suggest_value(full_name, subspace, trial)
                return suggested_dict
            
            if isinstance(space, (tuple, list)):
                if len(space) == 2:
                    low, high = space
                    if isinstance(low, int) and isinstance(high, int):
                        
                        def is_power_of_2(n):
                            return n > 0 and (n & (n - 1)) == 0
                        
                        if is_power_of_2(low) and is_power_of_2(high):
                            power_low = int(np.log2(low))
                            power_high = int(np.log2(high))
                            power_diff = power_high - power_low
                            
                            if power_diff > 3:
                                choices = [2**p for p in range(power_low, power_high + 1)]
                                return trial.suggest_categorical(name, choices)
                        
                        use_log = (high / max(1, low)) >= 1000
                        return trial.suggest_int(name, low, high, log=use_log)

                    if isinstance(low, float) and isinstance(high, float):
                        use_log = (high / max(1e-12, low)) >= 1000
                        return trial.suggest_float(name, low, high, log=use_log)
                else:
                    return trial.suggest_categorical(name, space)
                
            return space
        
        def optimization_function(trial) -> float:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = None

            try: 
                training_params = {}       
                for key, params in optimization_params.items():
                    training_params[key] = suggest_value(key, params, trial)
                training_params["save_logs_path"] = None
                
                model = cls(input_dim, output_dim, **training_params)
                if optuna_optimizer.pruner is not None:
                    epochs = training_params["epochs"]
                    training_params["epochs"] = 1
                    for epoch in range(epochs):
                        results = model.fit(train_dataset, eval_dataset, **training_params)
                        loss_val = results["test_loss"][-1]
                        trial.report(loss_val, epoch)
                        if trial.should_prune(): 
                            raise TrialPruned()
                else:
                    results = model.fit(train_dataset, eval_dataset, **training_params)
                    loss_val = results["test_loss"][-1]
                
                return loss_val
            
            except RuntimeError as e:
                if "out of memory" in str(e) or "MEMORY" in str(e).upper():
                    print(f"Trial {trial.number} failed due to out of memory error. Pruning the trial.")
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
        
        return cls(input_dim, output_dim, **optimization_params), optimization_params


class KAN_SIN(KAN):
    """
    KAN model with a sine layer at the beginning. This model adds a sin layer at the beggining of a KAN model.
    
    Args:
        nneuron_sin (int): The number of neurons in the sine layer.
        sigma (float): The sigma (standard deviation of the weights) parameter for the sine layer. Default is ``1.0``.
        *args: Additional positional arguments to pass to the KAN class.
        **kwargs: Additional keyword arguments to pass to the KAN class.
    
    """
    
    def __init__(
        self,
        nneuron_sin:int,
        sigma:float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nneuron_sin = nneuron_sin
        self.sigma = sigma

        self.input = SineLayer(self.ninput, self.nneuron_sin, self.sigma)

        self.kan_layers[0] = self.layer_type(self.nneuron_sin, self.hidden_neur, self.degree)
        self.to(self.device)

        if self.intro:
            print('Adding sin layer at the beginning:')
            keys_print = ['nneuron_sin', 'sigma']
            for key in keys_print:
                print(f"   {key}: {getattr(self, key)}")
                
    def _define_checkpoint(self):
        checkpoint = super().define_checkpoint()
        checkpoint["sigma"] = self.sigma
        checkpoint["nneuron_sin"] = self.nneuron_sin
        checkpoint["state_dict"] = self.state_dict()

        return checkpoint


def _as_domain(domain):
    if domain is None:
        return None
    low, high = (float(value) for value in domain)
    if not low < high:
        raiseError(f"The domain must be (low, high) with low < high, got {domain}")
    return (low, high)


def _layer_input(x, input_dim, domain, tanh):
    r"""
    Input of a polynomial layer as a (batch_size, input_dim) tensor, mapping ``domain`` onto [-1, 1] and bounding it
    with ``tanh`` when requested.
    """
    if x.ndim > 1 and x.shape[-1] != input_dim:
        raiseError(f"The layer expects {input_dim} input features, got a tensor of shape {tuple(x.shape)}")
    x = torch.reshape(x, (-1, input_dim))  # shape = (batch_size, inputdim)
    if domain is not None:
        low, high = domain
        x = x * (2 / (high - low)) - (high + low) / (high - low)
    if tanh:
        x = torch.tanh(x)
    return x


def _init_coefficients(coeffs, init, slope=1.0):
    r"""
    Initialize the coefficients, of shape (input_dim, output_dim, degree + 1), of a polynomial layer.

    - ``"default"``: all of them from N(0, (1/(input_dim (degree + 1)))^2).
    - ``"linear"``: the layer starts close to a random linear map. The degree-1 coefficients are drawn from
      N(0, 1/(slope^2 input_dim)), where ``slope`` is the derivative of the degree-1 polynomial, the constant terms are
      zero and the higher degrees are drawn from N(0, _LINEAR_INIT_HIGH_STD^2/input_dim).
    """
    if init not in _INITS:
        raiseError(f"Unknown init '{init}', available options are: {', '.join(_INITS)}")
    input_dim, _, n_basis = coeffs.shape
    if init == "default":
        nn.init.normal_(coeffs, mean=0.0, std=1 / (input_dim * n_basis))
        return
    with torch.no_grad():
        nn.init.normal_(coeffs, mean=0.0, std=_LINEAR_INIT_HIGH_STD / math.sqrt(input_dim))
        coeffs[:, :, 0] = 0.0
        if n_basis > 1:
            nn.init.normal_(coeffs[:, :, 1], mean=0.0, std=1 / (slope * math.sqrt(input_dim)))


class ChebyshevLayer(nn.Module):
    """
    Chebyshev layer for KAN model. Each output is ``y_o = sum_i sum_k c_{i,o,k} T_k(x_i)``, with the Chebyshev polynomials ``T_k`` up to the given degree.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        degree (int): The degree of the Chebyshev polynomial.
        domain (Tuple[float, float], Optional): Range of the inputs, which is mapped onto [-1, 1], where the polynomials are defined. If ``None``, the inputs are used as they are (default: ``None``).
        tanh (bool, Optional): Whether to apply ``tanh`` to the inputs, after the ``domain`` mapping, to keep them in [-1, 1] (default: ``False``).
        init (str, Optional): Initialization of the coefficients, ``"default"`` or ``"linear"`` (see ``KAN``) (default: ``"default"``).
    """

    def __init__(self, input_dim, output_dim, degree, domain=None, tanh=False, init="default", **kwargs):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.domain = _as_domain(domain)
        self.tanh = tanh

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        _init_coefficients(self.cheby_coeffs, init)

    def forward(self, x):
        x = _layer_input(x, self.inputdim, self.domain, self.tanh)
        # Chebyshev polynomials: T_0 = 1, T_1 = x, T_k = 2x T_{k-1} - T_{k-2}
        cheby = [torch.ones_like(x), x][: self.degree + 1]
        if self.degree > 1:
            two_x = 2 * x
        for _ in range(2, self.degree + 1):
            cheby.append(two_x * cheby[-1] - cheby[-2])
        cheby = torch.stack(cheby, dim=-1)  # shape = (batch_size, inputdim, degree + 1)
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", cheby, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        return y


class JacobiLayer(nn.Module):
    """
    Jacobi layer for KAN model.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        degree (int): The degree of the Jacobi polynomial.
        a (float, Optional): The first parameter of the Jacobi polynomial, greater than -1 (default: ``1.0``).
        b (float, Optional): The second parameter of the Jacobi polynomial, greater than -1 (default: ``1.0``).
        domain (Tuple[float, float], Optional): Range of the inputs, which is mapped onto [-1, 1], where the polynomials are defined. If ``None``, the inputs are used as they are (default: ``None``).
        tanh (bool, Optional): Whether to apply ``tanh`` to the inputs, after the ``domain`` mapping, to keep them in [-1, 1] (default: ``False``).
        init (str, Optional): Initialization of the coefficients, ``"default"`` or ``"linear"`` (see ``KAN``) (default: ``"default"``).
    """

    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0, domain=None, tanh=False, init="default"):
        super().__init__()
        if a <= -1 or b <= -1:
            raiseError(f"The parameters of the Jacobi polynomials must be greater than -1, got a={a} and b={b}")
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree
        self.domain = _as_domain(domain)
        self.tanh = tanh

        self.jacobi_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        # The degree-1 polynomial is ((a - b) + (a + b + 2) x) / 2
        _init_coefficients(self.jacobi_coeffs, init, slope=(a + b + 2) / 2)

    def forward(self, x):
        x = _layer_input(x, self.inputdim, self.domain, self.tanh)
        # Jacobi polynomials by their three-term recurrence
        jacobi = [torch.ones_like(x)]
        if self.degree > 0:
            jacobi.append(((self.a - self.b) + (self.a + self.b + 2) * x) / 2)
        for i in range(2, self.degree + 1):
            theta_k = (
                (2 * i + self.a + self.b)
                * (2 * i + self.a + self.b - 1)
                / (2 * i * (i + self.a + self.b))
            )
            theta_k1 = (
                (2 * i + self.a + self.b - 1)
                * (self.a * self.a - self.b * self.b)
                / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            theta_k2 = (
                (i + self.a - 1)
                * (i + self.b - 1)
                * (2 * i + self.a + self.b)
                / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            jacobi.append((theta_k * x + theta_k1) * jacobi[i - 1] - theta_k2 * jacobi[i - 2])
        jacobi = torch.stack(jacobi, dim=-1)  # shape = (batch_size, inputdim, degree + 1)
        # Compute the Jacobian interpolation
        y = torch.einsum(
            "bid,iod->bo", jacobi, self.jacobi_coeffs
        )  # shape = (batch_size, outdim)
        return y

class SineLayer(nn.Module):
    """
    Sine layer for KAN model.
    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        sigma (float, Optional): The standard deviation of the weights (default: ``1.0``).
    """

    def __init__(self, input_size, output_size, sigma=1.0):
        super(SineLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigma = sigma
        self.init_weights()
 
    def init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=self.sigma)
        nn.init.zeros_(self.linear.bias)
 
    def forward(self, x):
        return torch.sin( self.linear(x))

# Layers that KAN accepts by name and stores by name in its checkpoints
_LAYER_TYPES = {"ChebyshevLayer": ChebyshevLayer, "JacobiLayer": JacobiLayer}

