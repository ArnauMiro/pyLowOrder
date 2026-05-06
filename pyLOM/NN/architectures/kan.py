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


class KAN(nn.Module):
    r"""
    KAN (Kolmogorov-Arnold Network) model for regression tasks. This model is based on https://arxiv.org/abs/2404.19756, inspired by the Kolmogorov-Arnold representation theorem.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        n_layers (int): The number of hidden layers.
        hidden_size (int): The number of neurons in the hidden layers.
        layer_type (nn.Module): The type of layer to use in the model. It can be one of the following: ``JacobiLayer``, ``ChebyshevLayer``.
        degree (int, Optional): The degree of the polynomial for the layers. It is only used if the layer type is ``JacobiLayer`` or ``ChebyshevLayer`` (default: ``5``).
        p_dropouts (float, Optional): The dropout probability (default: ``0.0``).
        device (torch.device, Optional): The device where the model is loaded (default: gpu if available).
        seed (int, Optional): The random seed for reproducibility (default: ``None``).
        model_name (str, Optional): The name of the model (default: ``"kan"``).
        verbose (bool, Optional): Whether to print the model information (default: ``True``).
        **layer_kwargs: Additional keyword arguments to pass to the layer type. For example, the order of the Taylor series or the degree of the Chebyshev polynomial.
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
        **kwargs: Dict,
    ):
        super().__init__()
        if seed is not None:
            set_seed(seed)
        
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

        # Hidden layers with dropout
        hidden_layers = []
        for _ in range(self.n_layers):
            hidden_layers.append(self.layer_type(self.hidden_size, self.hidden_size, self.degree))
            hidden_layers.append(nn.Dropout(p=self.p_dropouts))

        self.kan_layers = nn.ModuleList(hidden_layers)

        # Input and output layers
        self.input = self.layer_type(self.input_size, self.hidden_size, self.degree)
        self.output = self.layer_type(self.hidden_size, self.output_size, self.degree)

        self.to(self.device)
        if verbose:
            pprint(0, f"Creating model KAN: {self._model_name}")
            keys_print = [
                "input_size",
                "output_size",
                "n_layers",
                "hidden_size",
                "layer_type",
                "p_dropouts",
                "device",
            ]
            for key in keys_print:
                pprint(0, f"\t{key}: {getattr(self, key)}")
            pprint(0,
                f"\ttotal_size (trained params):\t{sum(p.numel() for p in self.parameters() if p.requires_grad)}"
            )

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
        scheduler_type="StepLR",
        opti_kwargs={},
        lr_kwargs={},
        dataloader_kwargs: dict = {},
        print_eval_rate: int = 2,
        loss_fn=nn.MSELoss(),
        save_logs_path=None,
        verbose: bool = True,
        max_norm_grad=float("inf"),
        **kwargs,
    ):
        r"""
        Train the model using the provided training dataset. The model is trained using the Adam optimizer with the provided learning rate and learning rate decay factor.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            batch_size (int): The batch size. (default: ``32``).
            epochs (int): The number of epochs to train the model. (default: ``100``).
            lr (float): The learning rate for the Adam optimizer. (default: ``0.001``).
            optimizer_class (torch.optim, Optional): The optimizer to use. Available all optimizers from PyTorch except AdaDelta. (default: ``optim.Adam``).
            scheduler_type (str, opcional): Scheduler type to adjust the learning rate dynamically. (default: ``"StepLR"``).
                Available options:

                - "StepLR": Reduce the learning rate by a factor every ``step_size`` batches.
                - "ReduceLROnPlateau": Reduces the learning rate when a metric has stopped improving.
                - "OneCycleLR": Adjust the learning rate in a single cycle of the training.
            opti_kwargs (dict, Optional): Additional keyword arguments to pass to the optimizer (default: `{}`).
            lr_kwargs (dict, opcional): Dictionary containing the specific parameters for the learning rate scheduler. (default: ``{}``).
                Some examples are:
                
                - StepLR: {"step_size": int, "gamma": float}.
                - ReduceLROnPlateau: {"mode": str, "factor": float, "patience": int}.
                - OneCycleLR: {"anneal_strategy": str, "div_factor": float}.
            dataloader_kwargs (dict, optional): Additional keyword arguments to pass to the dataloader (default: ``{}``). See PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader. Overrides the following defaults: ``batch_size`` (taken from the ``batch_size`` argument),``shuffle=True``, ``num_workers=0``, ``pin_memory=PIN_MEMORY`` (default: ``False``).
            print_eval_rate (int, Optional): The model will be evaluated every ``print_eval_rate`` epochs and the losses will be printed. If set to 0, nothing will be printed (default: ``2``).
            loss_fn (torch.nn.Module, Optional): The loss function (default: ``nn.MSELoss()``).
            save_logs_path (str, Optional): Path to save the training and evaluation losses (default: ``None``).
            verbose (bool, Optional): Whether to print the training information (default: ``True``).
            max_norm_grad (float, Optional): The maximum norm of the gradients (default: ``float('inf')``).
        """
        _dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": PIN_MEMORY,
            **dataloader_kwargs,
        }

        if verbose:
            pprint(0, "")
            pprint(0, f"TRAINNING MODEL {self._model_name}")
            pprint(0, "")
            pprint(0, "Conditions:")
            pprint(0, f"\tepochs:     {epochs}")
            pprint(0, f"\tbatch size: {batch_size}")
            pprint(0, f"\toptimizer class:  {optimizer_class}")
            pprint(0, f"\tscheduler:  {scheduler_type}")
            pprint(0, f"\tloss_fn:  {loss_fn}")
            pprint(0, f"\tsave_path:  {save_logs_path}")
            pprint(0, "\t")
            pprint(0, "Scheduler conditions:")
            for key, value in sorted(lr_kwargs.items()):
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
        
        if not hasattr(self, "train_loader"):
            self.train_loader = DataLoader(train_dataset, **_dataloader_kwargs)
        
        if not hasattr(self, "test_loader") and eval_dataset is not None:
            self.test_loader = DataLoader(eval_dataset, **_dataloader_kwargs)

        train_losses = []
        test_losses = []
        loss_iterations_train = []
        loss_iterations_test = []
        current_lr_vec = []
        grad_norms = []

        if not hasattr(self, "optimizer"):
            self.optimizer = optimizer_class(
                self.parameters(),
                lr=lr,
                **opti_kwargs
            )

        if not hasattr(self, "scheduler"):
            if scheduler_type == "StepLR":
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **lr_kwargs)
            elif scheduler_type == "ReduceLROnPlateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **lr_kwargs
                )
            elif scheduler_type == "OneCycleLR":
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=lr,
                    steps_per_epoch=len(self.train_loader),
                    epochs=1,
                    **lr_kwargs,
                )
            else:
                raiseError(f"Invalid scheduler_type: {scheduler_type}. Available options are: 'StepLR', 'ReduceLROnPlateau', 'OneCycleLR'")

        if hasattr(self, "optimizer_state_dict"):
            self.optimizer.load_state_dict(self.optimizer_state_dict)
            del self.optimizer_state_dict

        if hasattr(self, "scheduler_state_dict"):
            self.scheduler.load_state_dict(self.scheduler_state_dict)
            del self.scheduler_state_dict

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0

            def closure():
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=max_norm_grad
                )
                loss_iterations_train.append(loss.item())
                return loss

            for inputs, targets in self.train_loader:
                inputs, targets = (
                    inputs.float().to(self.device),
                    targets.float().to(self.device),
                )
                train_loss += self.optimizer.step(closure).item()
                total_norm = torch.norm(
                    torch.stack([p.grad.norm() for p in self.parameters() if p.grad is not None])
                )
                grad_norms.append(total_norm.item())
                # if scheduler_type != "ReduceLROnPlateau":
                #     self.scheduler.step()
                #     current_lr = self.optimizer.param_groups[0]["lr"]
                #     current_lr_vec.append(current_lr)

            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)

            # if scheduler_type == "ReduceLROnPlateau":
            #     self.scheduler.step(train_loss)
            if scheduler_type == "StepLR":
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                current_lr_vec.append(current_lr)
                
            test_loss = 0.0
            if eval_dataset is not None:
                if (epoch + 1) % print_eval_rate == 0:
                    self.eval()
                    with torch.no_grad():
                        for inputs, targets in self.test_loader:
                            inputs, targets = (
                                inputs.float().to(self.device),
                                targets.float().to(self.device),
                            )
                            outputs = self(inputs)
                            loss = loss_fn(outputs, targets)
                            loss_iterations_test.append(loss.item())
                            test_loss += loss.item()

                    test_loss /= len(self.test_loader)
                    test_losses.append(test_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            current_lr_vec.append(current_lr)

            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / (1024**2)  # Memory usage in MB
                memory_usage_str = f", MEM: {mem_used:.2f} MB"
            else:
                memory_usage_str = ""

            if verbose:
                pprint(
                    0,
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4e}, Test Loss: {test_loss:.4e}, "
                    f"LR: {current_lr:.2e}{memory_usage_str}",
                )
            # check if the loss is NaN and stop training
            if torch.isnan(torch.tensor(test_loss)):
                if verbose:
                    pprint(0, f"Stopping training at epoch {epoch + 1} due to NaN in test loss.")
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
                if not os.path.isdir(save_logs_path):
                    save_logs_path = '.'
                fn = os.path.join(save_logs_path, f"training_results_{self._model_name}.npy")
            if verbose:
                pprint(0, f"Printing losses on path {fn}")
            if os.path.isfile(fn):
                results_old = np.load(fn, allow_pickle=True).item()
                for key in results.keys():
                    if key != 'check':
                        results[key] = np.concatenate((results_old[key], results[key]), axis=0)
                    else:
                        results[key] = results_old[key] + results[key][:]
                if verbose:
                    pprint(0, "Updating previous data in file" + fn)
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
            "layer_type": self.layer_type,
            "model_name": self._model_name,
            "p_dropouts": self.p_dropouts,
            "degree": self.degree,
            "state_dict": self.state_dict(),
        }

        return checkpoint
    
    def save(self, path: str, save_only_model: bool = False):
        r"""
        Save the model to a checkpoint file.

        Args:
            path (str): Path to save the model. It can be either a path to a directory or a file name.
            If it is a directory, the model will be saved with a filename that includes the number of epochs trained.
            save_only_model (bool, Optional): Whether to only save the model, or also the optimizer and scheduler. Note that when this is true, you won't be able to resume training from checkpoint.(default: ``False``).
        """
        checkpoint = self._define_checkpoint()

        if not save_only_model:
            checkpoint["optimizer"] = self.optimizer.state_dict()
            checkpoint["scheduler"] = self.scheduler.state_dict()

        if os.path.isdir(path):
            filename = f"{self._model_name}.pth"
            path = path + filename

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

class ChebyshevLayer(nn.Module):
    """
    Chebyshev layer for KAN model.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        degree (int): The degree of the Chebyshev polynomial.
    """

    def __init__(self, input_dim, output_dim, degree, **kwargs):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Initialize Chebyshev polynomial tensors
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = (
                2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
            )
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", cheby, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class JacobiLayer(nn.Module):
    """
    Jacobi layer for KAN model.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        degree (int): The degree of the Jacobi polynomial.
        a (float, Optional): The first parameter of the Jacobi polynomial (default: ``1.0``).
        b (float, Optional): The second parameter of the Jacobi polynomial (default: ``1.0``).
    """

    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )

        nn.init.normal_(
            self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1))
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if (self.degree > 0):  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
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
            jacobi[:, :, i] = (
                (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone()
                - theta_k2 * jacobi[:, :, i - 2].clone()
            )  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum(
            "bid,iod->bo", jacobi, self.jacobi_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
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