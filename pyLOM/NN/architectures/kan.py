import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ... import cr, pprint  # pyLOM/__init__.py
from .. import DEVICE  # pyLOM/NN/__init__.py
from ...utils.errors import raiseError, raiseWarning
from ..optimizer import OptunaOptimizer


class KAN(nn.Module):
    r"""
    KAN (Kolmogorov-Arnold Network) model for regression tasks. This model is based on https://arxiv.org/abs/2404.19756, inspired by the Kolmogorov-Arnold representation theorem.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        n_layers (int): The number of hidden layers.
        hidden_size (int): The number of neurons in the hidden layers.
        layer_type (nn.Module): The type of layer to use in the model. It can be one of the following: ``JacobiLayer``, ``ChebyshevLayer``.
        model_name (str): The name of the model.
        p_dropouts (float, Optional): The dropout probability (default: ``0.0``).
        device (torch.device, Optional): The device where the model is loaded (default: gpu if available).
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
        model_name: str = "KAN",
        p_dropouts: float = 0.0,
        device: torch.device = DEVICE,
        verbose: bool = True,
        **layer_kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.layer_type = layer_type
        self.model_name = model_name
        self.p_dropouts = p_dropouts
        self.device = device

        # Hidden layers with dropout
        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(layer_type(hidden_size, hidden_size, **layer_kwargs))
            hidden_layers.append(nn.Dropout(p=p_dropouts))

        self.kan_layers = nn.ModuleList(hidden_layers)

        # Input and output layers
        self.input = layer_type(input_size, hidden_size, **layer_kwargs)
        self.output = layer_type(hidden_size, output_size, **layer_kwargs)

        self.to(self.device)
        if verbose:
            pprint(0, f"Creating model KAN: {self.model_name}")
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

    @cr("KAN.fit")
    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        epochs: int = 100,
        lr: float = 0.001,
        optimizer_class=optim.Adam,
        scheduler_type="StepLR",
        opti_kwargs={},
        lr_kwargs={},
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
            lr_kwargs (dict, opcional): Dictionary containing the specific parameters for the learning rate scheduler. (default: ``{}``).
                Some examples are:
                
                - StepLR: {"step_size": int, "gamma": float}.
                - ReduceLROnPlateau: {"mode": str, "factor": float, "patience": int}.
                - OneCycleLR: {"anneal_strategy": str, "div_factor": float}.
            opti_kwargs (dict, Optional): Additional keyword arguments to pass to the optimizer (default: `{}`).
            print_eval_rate (int, Optional): The model will be evaluated every ``print_eval_rate`` epochs and the losses will be printed. If set to 0, nothing will be printed (default: ``2``).
            loss_fn (torch.nn.Module, Optional): The loss function (default: ``nn.MSELoss()``).
            save_logs_path (str, Optional): Path to save the training and evaluation losses (default: ``None``).
            verbose (bool, Optional): Whether to print the training information (default: ``True``).
            max_norm_grad (float, Optional): The maximum norm of the gradients (default: ``float('inf')``).
            kwargs (dict, Optional): Additional keyword arguments to pass to the DataLoader. Can be used to set the parameters of the DataLoader (see PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):
               
                - batch_size (int, Optional): Batch size (default: ``32``).
                - shuffle (bool, Optional): Shuffle the data (default: ``True``).
                - num_workers (int, Optional): Number of workers to use (default: ``0``).
                - pin_memory (bool, Optional): Pin memory (default: ``True``).
        """
        if verbose:
            pprint(0, "")
            pprint(0, f"TRAINNING MODEL {self.model_name}")
            pprint(0, "")
            pprint(0, "Conditions:")
            pprint(0, f"\tepochs:     {epochs}")
            pprint(0, f"\tbatch size: 2**{int(np.log2(batch_size))}")
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
        dataloader_params = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
        }
        for key in dataloader_params.keys():
            if key in kwargs:
                dataloader_params[key] = kwargs[key]
        train_loader = DataLoader(train_dataset, **dataloader_params)
        test_loader = DataLoader(eval_dataset, **dataloader_params)

        train_losses = torch.tensor([], device=self.device)
        test_losses = torch.tensor([], device=self.device)

        loss_iterations_train = []
        loss_iterations_test = []
        self.optimizer = optimizer_class(self.parameters(), lr=lr, **opti_kwargs)
        current_lr_vec = []

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
                steps_per_epoch=len(train_loader),
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

            for inputs, targets in train_loader:
                inputs, targets = (
                    inputs.float().to(self.device),
                    targets.float().to(self.device),
                )
                train_loss += self.optimizer.step(closure).item()
                if scheduler_type != "ReduceLROnPlateau":
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    current_lr_vec.append(current_lr)

            train_loss /= len(train_loader)
            train_losses = torch.cat(
                (
                    train_losses,
                    torch.tensor([train_loss], dtype=torch.float64, device=self.device),
                )
            )
            if scheduler_type == "ReduceLROnPlateau":
                self.scheduler.step(train_loss)

            if (epoch + 1) % print_eval_rate == 0:
                self.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = (
                            inputs.float().to(self.device),
                            targets.float().to(self.device),
                        )
                        outputs = self(inputs)
                        loss = loss_fn(outputs, targets)
                        loss_iterations_test.append(loss.item())
                        test_loss += loss.item()

                test_loss /= len(test_loader)
                test_losses = torch.cat(
                    (
                        test_losses,
                        torch.tensor(
                            [test_loss], dtype=torch.float64, device=self.device
                        ),
                    )
                )
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

        if save_logs_path is not None:
            if not os.path.exists(save_logs_path):
                os.makedirs(save_logs_path)
            train_losses_np = train_losses.cpu().numpy()
            test_losses_np = test_losses.cpu().numpy()
            current_lr_np = np.array(current_lr_vec)
            if os.path.isdir(save_logs_path):
                if verbose:
                    pprint(0, f"Printing losses on path {save_logs_path}")
            else:
                if verbose:
                    pprint(0, "Path not found. Printing losses on local folder (.)")
                save_logs_path = "."

            np.save(
                save_logs_path + "/train_losses_" + self.model_name + ".npy",
                train_losses_np,
            )
            np.save(
                save_logs_path + "/test_losses_" + self.model_name + ".npy",
                test_losses_np,
            )
            np.save(
                save_logs_path + "/current_lr_" + self.model_name + ".npy",
                current_lr_np,
            )
            np.save(
                save_logs_path + "/losses_iterations_train_" + self.model_name + ".npy",
                np.array(loss_iterations_train),
            )
            np.save(
                save_logs_path + "/losses_iterations_test_" + self.model_name + ".npy",
                np.array(loss_iterations_test),
            )

        return {
            "train_loss": train_losses,
            "test_loss": test_losses,
            "lr": current_lr_vec,
            "loss_iterations_train": loss_iterations_train,
            "loss_iterations_test": loss_iterations_test,
        }

    @cr("KAN.predict")
    def predict(
        self,
        X: torch.utils.data.Dataset,
        return_targets: bool = False,
        **kwargs,
    ):
        r"""
        Predict the target values for the input data. The dataset is loaded into a DataLoader with the provided keyword arguments.
        The model is set to evaluation mode and the predictions are made using the input data. The output can be rescaled using
        the dataset scaler.

        Args:
            X (torch.utils.data.Dataset): The dataset whose target values are to be predicted using the input data.
            rescale_output (bool): Whether to rescale the output with the scaler of the dataset (default: ``True``).
            kwargs (dict, Optional): Additional keyword arguments to pass to the DataLoader. Can be used to set the parameters of the DataLoader (see PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):
                
                - **batch_size** (int, Optional): Batch size (default: ``32``).  
                - **shuffle** (bool, Optional): Shuffle the data (default: ``True``).  
                - **num_workers** (int, Optional): Number of workers to use (default: ``0``).  
                - **pin_memory** (bool, Optional): Pin memory (default: ``True``).  

        Returns:
            Tuple[np.ndarray, np.ndarray]: The predictions and the true target values.
        """

        dataloader_params = {
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": True,
        }

        for key in dataloader_params.keys():
            if key in kwargs:
                dataloader_params[key] = kwargs[key]

        predict_dataloader = DataLoader(X, **dataloader_params)

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

    def save(self, path: str, save_only_model: bool = False):
        r"""
        Save the model to a checkpoint file.

        Args:
            path (str): Path to save the model. It can be either a path to a directory or a file name.
            If it is a directory, the model will be saved with a filename that includes the number of epochs trained.
            save_only_model (bool, Optional): Whether to only save the model, or also the optimizer and scheduler. Note that when this is true, you won't be able to resume training from checkpoint.(default: ``False``).
        """
        checkpoint = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "layer_type": self.layer_type,
            "model_name": self.model_name,
            "p_dropouts": self.p_dropouts,
            "state_dict": self.state_dict(),
        }
        if hasattr(self.input, "degree"):
            checkpoint["degree"] = self.input.degree

        if not save_only_model:
            checkpoint["optimizer"] = self.optimizer.state_dict()
            checkpoint["scheduler"] = self.scheduler.state_dict()

        if os.path.isdir(path):
            filename = f"{self.model_name}.pth"
            path = path + filename

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")):
        """
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

        degree = checkpoint["degree"]
        layer_kwargs = {"degree": degree}

        model = cls(
            input_size=checkpoint["input_size"],
            output_size=checkpoint["output_size"],
            n_layers=checkpoint["n_layers"],
            hidden_size=checkpoint["hidden_size"],
            layer_type=checkpoint["layer_type"],
            model_name=checkpoint["model_name"],
            p_dropouts=checkpoint["p_dropouts"],
            device=device,
            **layer_kwargs,  # Pass the specific layer arguments
        )

        if "optimizer" in checkpoint:
            model.optimizer_state_dict = checkpoint["optimizer"]
        if "scheduler" in checkpoint:
            model.scheduler_state_dict = checkpoint["scheduler"]

        model.load_state_dict(checkpoint["state_dict"])
        pprint(0, f"Loaded KAN model: {checkpoint['model_name']}")
        keys_print = [
            "input_size",
            "output_size",
            "n_layers",
            "hidden_size",
            "layer_type",
            "p_dropouts",
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
        """
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
        optimizing_parameters = optuna_optimizer.optimization_params
        input_size, output_size = (
            train_dataset[0][0].shape[0],
            train_dataset[0][1].shape[0],
        )

        def optimization_function(trial) -> float:
            model_parameters, training_parameters = cls._sample_kan_parameters(
                trial, optimizing_parameters
            )
            model = cls(
                input_size=input_size,
                output_size=output_size,
                **model_parameters,
            )
            model_logs = model.fit(train_dataset, eval_dataset, **training_parameters)
            return model_logs["test_loss"][-1].item()

        best_params = optuna_optimizer.optimize(optimization_function)
        # Update the optimizing parameters with the best parameters found
        optimizing_parameters.update(best_params)
        if "layer_kwargs" in optimizing_parameters:
            del optimizing_parameters["layer_kwargs"]
        # Update the learning rate kwargs with the best parameters found
        if "lr_kwargs" in optimizing_parameters:
            for key in optimizing_parameters["lr_kwargs"]:
                if key in best_params:
                    optimizing_parameters["lr_kwargs"][key] = best_params[key]

        # Separate the model and training parameters from the best parameters found.
        # Note: now optimizing_parameters contains the best parameters found and does not contain any tuple,
        # so this is an easy wan to separate the training and model parameters
        model_parameters, training_parameters = cls._sample_kan_parameters(
            None, optimizing_parameters
        )
        model = cls(input_size=input_size, output_size=output_size, **model_parameters)
        return model, training_parameters

    @classmethod
    def _sample_kan_parameters(cls, trial, optimizing_parameters):
        training_parameters = {}
        model_parameters = {}
        mandatory_params = ["n_layers", "hidden_size", "layer_type"]
        for param in mandatory_params:
            if param in optimizing_parameters:
                model_parameters[param] = cls._suggest_value(trial, param, optimizing_parameters)
            else:
                raiseError(f"A value or range to optimize for the {param} must be provided")
        if "p_dropouts" in optimizing_parameters:
            model_parameters["p_dropouts"] = cls._suggest_float_value(
            trial, "p_dropouts", optimizing_parameters
            )
        else:
            model_parameters["p_dropouts"] = 0.0
        
        if "layer_kwargs" in optimizing_parameters:
            for key in optimizing_parameters["layer_kwargs"]:
                if key not in ["degree", "a", "b"]:
                    raiseError(f"Invalid key {key} in layer_kwargs")
                else:
                    model_parameters[key] = cls._suggest_value(
                        trial, key, optimizing_parameters["layer_kwargs"])
        elif "degree" in optimizing_parameters:
            model_parameters["degree"] = cls._suggest_int_value(
                trial, "degree", optimizing_parameters
            )
        else:
            raiseError("Layer kwargs with at least a key for degree must be provided")

        if "epochs" in optimizing_parameters:
            training_parameters["epochs"] = cls._suggest_int_value(
                trial, "epochs", optimizing_parameters
            )
        else:
            training_parameters["epochs"] = 100
        if "batch_size" in optimizing_parameters:
            training_parameters["batch_size"] = cls._suggest_int_value(
                trial, "batch_size", optimizing_parameters
            )
        else:
            training_parameters["batch_size"] = 32
        if "lr" in optimizing_parameters:
            training_parameters["lr"] = cls._suggest_float_value(
                trial, "lr", optimizing_parameters, log_scale=True
            )
        else:
            training_parameters["lr"] = 0.001

        if "max_norm_grad" in optimizing_parameters:
            training_parameters["max_norm_grad"] = cls._suggest_float_value(
                trial, "max_norm_grad", optimizing_parameters
            )
        else:
            training_parameters["max_norm_grad"] = float("inf")
            
        if "optimizer_class" in optimizing_parameters:
            training_parameters["optimizer_class"] = cls._suggest_categorical_value(
                trial, "optimizer_class", optimizing_parameters
            )
        else:
            training_parameters["optimizer_class"] = optim.Adam

        if "lr_kwargs" in optimizing_parameters:
            training_parameters["lr_kwargs"] = {}
            for key in optimizing_parameters["lr_kwargs"]:
                training_parameters["lr_kwargs"][key] = cls._suggest_value(
                    trial, key, optimizing_parameters["lr_kwargs"]
                )
        else:
            training_parameters["lr_kwargs"] = {}


        # non optimizing parameters
        for no_optimizing_param in ["save_logs_path", "print_eval_rate", "loss_fn", "opti_kwargs", "scheduler_type"]:
            if no_optimizing_param in optimizing_parameters:
                training_parameters[no_optimizing_param] = optimizing_parameters[
                    no_optimizing_param
                ]
                if isinstance(training_parameters[no_optimizing_param], tuple):
                    raiseError(f"Invalid value for {no_optimizing_param}. It is not an optimizable parameter.")

        for no_optimizing_param in ["device", "model_name"]:
            if no_optimizing_param in optimizing_parameters:
                model_parameters[no_optimizing_param] = optimizing_parameters[
                    no_optimizing_param
                ]

        return model_parameters, training_parameters
    
    def _suggest_value(
        trial, parameter_name, optimizing_parameters,
    ):
        if isinstance(optimizing_parameters[parameter_name], tuple):
            if isinstance(optimizing_parameters[parameter_name][0], int):
                return trial.suggest_int(
                    parameter_name, *optimizing_parameters[parameter_name]
                )
            elif isinstance(optimizing_parameters[parameter_name][0], float):
                return trial.suggest_float(
                    parameter_name, *optimizing_parameters[parameter_name]
                )
            else:
                return trial.suggest_categorical(
                    parameter_name, optimizing_parameters[parameter_name]
                )
        else:
            return optimizing_parameters[parameter_name]

    def _suggest_int_value(
        trial, parameter_name, optimizing_parameters, log_scale=False
    ):
        if isinstance(optimizing_parameters[parameter_name], tuple):
            return trial.suggest_int(
                parameter_name, *optimizing_parameters[parameter_name], log=log_scale
            )
        else:
            return optimizing_parameters[parameter_name]

    def _suggest_float_value(
        trial, parameter_name, optimizing_parameters, log_scale=False
    ):
        if isinstance(optimizing_parameters[parameter_name], tuple):
            return trial.suggest_float(
                parameter_name, *optimizing_parameters[parameter_name], log=log_scale
            )
        else:
            return optimizing_parameters[parameter_name]

    def _suggest_categorical_value(trial, parameter_name, optimizing_parameters):
        if isinstance(optimizing_parameters[parameter_name], tuple):
            return trial.suggest_categorical(
                parameter_name, optimizing_parameters[parameter_name]
            )
        else:
            return optimizing_parameters[parameter_name]


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
