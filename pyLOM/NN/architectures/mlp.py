import os
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from ..optimizer import OptunaOptimizer, TrialPruned
from .. import DEVICE, set_seed  # pyLOM/NN/__init__.py
from ... import pprint, cr  # pyLOM/__init__.py
from ...utils.errors import raiseWarning


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
        checkpoint_file (str, optional): Path to a checkpoint file to load the model from (default: ``None``).
        activation (torch.nn.Module, optional): Activation function to use (default: ``torch.nn.functional.relu``).
        device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).
        seed (int, optional): Seed for reproducibility (default: ``None``).
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
        seed: int = None,
        **kwargs: Dict,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.p_dropouts = p_dropouts
        self.activation = activation
        self.device = device

        super().__init__()
        if seed is not None:
            set_seed(seed)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = hidden_size
            self.layers.append(nn.Linear(in_size, out_size))
            if p_dropouts > 0:
                self.layers.append(nn.Dropout(p_dropouts))
        self.oupt = nn.Linear(hidden_size, output_size)

        for layer in self.layers:
            if (isinstance(layer, nn.Linear) and i % 2 == 0): # Initialize only non-dropout linear layers
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

        self.to(self.device)
    
    def forward(self, x):
        for layer in self.layers:
            z = self.activation(layer(x))
            x = z
        z = self.oupt(x)
        return z
    
    @cr('MLP.fit')
    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset = None,
        epochs: int = 100,
        lr: float = 0.001,
        lr_gamma: float = 1,
        lr_scheduler_step: int = 1,
        loss_fn: torch.nn.Module = torch.nn.MSELoss(),
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR,
        print_rate_batch: int = 0,
        print_rate_epoch: int = 1,
        **kwargs,
    )-> Dict[str, List[float]]:
        r"""
        Fit the model to the training data. If eval_set is provided, the model will be evaluated on this set after each epoch. 
        
        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset to fit the model.
            eval_dataset (torch.utils.data.Dataset): Evaluation dataset to evaluate the model after each epoch (default: ``None``).
            epochs (int, optional): Number of epochs to train the model (default: ``100``).
            lr (float, optional): Learning rate for the optimizer (default: ``0.001``).
            lr_gamma (float, optional): Multiplicative factor of learning rate decay (default: ``1``).
            lr_scheduler_step (int, optional): Number of epochs to decay the learning rate (default: ``1``).
            loss_fn (torch.nn.Module, optional): Loss function to optimize (default: ``torch.nn.MSELoss()``).
            optimizer_class (torch.optim.Optimizer, optional): Optimizer class to use (default: ``torch.optim.Adam``).
            scheduler_class (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler class to use. If ``None``, no scheduler will be used (default: ``torch.optim.lr_scheduler.StepLR``).
            print_rate_batch (int, optional): Print loss every ``print_rate_batch`` batches (default: ``1``). If set to ``0``, no print will be done.
            print_rate_epoch (int, optional): Print loss every ``print_rate_epoch`` epochs (default: ``1``). If set to ``0``, no print will be done.
            kwargs (dict, optional): Additional keyword arguments to pass to the DataLoader. Can be used to set the parameters of the DataLoader (see PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):
                
                - batch_size (int, optional): Batch size (default: ``32``).
                - shuffle (bool, optional): Shuffle the data (default: ``True``).
                - num_workers (int, optional): Number of workers to use (default: ``0``).
                - pin_memory (bool, optional): Pin memory (default: ``True``).

        Returns:
            Dict[str, List[float]]: Dictionary containing the training and evaluation losses.
        """
        dataloader_params = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
        }

        if not hasattr(self, "train_dataloader"):
            for key in dataloader_params.keys():
                if key in kwargs:
                    dataloader_params[key] = kwargs[key]
            train_dataloader = DataLoader(train_dataset, **dataloader_params)
        
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params) if eval_dataset is not None else None 

        if not hasattr(self, "optimizer"):
            self.optimizer = optimizer_class(self.parameters(), lr=lr)
        if not hasattr(self, "scheduler"):
            self.scheduler = scheduler_class(self.optimizer, step_size=lr_scheduler_step, gamma=lr_gamma) if scheduler_class is not None else None
        
        if hasattr(self, "checkpoint"):
            self.optimizer.load_state_dict(self.checkpoint["state"][0])
            if self.scheduler is not None and len(self.checkpoint["state"][1]) > 0:
                self.scheduler.load_state_dict(self.checkpoint["state"][1])
                self.scheduler.gamma = lr_gamma
                self.scheduler.step_size = lr_scheduler_step
            epoch_list = self.checkpoint["state"][2]
            train_loss_list = self.checkpoint["state"][3]
            test_loss_list = self.checkpoint["state"][4]
        else:
            epoch_list = []
            train_loss_list = []
            test_loss_list = []

        total_epochs = len(epoch_list)+epochs
        for epoch in range(1+len(epoch_list), 1+total_epochs):
            train_loss = 0.0
            self.train()
            for b_idx, batch in enumerate(train_dataloader):
                x_train, y_train = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                oupt = self(x_train)
                loss_val = loss_fn(oupt, y_train)
                loss_val.backward()
                self.optimizer.step()
                loss_val_item = loss_val.item()
                train_loss_list.append(loss_val_item)
                train_loss += loss_val_item
                if print_rate_batch != 0 and (b_idx % print_rate_batch) == 0:
                    pprint(0, "Batch %4d/%4d | Train loss (x1e5) %0.4f" % (b_idx, len(train_dataloader), loss_val_item * 1e5), flush=True)
                   
            train_loss = train_loss / (b_idx + 1)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            test_loss = 0.0
            if eval_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    for n_idx, sample in enumerate(eval_dataloader):
                        x_test, y_test = sample[0].to(self.device), sample[1].to(self.device)
                        test_output = self(x_test)
                        loss_val = loss_fn(test_output, y_test)
                        test_loss += loss_val.item()

                test_loss = test_loss / (n_idx + 1)
                test_loss_list.append(test_loss)
            
            if print_rate_epoch != 0 and (epoch % print_rate_epoch) == 0:
                test_log = f" | Test loss (x1e5) {test_loss * 1e5:.4f}" if eval_dataloader is not None else ""
                pprint(0, f"Epoch {epoch}/{total_epochs} | Train loss (x1e5) {train_loss * 1e5:.4f} {test_log}", flush=True)

            epoch_list.append(epoch)
            self.state = (
                self.optimizer.state_dict(),
                self.scheduler.state_dict() if self.scheduler is not None else {},
                epoch_list,
                train_loss_list,
                test_loss_list,
                )
            
        return {"train_loss": train_loss_list, "test_loss": test_loss_list}
    
    @cr('MLP.predict')
    def predict(
        self, 
        X: torch.utils.data.Dataset, 
        return_targets: bool = False,
        **kwargs,
    ):
        r"""
        Predict the target values for the input data. The dataset is loaded to a DataLoader with the provided keyword arguments. 
        The model is set to evaluation mode and the predictions are made using the input data. 
        To make a prediction from a torch tensor, use the `__call__` method directly.

        Args:
            X (torch.utils.data.Dataset): The dataset whose target values are to be predicted using the input data.
            rescale_output (bool): Whether to rescale the output with the scaler of the dataset (default: ``True``).
            kwargs (dict, optional): Additional keyword arguments to pass to the DataLoader. Can be used to set the parameters of the DataLoader (see PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):
                
                - batch_size (int, optional): Batch size (default: ``256``).
                - shuffle (bool, optional): Shuffle the data (default: ``False``).
                - num_workers (int, optional): Number of workers to use (default: ``0``).
                - pin_memory (bool, optional): Pin memory (default: ``True``).
 
        Returns:
            Tuple [np.ndarray, np.ndarray]: The predictions and the true target values.
        """
        dataloader_params = {
            "batch_size": 256,
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
                if return_targets:
                    all_targets[start_idx:end_idx, :] = y.cpu().numpy()
                start_idx = end_idx

        if return_targets:
            return all_predictions, all_targets
        else:
            return all_predictions

    def save(
        self, 
        path: str,
    ):
        r"""
        Save the model to a checkpoint file.

        Args:
            path (str): Path to save the model. It can be either a path to a directory or a file name. 
            If it is a directory, the model will be saved with a filename that includes the number of epochs trained.
        """
        checkpoint = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "p_dropouts": self.p_dropouts,
            "activation": self.activation,
            "device": self.device,
            "state_dict": self.state_dict(),
            "state": self.state,
        }
        
        if os.path.isdir(path):
            filename = "/trained_model_{:06d}".format(len(self.state[2])) + ".pth"
            path = path + filename
        torch.save(checkpoint, path)
    
    @classmethod
    def load(
        cls, 
        path: str,
        device: torch.device = DEVICE,
    ):
        r"""
        Load the model from a checkpoint file. Does not require the model to be instantiated.

        Args:
            path (str): Path to the file to load the model from.
            device (torch.device, optional): Device to use (default: ``torch.device("cpu")``).

        Returns:
            Model (MLP): The loaded model.
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
        )
        
        model.load_state_dict(checkpoint["state_dict"])
        model.checkpoint = checkpoint
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
        Create an optimized model using Optuna. The model is trained on the training dataset and evaluated on the validation dataset.
        
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
        def optimization_function(trial) -> float:
            training_params = {}       
            for key, params in optimization_params.items():
                training_params[key] = cls._get_optimizing_value(key, params, trial)
            model = cls(input_dim, output_dim, **training_params)
            if optuna_optimizer.pruner is not None:
                epochs = training_params["epochs"]
                training_params["epochs"] = 1
                for epoch in range(epochs):
                    model.fit(train_dataset, **training_params)
                    y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                    loss_val = ((y_pred - y_true)**2).mean()
                    trial.report(loss_val, epoch)
                    if trial.should_prune(): 
                        raise TrialPruned()
            else:
                model.fit(train_dataset, **training_params)
                y_pred, y_true = model.predict(eval_dataset, return_targets=True)
                loss_val = ((y_pred - y_true)**2).mean()
            
            return loss_val
        
        best_params = optuna_optimizer.optimize(objective_function=optimization_function)

        # Update params with best ones
        for param in best_params.keys():
            if param in optimization_params:
                optimization_params[param] = best_params[param]
        
        return cls(input_dim, output_dim, **optimization_params), optimization_params

    def _get_optimizing_value(name, value, trial):
        if isinstance(value, tuple) or isinstance(value, list):
            use_log = value[1] / value[0] >= 1000
            if isinstance(value[0], int):
                return trial.suggest_int(name, value[0], value[1], log=use_log)
            elif isinstance(value[0], float):
                return trial.suggest_float(name, value[0], value[1], log=use_log)
        else:
            return value
