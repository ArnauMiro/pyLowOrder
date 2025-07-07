#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 30/03/2025

import numpy as np
import os
import json
import warnings

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import TensorDataset as TorchDataset
from torch.utils.data import DataLoader
from torch.nn import ELU
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

from ..utils import count_trainable_params
from ..gns import Graph, InputsInjector, ManualNeighborLoader, ShapeValidator
from .. import Dataset as NNDataset
from .. import DEVICE, set_seed
from ... import pprint, cr
from ..optimizer import OptunaOptimizer, TrialPruned

from typing import Dict, Tuple, Union, List




class GNSMLP(nn.Module):
    r"""
    Simple feedforward MLP with activation and dropout, designed for internal use in GNS.

    Args:
        input_size (int): Input size.
        output_size (int): Output size.
        hidden_size (int): Number of units in each hidden layer.
        num_hidden_layers (int): Number of hidden layers.
        drop_p (float): Dropout probability.
        activation (nn.Module): Activation function.
    """
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers=1, drop_p=0.5, activation=ELU()):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.activation = activation
        self.dropout = nn.Dropout(p=drop_p)

    @cr('GNSMLP.forward')
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)



class MessagePassingLayer(MessagePassing):
    r"""
    Message passing layer using mean aggregation and MLPs for message and update functions.

    Args:
        in_channels (int): Input feature size for messages.
        out_channels (int): Output feature size after message passing.
        hidden_size (int): Hidden size for internal MLPs.
        message_hidden_layers (int): Number of hidden layers in the message MLP.
        update_hidden_layers (int): Number of hidden layers in the update MLP.
        activation (nn.Module): Activation function (e.g., nn.ELU()).
        drop_p (float): Dropout probability.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        message_hidden_layers: int = 1,
        update_hidden_layers: int = 1,
        activation: nn.Module = ELU(),
        drop_p: float = 0.0,
    ):
        super().__init__(aggr='mean')
        self.dropout = nn.Dropout(p=drop_p)

        self.phi = GNSMLP(
            input_size=in_channels,
            output_size=out_channels,
            hidden_size=hidden_size,
            num_hidden_layers=message_hidden_layers,
            drop_p=drop_p,
            activation=activation,
        )

        self.gamma = GNSMLP(
            input_size=2 * out_channels,
            output_size=out_channels,
            hidden_size=hidden_size,
            num_hidden_layers=update_hidden_layers,
            drop_p=drop_p,
            activation=activation,
        )

    @cr('MessagePassingLayer.forward')
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    @cr('MessagePassingLayer.message')
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.phi(torch.cat([x_i, x_j, edge_attr], dim=1))

    @cr('MessagePassingLayer.update')
    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return self.gamma(torch.cat([x, aggr_out], dim=1))






class GNS(nn.Module):
    r"""
    Graph Neural Solver class for predicting aerodynamic variables on RANS meshes.
    The model uses a message-passing architecture with MLPs for the message and update functions.

    Args:
        input_dim (int): The dimension of the operational parameters.
        latent_dim (int): The number of latent features.
        output_dim (int): The number of output features.
        hidden_size (int): The number of hidden units in the MLPs.
        num_msg_passing_layers (int): The number of GNN layers.
        encoder_hidden_layers (int): The number of hidden layers in the encoder.
        decoder_hidden_layers (int): The number of hidden layers in the decoder.
        message_hidden_layers (int): The number of hidden layers in the message MLP.
        update_hidden_layers (int): The number of hidden layers in the update MLP.
        graph (Union[torch_geometric.data.Data, Graph]): The graph object with node and edge features.
        p_dropouts (float, optional): The dropout probability. Default is ``0``.
        checkpoint_file (str, optional): The path to the checkpoint file. Default is ``None``.
        activation (Union[str, nn.Module]): The activation function to use.
        device (Union[str, torch.device]): The device to use for training. Default is ``'cuda'`` if available, otherwise ``'cpu'``.
        seed (int): The random seed for reproducibility. Default is None.
    """

    def __init__(self,
                graph: Graph,
                input_dim: int,
                latent_dim: int,
                output_dim: int,
                hidden_size: int,
                num_msg_passing_layers: int,
                encoder_hidden_layers: int,
                decoder_hidden_layers: int,
                message_hidden_layers: int,
                update_hidden_layers: int,
                **kwargs) -> None:
        super().__init__()

        # --- Parse kwargs ---
        self.device = torch.device(kwargs.get("device", DEVICE))
        self.seed = kwargs.get("seed", None)
        self.p_dropouts = kwargs.get("p_dropouts", 0.0)
        
        # --- Activation setup ---
        activation = kwargs.get("activation", ELU())
        if isinstance(activation, str):
            if not hasattr(nn, activation):
                raise ValueError(f"Activation function '{activation}' not found in torch.nn")
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation

        # --- Device setup ---
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available. Please use CPU instead.")
            torch.cuda.set_device(0)
        pprint(0, f"Using device: {self.device}", flush=True)

        # --- Seed ---
        if self.seed is not None:
            set_seed(self.seed)

        # --- Save config ---
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_msg_passing_layers = num_msg_passing_layers
        self.encoder_hidden_layers = encoder_hidden_layers
        self.decoder_hidden_layers = decoder_hidden_layers
        self.message_hidden_layers = message_hidden_layers
        self.update_hidden_layers = update_hidden_layers

        self.model_dict = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "hidden_size": hidden_size,
            "num_msg_passing_layers": num_msg_passing_layers,
            "encoder_hidden_layers": encoder_hidden_layers,
            "decoder_hidden_layers": decoder_hidden_layers,
            "message_hidden_layers": message_hidden_layers,
            "update_hidden_layers": update_hidden_layers,
            "p_dropouts": self.p_dropouts,
            "activation": activation.__class__.__name__,
            "seed": self.seed,
        }

        # --- Validator setup ---
        self.validator = ShapeValidator(self.input_dim, self.output_dim)

        # --- Graph setup ---
        self._graph = None
        self._encoder_input_dim = None
        self._edge_dim = None
        self.graph = graph  # sets _encoder_input_dim and _edge_dim

        # --- Inputs injector (prepares subgraphs for forward pass) ---
        self.injector = InputsInjector(device=self.device)

        # --- Optimizer and training state ---
        self.state = {}
        self.optimizer = None
        self.scheduler = None
        self.checkpoint = None

        # --- Build modules ---
        self._build_encoder()
        self._build_message_passing_layers()
        self._build_decoder()
        self.groupnorm = nn.GroupNorm(num_groups=min(2, self.latent_dim), num_channels=self.latent_dim).to(self.device)

    def _build_encoder(self):
        self.encoder = GNSMLP(
            input_size=self._encoder_input_dim,
            output_size=self.latent_dim,
            hidden_size=self.hidden_size,
            activation=self.activation,
            drop_p=self.p_dropouts
        ).to(self.device)

    def _build_decoder(self):
        self.decoder = GNSMLP(
            input_size=self.latent_dim,
            output_size=self.output_dim,
            hidden_size=self.hidden_size,
            activation=self.activation,
            drop_p=self.p_dropouts
        ).to(self.device)

    def _build_message_passing_layers(self):
        self.conv_layers_list = nn.ModuleList([
            MessagePassingLayer(
                in_channels=2 * self.latent_dim + self._edge_dim,
                out_channels=self.latent_dim,
                hidden_size=self.hidden_size,
                message_hidden_layers=self.message_hidden_layers,
                update_hidden_layers=self.update_hidden_layers,
                activation=self.activation,
                drop_p=self.p_dropouts,
            ) for _ in range(self.num_msg_passing_layers)
        ]).to(self.device)

    @property
    def graph(self) -> Graph:
        r"""Graph property to get the graph object."""
        return self._graph
    
    @graph.setter
    def graph(self, graph: Graph) -> None:
        r"""Graph property to set the graph object."""
        if not isinstance(graph, Graph):
            raise TypeError("Graph must be of type Graph.")
        if self._graph is not None:
            warnings.warn("Graph is already set. Overwriting.")
        
        graph.validate()
        
        if graph.device != self.device:
            graph = graph.to(self.device)


        self._encoder_input_dim = graph.x.shape[1] + self.input_dim # Update node features dimension
        self._edge_dim = graph.edge_attr.shape[1] # Update edge features dimension

        self._graph = graph

    @cr('GNS.forward')
    def forward(self, graph: Union[Data, Graph]) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            graph (Data or Graph): Prepared input graph with operational parameters embedded, useful for internal calls or external debugging.

        Returns:
            Tensor: Predicted values for all nodes in the graph.
        """
        x, edge_index, edge_attr = graph.x_injected, graph.edge_index, graph.edge_attr

        h = self.activation(self.encoder(x))
        for conv in self.conv_layers_list:
            h = self.groupnorm(self.activation(conv(h, edge_index, edge_attr)))
        y_hat = self.decoder(h)
        return y_hat
    
    @cr('GNS.predict')
    def predict(
        self,
        X: Union[Tensor, TorchDataset],
        batch_size: int = 1,
        node_batch_size: int = 256,
        **kwargs
    ) -> torch.Tensor:
        """
        Run inference on the model.

        Args:
            X (Tensor or Dataset):
                - Tensor: shape [B, D] where B=batch_size and D=input_dim.
                - Dataset: yields (input, target) or input tensors with shapes [B, D] and [B, N, F] respectively where B=batch_size, N=num_nodes and F=output_dim.
            batch_size (int): Used only if X is a Dataset. Ignored if X is a Tensor (batch size is set to B).

        Returns:
            Tensor: Predictions for all nodes in each graph. Shape [B * N, F] where B=batch_size, N=num_nodes and F=output_dim.
        """
        self._validate_shapes(X)
        self.eval()

        with torch.no_grad():
            if isinstance(X, Tensor):
                dataset = TorchDataset(X)
                B = len(dataset)
                input_dataloader = self._init_dataloader(dataset, batch_size=B)

            elif isinstance(X, TorchDataset):
                input_dataloader = self._init_dataloader(X, batch_size=batch_size)

            subgraph_loader = self._init_subgraph_loader(
                batch_size=node_batch_size,
                input_nodes=kwargs.get("input_nodes", None)
            )

            return self._run_epoch(input_dataloader, subgraph_loader, is_train=False, return_loss=False)


    @cr('GNS.fit')
    def fit(self, train_dataset:Union[TorchDataset, NNDataset], eval_dataset: Union[TorchDataset, NNDataset]=None, **kwargs) -> Dict:
        """
        Train the model using subgraph batching over both the training inputs and the node space.

        Supports evaluation on a separate dataset and configurable training parameters.

        Args:
            train_dataset: Dataset of input parameters and labels. At each iteration, it yields:
                - inputs (Tensor): Shape [B, D] where B=batch_size and D=input_dim.
                - targets (Tensor): Shape [B, N, F] where B=batch_size, N=num_nodes and F=output_dim.
            eval_dataset: Optional dataset for evaluation. Same format as `train_dataset`.
            **kwargs: Training hyperparameters such as:
                - epochs (int)
                - lr (float)
                - lr_gamma (float)
                - lr_scheduler_step (int)
                - loss_fn (callable)
                - optimizer (class)
                - scheduler (class)
                - batch_size (int)
                - node_batch_size (int)
                - input_nodes (Tensor or None)

        Returns:
            Dict: Dictionary with keys `"train_loss"` and `"test_loss"` listing per-epoch values.
        """
        # --- Validate inputs ---
        self._validate_shapes(train_dataset)
        if eval_dataset is not None:
            self._validate_shapes(eval_dataset)
        
        # --- Parse kwargs ---
        epochs = kwargs.get("epochs", 100)
        lr = kwargs.get("lr", 1e-4)
        lr_gamma = kwargs.get("lr_gamma", 0.1)
        lr_scheduler_step = kwargs.get("lr_scheduler_step", 1)
        loss_fn = kwargs.get("loss_fn", torch.nn.MSELoss(reduction='mean'))
        optimizer = kwargs.get("optimizer", torch.optim.Adam)
        scheduler = kwargs.get("scheduler", torch.optim.lr_scheduler.StepLR)
        print_rate_epoch = kwargs.get("print_rate_epoch", 1)

        # --- Instantiate dataloaders ---        
        input_dataloader = self._init_dataloader(train_dataset, **kwargs)
        eval_dataloader = self._init_dataloader(eval_dataset, **kwargs) if eval_dataset is not None else None
        subgraph_loader = self._init_subgraph_loader(
            batch_size=kwargs.get("node_batch_size", 256),
            input_nodes=kwargs.get("input_nodes", None)
        )

        # --- Set optimizer and scheduler ---
        if self.optimizer is None:
            self.optimizer = optimizer(self.parameters(), lr=lr)
        if self.scheduler is None:
            self.scheduler = scheduler(self.optimizer, step_size=lr_scheduler_step, gamma=lr_gamma) if scheduler is not None else None

        # --- Initialize state ---
        if self.checkpoint is not None:
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

        # --- Start training loop ---
        total_epochs = len(epoch_list) + epochs
        for epoch in range(1 + len(epoch_list), 1 + total_epochs):

            train_loss = self._run_epoch(input_dataloader, subgraph_loader, return_loss=True, loss_fn=loss_fn, is_train=True)
            train_loss_list.append(train_loss)

            if eval_dataloader is not None:
                test_loss = self._run_epoch(eval_dataloader, subgraph_loader, return_loss=True, loss_fn=loss_fn, is_train=False)
                test_loss_list.append(test_loss)

            if print_rate_epoch and epoch % print_rate_epoch == 0:
                test_log = f" | Eval loss:{test_loss:.4e}" if eval_dataloader else ""
                pprint(0, f"Epoch {epoch}/{total_epochs} | Train loss: {train_loss:.4e}{test_log}", flush=True)
                if self.device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 2
                    reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 2
                    pprint(0, f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB", flush=True)

            epoch_list.append(epoch)
            self.state = (
                self.optimizer.state_dict(),
                self.scheduler.state_dict() if self.scheduler is not None else {},
                epoch_list,
                train_loss_list,
                test_loss_list,
            )

        return {"train_loss": train_loss_list, "test_loss": test_loss_list}

    @cr('GNS._run_epoch')
    def _run_epoch(self, input_dataloader, subgraph_loader, return_loss: bool = False, loss_fn=None, is_train: bool = False) -> Union[float, Tensor]:
        """
        Core routine for both training and inference, handling subgraph batching and device transfer.

        Args:
            input_dataloader: Dataloader yielding input parameter batches.
            node_dataloader: Dataloader yielding seed node indices for subgraph batching.
            return_loss (bool): Whether to compute and return loss instead of predictions.
            loss_fn (callable, optional): Loss function to apply if return_loss is True.
            is_train (bool): Whether to run in training mode (with gradient updates).

        Returns:
            float or Tensor:
                - If return_loss: Average loss over the full dataset.
                - Else: Concatenated predictions over all batches.
        """

        self.train() if is_train else self.eval()
        outputs = []
        total_loss = 0.0
        context = torch.enable_grad() if is_train else torch.no_grad()

        last_graph = None
        last_output = None
        last_targets = None
        last_loss = None

        with context:
            num_batches = 0
            for batch in input_dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs_batch = batch[0]
                    targets_batch = batch[1] if len(batch) > 1 else None
                inputs_batch = inputs_batch.to(self.device)

                for subgraph in subgraph_loader:

                    G_batch_prepared = self.injector.inject_replicate(subgraph, inputs_batch, 
                                                                     targets_batch=targets_batch)

                    if is_train:
                        self.optimizer.zero_grad()

                    output = self.forward(G_batch_prepared)[G_batch_prepared.seed_mask] # [B * N, F]

                    if return_loss:
                        targets = G_batch_prepared.y[G_batch_prepared.seed_mask]
                        loss = loss_fn(output, targets)
                        if is_train:
                            loss.backward()
                            self.optimizer.step()
                        total_loss += loss.item()
                        num_batches += 1
                        last_graph = G_batch_prepared
                        last_output = output
                        last_targets = targets
                        last_loss = loss
                    else:
                        outputs.append(output)

            if is_train and self.scheduler is not None:
                self.scheduler.step()

        if return_loss:
            self._cleanup({
                "graph": last_graph,
                "output": last_output,
                "targets": last_targets,
                "loss": last_loss,
            })
            return total_loss / num_batches
        else:
            return torch.cat(outputs, dim=0)

    def _init_dataloader(self, dataset:Union[TorchDataset, NNDataset], **kwargs):
        """ Initialize a DataLoader for the given dataset.
        Args:
            dataset (Dataset): The dataset to load.
            is_node (bool): Whether the dataset contains node indices (default: False).
            **kwargs: Additional arguments for DataLoader.
        Returns:
            DataLoader: Configured DataLoader instance.
        """
        default_pin = self.device.type == "cuda" and torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=kwargs.get(f"batch_size", 15),
            shuffle=kwargs.get("shuffle", True),
            num_workers=kwargs.get("num_workers", 0),
            pin_memory = kwargs.get("pin_memory", default_pin),
        )

    def _init_subgraph_loader(self, batch_size: int = 256, input_nodes=None) -> ManualNeighborLoader:
        """
        Initialize the subgraph loader for batching.

        Args:
            batch_size (int): Size of each subgraph batch.

        Returns:
            ManualNeighborLoader: Configured subgraph loader.
        """
        return ManualNeighborLoader(
            device=self.device,
            base_graph=self.graph,
            num_hops=self.num_msg_passing_layers,
            input_nodes=input_nodes,  # or a specific mask if needed
            batch_size=batch_size,
            shuffle=True
        )

    def _validate_shapes(self, X):
        try:
            self.validator.validate(X)
        except Exception as e:
            raise ValueError(f"Invalid dataset for {self.__class__.__name__}: {e}") from e
    
    @cr('GNS._cleanup')
    @staticmethod
    def _cleanup(tensors: Union[Tensor, Dict, None, Tuple, List]) -> None:
        if isinstance(tensors, (tuple, list)):
            for t in tensors:
                del t
        else:
            del tensors
        torch.cuda.empty_cache()



    def save(self, path: str) -> None:
        """
        Save the model to a checkpoint file.

        If the given path is a directory, appends a filename with the number of trained epochs.

        Args:
            path (str): Directory or full file path to save the model.
        """
        checkpoint = {
            **self.model_dict,
            "state_dict": self.state_dict(),
            "state": self.state,
            "graph": self.graph,
        }

        if os.path.isdir(path):
            filename = f"/trained_model_{len(self.state[2]):06d}.pth"
            path = os.path.join(path, filename)

        torch.save(checkpoint, path)


    @classmethod
    def load(cls, path: str, device: Union[str, torch.device] = DEVICE) -> "GNS":
        """
        Load a model from a checkpoint file.

        Args:
            path (str): Path to the `.pth` checkpoint file.
            device (Union[str, torch.device]): Target device for model (default: inferred).

        Returns:
            GNS: Loaded model instance.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file '{path}' not found.")
        if not path.endswith(".pth"):
            raise ValueError("Model file must have a '.pth' extension.")

        device = torch.device(device or DEVICE)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. Use CPU instead.")
            torch.cuda.set_device(0)

        checkpoint = torch.load(path, map_location='cpu')

        required_keys = ["graph", "state_dict"]
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Missing key '{key}' in checkpoint.")
        
        model = cls(
            graph=checkpoint["graph"],
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            output_dim=checkpoint["output_dim"],
            hidden_size=checkpoint["hidden_size"],
            num_msg_passing_layers=checkpoint["num_msg_passing_layers"],
            encoder_hidden_layers=checkpoint["encoder_hidden_layers"],
            decoder_hidden_layers=checkpoint["decoder_hidden_layers"],
            message_hidden_layers=checkpoint["message_hidden_layers"],
            update_hidden_layers=checkpoint["update_hidden_layers"],
            p_dropouts=checkpoint.get("p_dropouts"),
            activation=checkpoint.get("activation"),
            seed=checkpoint.get("seed"),
            device=device,
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.state = checkpoint["state"]
        model.eval()

        return model

    @classmethod
    def create_optimized_model(
    cls,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    optuna_optimizer: OptunaOptimizer,
) -> Tuple["GNS", Dict]:

        r"""
        Create an optimized model using Optuna. The model is trained on the training dataset and evaluated on the validation dataset.
        
        Args:
            train_dataset (Dataset): The training dataset.
            eval_dataset (Dataset): The evaluation dataset.
            optuna_optimizer (OptunaOptimizer): The Optuna optimizer object.
        Returns:
            Tuple[nn.Module, Dict]: The optimized model and the optimization parameters.

        Example:
            >>> from pyLOM.NN import GNS, OptunaOptimizer
            >>> # Split the dataset
            >>> train_dataset, eval_dataset = dataset.get_splits([0.8, 0.2])
            >>> # Create the graph
            >>> graph = Graph.from_mesh(mesh, x=x, y=y, scaler=scaler)
            >>> 
            >>> # Define the optimization parameters
            >>> optimization_params = {
            >>>     "graph": graph,
            >>>     "input_dim": 2,
            >>>     "latent_dim": 16,
            >>>     "output_dim": 1,
            >>>     "hidden_size": (64, 512),
            >>>     "num_msg_passing_layers": (1, 10),
            >>>     "encoder_hidden_layers": (1, 10),
            >>>     "decoder_hidden_layers": (1, 10),
            >>>     "message_hidden_layers": (1, 10),
            >>>     "update_hidden_layers": (1, 10),
            >>>     "activation": "ELU",
            >>>     "epochs": 1000,
            >>>     "lr": (1e-5, 1e-2),
            >>>     "lr_gamma": (0.1, 1),
            >>>     "lr_scheduler_step": (1, 10),
            >>>     "loss_fn": nn.MSELoss(reduction='mean'),
            >>>     "optimizer": torch.optim.Adam,
            >>>     "scheduler": torch.optim.lr_scheduler.StepLR,
            >>>     "batch_size": (1, 16),
            >>>     "node_batch_size": (2**3, 2**16),
            >>> }
            >>>
            >>> # Define the optimizer
            >>> optimizer = OptunaOptimizer(
            >>>     optimization_params=optimization_params,
            >>>     n_trials=100,
            >>>     direction="minimize",
            >>>     pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=5),
            >>>     save_dir=None,
            >>> )
            >>>
            >>> # Create the optimized model
            >>> model, optimization_params = GNS.create_optimized_model(train_dataset, eval_dataset, optimizer)
            >>> 
            >>> # Fit the model
            >>> model.fit(train_dataset, eval_dataset, **optimization_params)
        """
        optimization_params = optuna_optimizer.optimization_params
        seed_base = optimization_params.get("seed", 42)

        MODEL_KWARGS = {
            'activation', 'p_dropouts', 'device'
        }
        TRAINING_KWARGS = {
            'batch_size', 'node_batch_size', 'num_workers', 'pin_memory',
            'epochs', 'lr', 'lr_gamma', 'lr_scheduler_step',
            'loss_fn', 'optimizer', 'scheduler',
            'print_rate_epoch', 'print_rate_batch'
        }
        
        @cr('GNS.optimization_function')
        def optimization_function(trial) -> float:
            hyperparams = {
                key: cls._get_optimizing_value(key, val, trial)
                for key, val in optimization_params.items()
            }

            model_kwargs = {k: v for k, v in hyperparams.items() if k in MODEL_KWARGS}
            fit_kwargs = {k: v for k, v in hyperparams.items() if k in TRAINING_KWARGS}

            model = cls(
                graph=hyperparams["graph"],
                input_dim=hyperparams["input_dim"],
                latent_dim=hyperparams["latent_dim"],
                output_dim=hyperparams["output_dim"],
                hidden_size=hyperparams["hidden_size"],
                num_msg_passing_layers=hyperparams["num_msg_passing_layers"],
                encoder_hidden_layers=hyperparams["encoder_hidden_layers"],
                decoder_hidden_layers=hyperparams["decoder_hidden_layers"],
                message_hidden_layers=hyperparams["message_hidden_layers"],
                update_hidden_layers=hyperparams["update_hidden_layers"],
                seed=seed_base + trial.number,
                **model_kwargs
            )

            pprint(0, f"\nTrial {trial.number + 1}/{optuna_optimizer.num_trials}. Training with hyperparams:\n",
                json.dumps(hyperparams, indent=4, default=cls._hyperparams_serializer), flush=True)

            try:
                if optuna_optimizer.pruner is not None:
                    original_epochs = fit_kwargs.get("epochs", 100)
                    fit_kwargs_loop = dict(fit_kwargs)
                    fit_kwargs_loop.pop("epochs", None)

                    for epoch in range(original_epochs):
                        losses = model.fit(train_dataset, eval_dataset, epochs=1, **fit_kwargs_loop)
                        loss_val = losses["test_loss"][-1]
                        trial.report(loss_val, epoch)
                        pprint(0, f"Epoch {epoch + 1}/{original_epochs}", flush=True)
                        if trial.should_prune():
                            model._cleanup()
                            del model
                            pprint(0, f"\nTrial pruned at epoch {epoch + 1}", flush=True)
                            raise TrialPruned()
                else:
                    losses = model.fit(train_dataset, eval_dataset, **fit_kwargs)
                    loss_val = losses["test_loss"][-1]
                    trial.report(loss_val, 0)


            except RuntimeError as e:
                model._cleanup()
                del model
                raise e

            model._cleanup()
            del model
            return loss_val

        best_params = optuna_optimizer.optimize(objective_function=optimization_function)
        optimization_params.update(best_params)

        final_model = cls(
            input_dim=best_params["input_dim"],
            latent_dim=best_params["latent_dim"],
            output_dim=best_params["output_dim"],
            hidden_size=best_params["hidden_size"],
            num_msg_passing_layers=best_params["num_msg_passing_layers"],
            encoder_hidden_layers=best_params["encoder_hidden_layers"],
            decoder_hidden_layers=best_params["decoder_hidden_layers"],
            message_hidden_layers=best_params["message_hidden_layers"],
            update_hidden_layers=best_params["update_hidden_layers"],
            **{k: best_params[k] for k in MODEL_KWARGS if k in best_params}
        )

        return final_model, optimization_params

    @cr('GNS._get_optimizing_value')
    @staticmethod
    def _get_optimizing_value(name, value, trial) -> Union[int, float, str]:
        """
        Suggest a value for a given hyperparameter depending on its type and content.

        Args:
            name (str): Hyperparameter name.
            value (Any): Either a fixed value, a range, or a list of options.
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Union[int, float, str]: Suggested value.
        """
        if isinstance(value, (tuple, list)):
            if all(isinstance(v, (int, float)) for v in value):
                use_log = np.abs(value[1]) / (np.abs(value[0]) + 1e-8) >= 1000
                low, high = value

                if isinstance(low, int):
                    if name == "latent_dim":
                        return trial.suggest_int(name, low + low % 2, high + high % 2, step=2)
                    return trial.suggest_int(name, low, high, log=use_log)

                elif isinstance(low, float):
                    if use_log and low <= 0:
                        use_log = False  # fallback to linear scale
                    return trial.suggest_float(name, low, high, log=use_log)
            elif all(isinstance(v, str) for v in value):
                return trial.suggest_categorical(name, value)
            else:
                raise ValueError(f"Unsupported value list for {name}: {value}")
        else:
            return value



    @staticmethod
    def _hyperparams_serializer(obj) -> str:
        r"""
        Function used to print hyperparams in JSON format.
        Args:
            obj (Any): The object to serialize.
        Returns:
            str: The serialized object.
        """

        if hasattr(obj, "__class__"):  # Verify whether the object has a class
            return obj.__class__.__name__  # Return the class name
        raise TypeError(f"Type {type(obj)} not serializable")  # Raise an error if the object is not serializable


    def __repr__(self):
        return (
            f"<GNSModel: {self.input_dim} → {self.latent_dim} → {self.output_dim}>\n"
            f" Layers: encoder({self.encoder_hidden_layers}), message({self.num_msg_passing_layers}), decoder({self.decoder_hidden_layers})\n"
            f" MLPs: message({self.message_hidden_layers}), update({self.update_hidden_layers})\n"
            f" Activation: {self.activation.__class__.__name__}, Dropout: {self.p_dropouts}, Device: {self.device}\n"
            f" Graph: {repr(self.graph)}\n"
            f" Params: {count_trainable_params(self):,} trainable\n>"
        )
