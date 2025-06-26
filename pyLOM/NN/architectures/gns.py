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

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import ELU
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph

from ..utils import Graph, VectorizedBatcher, ListBasedBatcher
from .. import DEVICE, set_seed
from ... import pprint, cr
from ..optimizer import OptunaOptimizer, TrialPruned

from typing import Dict, Tuple, Union, Optional

class MLP(torch.nn.Module):
    r'''Simple MLP with dropout and activation function
    Args:
        input_size (int): Input size.
        output_size (int): Output size.
        hidden_sizes (list): List of hidden sizes.
        drop_p (float): Dropout probability.
        activation (torch.nn.Module): Activation function.
    '''
    def __init__(self, input_size, output_size, hidden_sizes, drop_p=0.5, activation=ELU()):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
        self.activation = activation
        self.dropout = torch.nn.Dropout(p=drop_p)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
    
    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MessagePassingLayer(MessagePassing):
    r'''Message passing layer for the GNN. Uses mean aggregation and MLPs for message and update functions.
    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        hiddim (int): Hidden dimension.
        drop_p (float): Dropout probability.
    '''
    def __init__(self, in_channels, out_channels, hiddim, activation=ELU(), drop_p=0.):
        # Message passing with "mean" aggregation.
        super().__init__(aggr='mean')
        self.dropout = torch.nn.Dropout(p=drop_p)

        # MLP for the message function
        self.phi = MLP(in_channels, out_channels, 1*[hiddim], drop_p=0, activation=activation)
        
        # MLP for the update function
        self.gamma = MLP(2*out_channels, out_channels, 1*[hiddim], drop_p=0, activation=activation)


    def forward(self, x, edge_index, edge_features):
        # Start propagating messages.
        return self.propagate(edge_index, x=x, edge_features=edge_features)

    def message(self, x_i, x_j, edge_features):
        # x_i defines the features of central nodes as shape [num_edges, in_channels-6]
        # x_j defines the features of neighboring nodes as shape [num_edges, in_channels-6]
        # edge_features defines the attributes of intersecting edges as shape [num_edges, 6]

        input = torch.cat([x_i, x_j, edge_features], dim=1)

        return self.phi(input)  # Apply MLP phi
    
    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]

        input = torch.cat([x, aggr_out], dim=1)

        # Apply MLP gamma
        return self.gamma(input)
    
    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)





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
        graph (Union[torch_geometric.data.Data, Graph]): The graph object with node and edge attributes.
        p_dropouts (float, optional): The dropout probability. Default is ``0``.
        checkpoint_file (str, optional): The path to the checkpoint file. Default is ``None``.
        activation (Union[str, nn.Module]): The activation function to use.
        device (Union[str, torch.device]): The device to use for training. Default is ``'cuda'`` if available, otherwise ``'cpu'``.
        seed (int): The random seed for reproducibility. Default is None.
    """

    def __init__(self,
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
        graph = kwargs.get("graph")
        p_dropouts = kwargs.get("p_dropouts", 0.0)
        activation = kwargs.get("activation", 'ELU')
        device = kwargs.get("device", DEVICE)
        seed = kwargs.get("seed", None)

        # --- Device setup ---
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available. Please use CPU instead.")
            torch.cuda.set_device(0)

        self.device = device
        self.seed = seed
        if seed is not None:
            set_seed(seed)
        print(f"Using device: {self.device}", flush=True)

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
        self.p_dropouts = p_dropouts

        # Will be filled when graph is set
        self.encoder_input_dim = None
        self._edge_dim = None
        self._graph = None

        # Optimizer state
        self.state = {}
        self.optimizer = None
        self.scheduler = None
        self.checkpoint = None

        # Activation
        if isinstance(activation, str):
            if not hasattr(nn, activation):
                raise ValueError(f"Activation function '{activation}' not found in torch.nn")
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation

        # Set graph (also sets encoder_input_dim and _edge_dim)
        if graph is None:
            raise Warning("Graph not provided. A graph object must be set with cls.graph setter before training.")
        self.graph = graph

        # Save config dictionary
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
            "p_dropouts": p_dropouts,
            "activation": activation,
            "seed": seed
        }

        # --- Network Components ---
        self.batcher = ListBasedBatcher(
            graph=self.graph,
            num_hops=self.num_msg_passing_layers,
            input_dim=self.input_dim,
            device=self.device,
        )

        self.encoder = MLP(
            input_size=self.encoder_input_dim,
            output_size=self.latent_dim,
            hidden_sizes=[self.hidden_size] * self.encoder_hidden_layers,
            activation=self.activation,
            drop_p=self.p_dropouts
        ).to(self.device)

        self.decoder = MLP(
            input_size=self.latent_dim,
            output_size=self.output_dim,
            hidden_sizes=[self.hidden_size] * self.decoder_hidden_layers,
            activation=self.activation,
            drop_p=self.p_dropouts
        ).to(self.device)

        self.conv_layers_list = nn.ModuleList([
            MessagePassingLayer(
                in_channels=2 * self.latent_dim + self._edge_dim,
                out_channels=self.latent_dim,
                hiddim=self.hidden_size,
                activation=self.activation,
                drop_p=self.p_dropouts,
            ) for _ in range(self.num_msg_passing_layers)
        ]).to(self.device)

        self.groupnorm = nn.GroupNorm(2, self.latent_dim).to(self.device)


    @property
    def graph(self) -> Graph:
        r'''Graph property to get the graph object.'''
        return self._graph
    
    @graph.setter
    def graph(self, graph: Graph) -> None:
        r'''Graph property to set the graph object.'''
        if self._graph is not None:
            raise Warning("Graph is already set! Graph name is: {}".format(self._graph.name))
        if not isinstance(graph, Graph):
            if not isinstance(graph, Data):
                raise TypeError("Graph must be of type torch_geometric.data.Data or Graph.")
        if getattr(graph, "edge_index", None) is None:
            raise ValueError("Graph must have edge_index attribute.")
        if getattr(graph, "edge_features", None) is None:
            raise ValueError("Graph must have edge_features attribute.")
        
        graph = graph.to(self.device)

        self.encoder_input_dim = graph.node_features.shape[1] # Update node features dimension
        self._edge_dim = graph.edge_features.shape[1] # Update edge features dimension

        self._graph = graph

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward(
        self,
        inputs: Optional[Tensor] = None,
        graph: Optional[Union[Graph, Data]] = None
    ) -> Tensor:
        r"""
        Perform a forward pass through the network.

        During training, a pre-batched graph (e.g., via DataLoader) is passed directly.
        During inference, a tensor of operational parameters is provided and used along with
        the stored full graph (`self.graph`).

        Args:
            inputs (torch.Tensor, optional): Operational parameters of shape [B, input_dim].
                Required for inference.
            graph (torch_geometric.data.Data, optional): Input graph with node/edge features.
                Required for training.

        Returns:
            torch.Tensor: Predicted values for the nodes or batch of nodes.
        """
        if graph is not None:
            return self._forward_from_graph(graph)
        elif inputs is not None:
            return self._forward_from_tensor(inputs)
        else:
            raise ValueError("Either `graph` or `inputs` must be provided.")

    def _forward_from_graph(self, graph: Union[Graph, Data]) -> Tensor:
        """Forward pass given a graph with operational parameters already embedded."""
        x, edge_index, edge_features = graph.node_features, graph.edge_index, graph.edge_features

        h = self.activation(self.encoder(x))
        for conv in self.conv_layers_list:
            h = self.groupnorm(self.activation(conv(h, edge_index, edge_features)))
        y_hat = self.decoder(h)
        return y_hat

    def _forward_from_tensor(self, inputs: Tensor) -> Tensor:
        """
        Vectorized forward pass over the full graph for a batch of operational parameters.
        
        Args:
            inputs (Tensor): Tensor of shape [B, input_dim] representing B operational conditions.
            
        Returns:
            Tensor: Predicted outputs of shape [B, N, output_dim], where N is the number of nodes.
            
        Note:
            This method assumes a stored full graph (`self.graph`) and performs inference over the entire mesh.
            It is not used for subgraph-based training.
        """
        if self.graph is None:
            raise ValueError("Stored graph (`self.graph`) is required for inference.")
        
        if inputs.size(1) != self.input_dim:
            raise ValueError(f"Expected inputs of shape [B, {self.input_dim}], got {inputs.shape}")

        node_features = self.graph.node_features
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # Shape [1, D]

        B = inputs.size(0)
        N = self.graph.node_features.size(0)

        op_repeated = inputs.repeat_interleave(N, dim=0)
        x_repeated = self.graph.node_features.repeat(B, 1)
        x_repeated[:, :self.input_dim] = op_repeated

        edge_index = torch.cat([
            self.graph.edge_index + i * N for i in range(B)
        ], dim=1)
        edge_features = self.graph.edge_features.repeat(B, 1)

        h = self.activation(self.encoder(x_repeated))
        for conv in self.conv_layers_list:
            h = self.groupnorm(self.activation(conv(h, edge_index, edge_features)))
        y_hat = self.decoder(h)
        return y_hat.reshape(B, N, -1)


    def fit(self,
            train_dataset,
            eval_dataset=None,
            **kwargs
            ) -> Dict:
        r"""
        Fit the model to the training data.
        """

        epochs = kwargs.get("epochs", 100)
        lr = kwargs.get("lr", 1e-4)
        lr_gamma = kwargs.get("lr_gamma", 0.1)
        lr_scheduler_step = kwargs.get("lr_scheduler_step", 1)
        loss_fn = kwargs.get("loss_fn", nn.MSELoss(reduction='mean'))
        optimizer = kwargs.get("optimizer", torch.optim.Adam)
        scheduler = kwargs.get("scheduler", torch.optim.lr_scheduler.StepLR)
        print_rate_epoch = kwargs.get("print_rate_epoch", 1)

        op_dataloader = self._init_dataloader(train_dataset, is_node=False, **kwargs)
        node_indices = torch.arange(self.graph.num_nodes, dtype=torch.long)
        node_dataloader = self._init_dataloader(node_indices, is_node=True, **kwargs)
        eval_dataloader = self._init_dataloader(eval_dataset, is_node=False, **kwargs) if eval_dataset is not None else None

        if self.optimizer is None:
            self.optimizer = optimizer(self.parameters(), lr=lr)
        if self.scheduler is None:
            self.scheduler = scheduler(self.optimizer, step_size=lr_scheduler_step, gamma=lr_gamma) if scheduler is not None else None

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

        total_epochs = len(epoch_list) + epochs
        for epoch in range(1 + len(epoch_list), 1 + total_epochs):

            train_loss = self._train(op_dataloader, node_dataloader, loss_fn)
            train_loss_list.append(train_loss)

            if eval_dataloader is not None:
                test_loss = self._eval(eval_dataloader, loss_fn)
                test_loss_list.append(test_loss)

            if print_rate_epoch and epoch % print_rate_epoch == 0:
                test_log = f" | Test loss:{test_loss:.4e}" if eval_dataloader else ""
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

    def predict(self, X, return_targets: bool = False, **kwargs) -> Tuple[np.ndarray, ...]:
        dataloader = self._init_dataloader(X, is_node=False, **kwargs)
        num_rows = len(dataloader.dataset)
        num_columns = self.graph.num_nodes * self.output_dim
        all_predictions = np.zeros((num_rows, num_columns))
        all_targets = np.zeros((num_rows, num_columns))

        with torch.no_grad():
            self.eval()
            i = 0
            for params_batch, y_batch in dataloader:
                params_batch, y_batch = self._to_device(params_batch, y_batch)
                for params, y in zip(params_batch, y_batch):
                    targets = y.reshape(-1, self.output_dim)
                    output = self(params)
                    all_predictions[i] = output.cpu().numpy().reshape(-1)
                    all_targets[i] = targets.cpu().numpy().reshape(-1)
                    i += 1

        return (all_predictions, all_targets) if return_targets else all_predictions

    def _train(self, op_dataloader, node_dataloader, loss_fn) -> float:
        if self.seed is not None:
            set_seed(self.seed)

        self.train()
        total_loss = 0.0

        for params_batch, y_batch in op_dataloader:
            params_batch, y_batch = self._to_device(params_batch, y_batch)

            for seed_nodes in node_dataloader:
                seed_nodes, = self._to_device(seed_nodes)

                G_batch = self.batcher.create_batch(params_batch, y_batch, seed_nodes)

                self.optimizer.zero_grad()
                output = self(graph=G_batch)[G_batch.seed_nodes]
                targets = G_batch.y[G_batch.seed_nodes]
                loss = loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                self._cleanup(G_batch, output, targets, loss)

            if self.scheduler is not None:
                self.scheduler.step()

        return total_loss / float(len(op_dataloader) * len(node_dataloader))

    def _eval(self, eval_dataloader, loss_fn) -> float:
        self.eval()
        total_loss = 0.0

        with torch.no_grad():
            for params_batch, y_batch in eval_dataloader:
                params_batch, y_batch = self._to_device(params_batch, y_batch)
                for params, y in zip(params_batch, y_batch):
                    targets = y.reshape(-1, self.output_dim)
                    output = self(params)
                    loss = loss_fn(output, targets)
                    total_loss += loss.item()

        return total_loss / float(len(eval_dataloader.dataset))

    def _to_device(self, *tensors):
        return [t.to(self.device) for t in tensors]

    def _init_dataloader(self, dataset, is_node=False, **kwargs):
        key_prefix = "node_" if is_node else ""
        return DataLoader(
            dataset,
            batch_size=kwargs.get(f"{key_prefix}batch_size", 256 if is_node else 15),
            shuffle=kwargs.get("shuffle", True),
            num_workers=kwargs.get("num_workers", 0),
            pin_memory=kwargs.get("pin_memory", True),
        )


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
            path = path + filename

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
    @cr('MLP.create_optimized_model')
    def create_optimized_model(
        cls,
        train_dataset, 
        eval_dataset, 
        optuna_optimizer: OptunaOptimizer,
        ) -> Tuple[nn.Module, Dict]:
        r"""
        Create an optimized model using Optuna. The model is trained on the training dataset and evaluated on the validation dataset.
        
        Args:
            graph (Graph): The graph object used to train the GNN.
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
            >>> model, optimization_params = MLP.create_optimized_model(train_dataset, eval_dataset, optimizer)
            >>> 
            >>> # Fit the model
            >>> model.fit(train_dataset, eval_dataset, **optimization_params)
        """
        optimization_params = optuna_optimizer.optimization_params
        def optimization_function(trial) -> float:
            hyperparams = {}       
            for key, params in optimization_params.items():
                hyperparams[key] = cls._get_optimizing_value(key, params, trial)

            model_kwargs = {key: value for (key, value) in hyperparams.items() if key in ['graph', 'activation', 'p_dropouts', 'device', 'seed']}
            training_kwargs = {key: value for (key, value) in hyperparams.items() if key in ['batch_size', 'node_batch_size', 'num_workers', 'pin_memory']}

            
            model = cls(
                input_dim=hyperparams["input_dim"],
                latent_dim=hyperparams["latent_dim"],
                output_dim=hyperparams["output_dim"],
                hidden_size=hyperparams["hidden_size"],
                num_msg_passing_layers=hyperparams["num_msg_passing_layers"],
                encoder_hidden_layers=hyperparams["encoder_hidden_layers"],
                decoder_hidden_layers=hyperparams["decoder_hidden_layers"],
                message_hidden_layers=hyperparams["message_hidden_layers"],
                update_hidden_layers=hyperparams["update_hidden_layers"],
                **model_kwargs
            )

            print(f"\n\n\nTrial {trial._trial_id +1}/{optuna_optimizer.num_trials}. Training with hyperparams:\n",
                  json.dumps(hyperparams, indent=4, default = cls._hyperparams_serializer), flush=True)
            if optuna_optimizer.pruner is not None:
                # prune epoch-wise
                epochs = hyperparams["epochs"]
                hyperparams["epochs"] = 1
                for epoch in range(epochs):
                    losses = model.fit(train_dataset, eval_dataset, **hyperparams)
                    loss_val = losses["test_loss"][-1]
                    # Report the loss to Optuna
                    trial.report(loss_val, epoch)
                    print("Epoch {}/{}".format(epoch+1, epochs), flush=True)
                    if trial.should_prune():
                        print("\nTrial pruned at epoch {}".format(epoch+1), flush=True)
                        raise TrialPruned()
            else:
                losses = model.fit(
                    train_dataset,
                    epochs = hyperparams.get("epochs"),
                    lr = hyperparams.get("lr"),
                    lr_gamma = hyperparams.get("lr_gamma"),
                    lr_scheduler_step = hyperparams.get("lr_scheduler_step"),
                    loss_fn = hyperparams.get("loss_fn"),
                    optimizer = hyperparams.get("optimizer"),
                    scheduler = hyperparams.get("scheduler"),
                    print_rate_batch = hyperparams.get("print_rate_batch"),
                    print_rate_epoch = hyperparams.get("print_rate_epoch"),
                    **training_kwargs
                    )
                loss_val = losses["test_loss"][-1]
                # Report the loss to Optuna
                trial.report(loss_val, 0)
            
            return loss_val
        
        best_params = optuna_optimizer.optimize(objective_function=optimization_function)

        best_model_kwargs = {key: value for key, value in optimization_params.items() if key in ['graph', 'activation', 'p_dropouts', 'device', 'seed']}

        # Update params with best ones
        for param in best_params.keys():
            if param in optimization_params:
                optimization_params[param] = best_params[param]
            if param in best_model_kwargs:
                best_model_kwargs[param] = best_params[param]

        
        return cls(
            input_dim=optimization_params["input_dim"],
            latent_dim=optimization_params["latent_dim"],
            output_dim=optimization_params["output_dim"],
            hidden_size=optimization_params["hidden_size"],
            num_msg_passing_layers=optimization_params["num_msg_passing_layers"],
            encoder_hidden_layers=optimization_params["encoder_hidden_layers"],
            decoder_hidden_layers=optimization_params["decoder_hidden_layers"],
            message_hidden_layers=optimization_params["message_hidden_layers"],
            update_hidden_layers=optimization_params["update_hidden_layers"],
            **best_model_kwargs
        ), optimization_params

    @staticmethod
    def _get_optimizing_value(name, value, trial) -> Union[int, float, str]:
        r"""
        Function used to get the optimizing value for a given hyperparameter.
        Args:
            name (str): The name of the hyperparameter.
            value (Any): The value of the hyperparameter.
            trial (optuna.Trial): The Optuna trial object.
        Returns:
            Any: The optimizing value for the hyperparameter.
        """

        if isinstance(value, tuple) or isinstance(value, list):
            use_log = value[1] / (value[0] + 0.1) >= 1000
            if isinstance(value[0], int):
                if name == 'latent_dim':
                    return trial.suggest_int(name, value[0]+value[0]%2, value[1]+value[1]%2, step=2)
                else:
                    return trial.suggest_int(name, value[0], value[1], log=use_log)
            elif isinstance(value[0], float):
                return trial.suggest_float(name, value[0], value[1], log=use_log)
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

    @staticmethod
    def _cleanup(*tensors) -> None:
        r"""
        Cleanup the GPU memory by deleting the tensors and emptying the cache.
        Args:
            tensors (tuple): The tensors to delete.
        """

        for t in tensors:
            del t
        torch.cuda.empty_cache()
