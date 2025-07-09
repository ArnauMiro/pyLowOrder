#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 30/03/2025

import os
import json
import warnings
from dataclasses import asdict
from typing import Dict, Tuple, Union

import torch
from torch import Tensor
from torch.nn import ELU
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset
from torch_geometric.data import Data

from .. import Dataset as NNDataset
from .. import DEVICE, set_seed
from ..utils import count_trainable_params, cleanup_tensors, get_optimizing_value, hyperparams_serializer
from ..gns import GNSMLP, MessagePassingLayer, Graph, InputsInjector, _ShapeValidator, _GNSHelpers, GNSConfig, TrainingConfig
from ..optimizer import OptunaOptimizer, TrialPruned
from ... import pprint, cr





class GNS(torch.nn.Module):
    r"""
    Graph Neural Solver class for predicting aerodynamic variables on RANS meshes.

    This model uses a message-passing GNN architecture with MLPs for the message and update functions.
    It supports subgraph batching and is optimized for training on large RANS meshes.

    Args:
        graph (Graph): The input graph object containing node and edge features.
        config (GNSConfig): Configuration dataclass with all model hyperparameters, including architecture,
            activation, dropout, device, and random seed.

    Example:
        >>> graph = Graph.from_mesh(...)
        >>> config = GNSConfig(
        >>>     input_dim=2,
        >>>     latent_dim=16,
        >>>     output_dim=1,
        >>>     hidden_size=128,
        >>>     num_msg_passing_layers=4,
        >>>     encoder_hidden_layers=2,
        >>>     decoder_hidden_layers=2,
        >>>     message_hidden_layers=2,
        >>>     update_hidden_layers=2,
        >>>     activation="ELU",
        >>>     device="cuda",
        >>> )
        >>> model = GNS(graph, config)
    """

    def __init__(self, graph: Graph, config: GNSConfig) -> None:
        """
        Initialize the GNS model.

        Args:
            graph (Graph): The graph containing node and edge features.
            config (GNSConfig): The configuration dataclass containing all model hyperparameters.
        """
        super().__init__()

        # --- Store config and extract common parameters ---
        self.config = config
        self.device = torch.device(config.device)
        self.seed = config.seed
        self.p_dropouts = config.p_dropouts

        # --- Activation function ---
        if isinstance(config.activation, str):
            if not hasattr(torch.nn, config.activation):
                raise ValueError(f"Activation function '{config.activation}' not found in torch.nn")
            self.activation = getattr(torch.nn, config.activation)()
        else:
            self.activation = config.activation

        # --- Device setup ---
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available. Please use CPU instead.")
            torch.cuda.set_device(self.device)  # Safer than hardcoding device index
        pprint(0, f"Using device: {self.device}", flush=True)

        # --- Reproducibility ---
        if self.seed is not None:
            set_seed(self.seed)

        # --- Optional convenience aliases ---
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        self.output_dim = config.output_dim
        self.hidden_size = config.hidden_size
        self.num_msg_passing_layers = config.num_msg_passing_layers
        self.encoder_hidden_layers = config.encoder_hidden_layers
        self.decoder_hidden_layers = config.decoder_hidden_layers
        self.message_hidden_layers = config.message_hidden_layers
        self.update_hidden_layers = config.update_hidden_layers

        # --- Graph setup ---
        self._graph = None
        self._encoder_input_dim = None
        self._edge_dim = None
        self.graph = graph  # Triggers setter logic: sets _encoder_input_dim and _edge_dim

        # --- Inputs injector (subgraph preparation) ---
        self.injector = InputsInjector(device=self.device)

        # --- Training state (can be loaded from checkpoint) ---
        self.state = {}
        self.optimizer = None
        self.scheduler = None
        self.checkpoint = None

        # --- Shape validation helper ---
        self.validator = _ShapeValidator(self.input_dim, self.output_dim, self.graph.num_nodes)

        # --- Subgraph batching and utility helpers ---
        self._helpers = _GNSHelpers(
            device=self.device,
            graph=self.graph,
            num_msg_passing_layers=self.num_msg_passing_layers
        )

        # --- Model components ---
        self._build_encoder()
        self._build_message_passing_layers()
        self._build_decoder()

        # --- Normalization layer ---
        self.groupnorm = torch.nn.GroupNorm(
            num_groups=min(2, self.latent_dim),
            num_channels=self.latent_dim
        ).to(self.device)

    @property
    def model_dict(self) -> Dict:
        return asdict(self.config)

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
        self.conv_layers_list = torch.nn.ModuleList([
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
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr

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
            X (Tensor or Dataset): Operational inputs. If Tensor, shape [B, D].
            batch_size (int): Used only if X is a Dataset.
            node_batch_size (int): Number of seed nodes per subgraph batch.
        Returns:
            Tensor: Predictions of shape [B * N, F].
        """
        self._validate_shapes(X)
        self.eval()

        with torch.no_grad():
            input_dataloader = self._helpers.init_dataloader(X, batch_size=batch_size)
            subgraph_loader = self._helpers.init_subgraph_loader(
                batch_size=node_batch_size,
                input_nodes=kwargs.get("input_nodes", None)
            )
            return self._run_epoch(input_dataloader, subgraph_loader, is_train=False, return_loss=False)

    @cr('GNS.fit')
    def fit(
        self,
        train_dataset: TorchDataset,
        eval_dataset: TorchDataset = None,
        config: TrainingConfig = TrainingConfig()
    ) -> Dict[str, list]:
        """
        Train the model using subgraph batching over both the input parameter space and the node space.

        Args:
            train_dataset: Dataset of input parameters and targets. Each batch should yield:
                - inputs: Tensor of shape [B, D], where D = input_dim
                - targets: Tensor of shape [B, N, F], where N = num_nodes, F = output_dim
            eval_dataset: Optional validation dataset with the same format as train_dataset.
            config: TrainingConfig dataclass that holds all training hyperparameters.

        Returns:
            Dict: Dictionary containing lists of per-epoch `"train_loss"` and `"test_loss"`.
        """
        # --- Validate dataset shapes ---
        self._validate_shapes(train_dataset)
        if eval_dataset is not None:
            self._validate_shapes(eval_dataset)

        # --- Instantiate dataloaders ---
        input_dataloader = self._helpers.init_dataloader(train_dataset, batch_size=config.batch_size)
        eval_dataloader = (
            self._helpers.init_dataloader(eval_dataset, batch_size=config.batch_size)
            if eval_dataset is not None else None
        )
        subgraph_loader = self._helpers.init_subgraph_loader(
            batch_size=config.node_batch_size,
            input_nodes=config.input_nodes
        )

        # --- Initialize optimizer and scheduler if not set ---
        if self.optimizer is None:
            self.optimizer = config.optimizer(self.parameters(), lr=config.lr)

        if self.scheduler is None and config.scheduler is not None:
            self.scheduler = config.scheduler(
                self.optimizer,
                step_size=config.lr_scheduler_step,
                gamma=config.lr_gamma
            )

        # --- Load from checkpoint if available ---
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint["state"][0])
            if self.scheduler is not None and self.checkpoint["state"][1]:
                self.scheduler.load_state_dict(self.checkpoint["state"][1])
                self.scheduler.gamma = config.lr_gamma
                self.scheduler.step_size = config.lr_scheduler_step
            epoch_list = self.checkpoint["state"][2]
            train_loss_list = self.checkpoint["state"][3]
            test_loss_list = self.checkpoint["state"][4]
        else:
            epoch_list = []
            train_loss_list = []
            test_loss_list = []

        # --- Training loop ---
        total_epochs = len(epoch_list) + config.epochs
        for epoch in range(1 + len(epoch_list), 1 + total_epochs):
            train_loss = self._run_epoch(
                input_dataloader,
                subgraph_loader,
                return_loss=True,
                loss_fn=config.loss_fn,
                is_train=True
            )
            train_loss_list.append(train_loss)

            if eval_dataloader is not None:
                test_loss = self._run_epoch(
                    eval_dataloader,
                    subgraph_loader,
                    return_loss=True,
                    loss_fn=config.loss_fn,
                    is_train=False
                )
                test_loss_list.append(test_loss)

            # Determine if this epoch should be logged
            log_this_epoch = (
                config.verbose is True
                or (isinstance(config.verbose, int) and config.verbose > 0 and epoch % config.verbose == 0)
            )

            if log_this_epoch:
                test_log = f" | Eval loss: {test_loss:.4e}" if eval_dataloader else ""
                pprint(0, f"Epoch {epoch}/{total_epochs} | Train loss: {train_loss:.4e}{test_log}", flush=True)

                if self.device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 2
                    reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 2
                    pprint(0, f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB", flush=True)

            # Save state after each epoch
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
    def _run_epoch(
        self,
        input_dataloader,
        subgraph_loader,
        loss_fn: torch.nn.Module = None,
        return_loss: bool = False,
        is_train: bool = False
    ) -> Union[float, Tensor]:
        """
        Run one epoch over the dataset, either in training or evaluation mode.

        Args:
            input_dataloader (DataLoader): Yields batches of [inputs, targets].
            subgraph_loader (DataLoader): Yields seed nodes for subgraph sampling.
            loss_fn (callable, optional): Loss function to apply (only if return_loss=True).
            return_loss (bool): Whether to compute and return averaged loss.
            is_train (bool): Whether to perform gradient updates (training) or not (evaluation/prediction).

        Returns:
            Union[float, Tensor]:
                - If return_loss is True: average loss over all batches.
                - If return_loss is False: predictions of shape [B * N, F].
        """
        self.train() if is_train else self.eval()
        outputs = []
        total_loss = 0.0
        context = torch.enable_grad() if is_train else torch.no_grad()

        last_graph = None
        last_output = None
        last_targets = None
        last_loss = None
        num_batches = 0

        with context:
            for batch in input_dataloader:
                inputs_batch = batch[0].to(self.device)
                try:
                    targets_batch = batch[1].to(self.device)
                except IndexError:
                    targets_batch = None

                for subgraph in subgraph_loader:
                    if is_train:
                        loss_val, G, out, targets, loss = self._train_one_batch(
                            subgraph, inputs_batch, targets_batch, loss_fn
                        )
                    elif return_loss:
                        loss_val, G, out, targets, loss = self._eval_one_batch(
                            subgraph, inputs_batch, targets_batch, loss_fn
                        )
                    else:
                        out = self._eval_one_batch(subgraph, inputs_batch, targets_batch)
                        outputs.append(out)
                        continue  # Skip loss-related code

                    if return_loss:
                        total_loss += loss_val
                        num_batches += 1
                        last_graph = G
                        last_output = out
                        last_targets = targets
                        last_loss = loss

            if is_train and self.scheduler is not None:
                self.scheduler.step()

        if return_loss:
            cleanup_tensors({
                "graph": last_graph,
                "output": last_output,
                "targets": last_targets,
                "loss": last_loss,
            })
            return total_loss / num_batches
        else:
            return torch.cat(outputs, dim=0).reshape(-1, self.graph.num_nodes, self.output_dim)


    @cr('GNS._train_one_batch')
    def _train_one_batch(
        self,
        subgraph: Data,
        inputs_batch: Tensor,
        targets_batch: Tensor,
        loss_fn: torch.nn.Module
    ) -> Tuple[float, Data, Tensor, Tensor, Tensor]:
        """
        Execute a single training step over a subgraph batch.

        Args:
            subgraph (Data): Subgraph seed data from the loader.
            inputs_batch (Tensor): Input parameter batch of shape [B, D].
            targets_batch (Tensor): Ground truth tensor of shape [B, N, F].
            loss_fn (callable): Loss function to compute training loss.

        Returns:
            Tuple[float, Data, Tensor, Tensor, Tensor]:
                - Loss value for this batch.
                - The processed graph with injected inputs and targets.
                - Model outputs on the seed nodes.
                - Ground truth targets on the seed nodes.
                - Loss tensor.
        """
        G = self.injector.replicate_inject(subgraph, inputs_batch, targets_batch)

        self.optimizer.zero_grad()
        output = self.forward(G)[G.seed_mask]
        targets = G.y[G.seed_mask]

        assert output.shape == targets.shape, f"Output shape {output.shape} != target shape {targets.shape}"
        loss = loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item(), G, output, targets, loss

    @cr('GNS._eval_one_batch')
    def _eval_one_batch(
        self,
        subgraph: Data,
        inputs_batch: Tensor,
        targets_batch: Union[Tensor, None],
        loss_fn: torch.nn.Module = None
    ) -> float | Data | Tensor:
        """
        Perform a single evaluation (validation or prediction) step.

        Args:
            subgraph (Data): Subgraph seed data from the loader.
            inputs_batch (Tensor): Input parameter batch of shape [B, D].
            targets_batch (Tensor or None): Ground truth targets if available.
            loss_fn (callable, optional): If provided, computes loss against targets.

        Returns:
            - If loss_fn is provided:
                Tuple[float, Data, Tensor, Tensor, Tensor]: (loss value, graph, output, targets, loss tensor)
            - Else:
                Tensor: Predictions on seed nodes.
        """
        G = self.injector.replicate_inject(subgraph, inputs_batch, targets_batch)
        output = self.forward(G)[G.seed_mask]

        if loss_fn is not None:
            targets = G.y[G.seed_mask]
            assert output.shape == targets.shape, f"Output shape {output.shape} != target shape {targets.shape}"
            loss = loss_fn(output, targets)
            return loss.item(), G, output, targets, loss
        else:
            return output


    def _validate_shapes(self, X):
        try:
            self.validator.validate(X)
        except Exception as e:
            raise ValueError(f"Invalid dataset for {self.__class__.__name__}: {e}") from e


    def save(self, path: str) -> None:
        """
        Save the model to a checkpoint file.

        If the given path is a directory, appends a filename with the number of trained epochs.

        Args:
            path (str): Directory or full file path to save the model.
        """
        checkpoint = {
            "config": asdict(self.config),
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
        
        config = GNSConfig(**checkpoint["config"])
        model = cls(graph=checkpoint["graph"], config=config)
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
        """
        Create and train an optimized GNS model using Optuna hyperparameter search.

        Args:
            train_dataset (Dataset): Training dataset with input/target pairs.
            eval_dataset (Dataset): Validation dataset for early stopping and evaluation.
            optuna_optimizer (OptunaOptimizer): Wrapper around Optuna to manage the optimization process.
        
        Returns:
            Tuple[GNS, Dict]: A trained GNS model and the best hyperparameter set found.
        
        Notes:
            This method assumes that `optimization_params` inside `optuna_optimizer` includes keys compatible
            with `GNSConfig` and `TrainingConfig`, such as:
                - For GNSConfig: input_dim, latent_dim, ...
                - For TrainingConfig: epochs, lr, batch_size, ...
        """

        optimization_params = optuna_optimizer.optimization_params

        MODEL_KWARGS = {
            'graph', 'input_dim', 'latent_dim', 'output_dim',
            'hidden_size', 'num_msg_passing_layers',
            'encoder_hidden_layers', 'decoder_hidden_layers',
            'message_hidden_layers', 'update_hidden_layers',
            'p_dropouts', 'activation', 'seed', 'device',
        }
        TRAINING_KWARGS = {
            'epochs', 'lr', 'lr_gamma', 'lr_scheduler_step',
            'loss_fn', 'optimizer', 'scheduler',
            'batch_size', 'node_batch_size', 'num_workers',
            'input_nodes', 'verbose'
        }
        
        @cr('GNS.optimization_function')
        def optimization_function(trial) -> float:
            # Sample hyperparameters from Optuna
            hyperparams = {
                key: get_optimizing_value(key, val, trial)
                for key, val in optimization_params.items()
            }

            # Construct model config and training config
            model_config = GNSConfig(**{k: v for k, v in hyperparams.items() if k in MODEL_KWARGS})
            training_config = TrainingConfig(**{k: v for k, v in hyperparams.items() if k in TRAINING_KWARGS})

            # Create model
            model = cls(
                graph=optimization_params["graph"],  # graph stays outside config
                config=model_config
            )

            # Logging sampled hyperparameters
            pprint(0, f"\nTrial {trial.number + 1}/{optuna_optimizer.num_trials}. Training with hyperparams:\n",
                json.dumps(hyperparams, indent=4, default=hyperparams_serializer), flush=True)

            try:
                if optuna_optimizer.pruner is not None:
                    for epoch in range(training_config.epochs):
                        training_config_epoch = TrainingConfig(**{**asdict(training_config), "epochs": 1})
                        losses = model.fit(train_dataset, eval_dataset, config=training_config_epoch)
                        loss_val = losses["test_loss"][-1]
                        trial.report(loss_val, epoch)
                        if trial.should_prune():
                            torch.cuda.empty_cache()
                            del model
                            raise TrialPruned()
                else:
                    losses = model.fit(train_dataset, eval_dataset, config=training_config)
                    loss_val = losses["test_loss"][-1]
                    trial.report(loss_val, step=9999)

            except RuntimeError as e:
                torch.cuda.empty_cache()
                del model
                raise e

            torch.cuda.empty_cache()
            del model
            return loss_val

        best_params = optuna_optimizer.optimize(objective_function=optimization_function)
        pprint(0, f"\nBest hyperparameters found: {json.dumps(best_params, indent=4, default=hyperparams_serializer)}\n", flush=True)

        # Rebuild final model using best config
        model_config = GNSConfig(**{k: v for k, v in best_params.items() if k in MODEL_KWARGS})
        final_model = cls(
            graph=optimization_params["graph"],
            config=model_config
        )

        return final_model, best_params


    def __repr__(self):
        return (
            f"<GNSModel: {self.input_dim} → {self.latent_dim} → {self.output_dim}>\n"
            f" Layers: encoder({self.encoder_hidden_layers}), message({self.num_msg_passing_layers}), decoder({self.decoder_hidden_layers})\n"
            f" MLPs: message({self.message_hidden_layers}), update({self.update_hidden_layers})\n"
            f" Activation: {self.activation.__class__.__name__}, Dropout: {self.p_dropouts}, Device: {self.device}\n"
            f" Graph: {repr(self.graph)}\n"
            f" Params: {count_trainable_params(self):,} trainable\n>"
        )
