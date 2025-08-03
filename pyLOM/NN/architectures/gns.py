#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 30/03/2025

import os
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import datetime

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from optuna import Trial

from .. import DEVICE, set_seed
from ..utils import count_trainable_params, cleanup_tensors, get_optimizing_value, hyperparams_serializer
from ..gns import GNSMLP, MessagePassingLayer, Graph, InputsInjector, _ShapeValidator, _GNSHelpers, GNSModelParams, GNSTrainingParams
from ..utils.wrappers import accepts_config
from ..utils.config_serialization import serialize_config, deserialize_config
from ..optimizer import OptunaOptimizer, TrialPruned
from ... import pprint, cr
from ...utils import raiseError



class GNS(torch.nn.Module):
    r"""
    Graph Neural Solver class for predicting aerodynamic variables on RANS meshes.

    This model uses a message-passing GNN architecture with MLPs for the message and update functions.
    It supports subgraph batching and is optimized for training on large RANS meshes.

    Note:
        This constructor assumes a fully loaded `Graph` object and a validated `GNSModelParams` config.
        For standard usage, prefer using `GNS.from_graph(...)` or `GNS.from_graph_path(...)`.
    """

    def __init__(self, *, config: GNSModelParams, graph: Graph) -> None:
        """
        Internal constructor for GNS. Do not use directly unless you know what you're doing.

        Args:
            config (GNSModelParams): Fully validated model configuration.
            graph (Graph): In-memory Graph object required by all components of the model.
        """
        super().__init__()

        # --- Basic validation ---
        if not isinstance(config, GNSModelParams):
            raiseError("Expected config to be a GNSModelParams.")
        if not isinstance(graph, Graph):
            raiseError("Expected graph to be a Graph instance.")

        self.config = config
        self.device = torch.device(config.device)
        self.seed = config.seed
        self.p_dropouts = config.p_dropouts
        self.activation = config.activation

        # --- Graph setup ---
        self.graph = graph  # setter handles .to(device)
        self.graph_path = None  # Will be set in from_graph_path

        # --- Inputs injector (strong dependency) ---
        self.injector = InputsInjector(device=self.device)

        # --- Training state (used by fit) ---
        self.last_training_config = None  # Last training config used
        self.state = {}
        self.optimizer = None
        self.scheduler = None

        # --- Shape validator ---
        self.validator = _ShapeValidator(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            num_nodes=self.graph.num_nodes
        )

        # --- Helpers and model components ---
        self._helpers = _GNSHelpers(
            device=self.device,
            graph=self.graph,
            num_msg_passing_layers=config.num_msg_passing_layers
        )

        self._build_encoder()
        self._build_message_passing_layers()
        self._build_decoder()

        # --- GroupNorm configuration ---
        self.groupnorm = torch.nn.GroupNorm(
            num_groups=self.config.num_groups,
            num_channels=config.latent_dim
        ).to(self.device)

        # --- Seed (optional) ---
        if self.seed is not None:
            set_seed(self.seed)


    @classmethod
    @accepts_config(GNSModelParams)
    def from_graph(cls, *, config: GNSModelParams, graph: Graph) -> "GNS":
        """
        Construct a GNS model from a loaded Graph and a model config.

        This constructor is intended for scenarios where the Graph object
        is already loaded in memory. The config must not contain `graph_path`.

        Args:
            config (GNSModelParams): Fully resolved model configuration.
            graph (Graph): In-memory Graph instance.

        Returns:
            GNS: Fully constructed model.

        Raises:
            RuntimeError: If `graph_path` is set in config (to avoid ambiguity).
        """
        return cls(config=config, graph=graph)


    @classmethod
    @accepts_config(GNSModelParams)
    def from_graph_path(cls, *, config: GNSModelParams, graph_path: Union[str, Path]) -> "GNS":
        """
        Construct a GNS model by loading the Graph from disk.

        This constructor is intended for cases where only the path to the graph
        is known (not the graph itself).

        Args:
            config (GNSModelParams): Fully resolved model configuration.
            graph_path (str or Path): Path to the graph file. Must be a valid Graph file.

        Returns:
            GNS: Fully constructed model.

        Raises:
            RuntimeError: If no graph path is provided via config or argument.
        """
        graph = Graph.load(graph_path)
        model = cls(config=config, graph=graph)
        model.graph_path = graph_path
        
        return model


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
        if hasattr(self, "_graph") and self._graph is not None:
            raiseError("Graph has already been set and cannot be reassigned.")
        if not isinstance(graph, Graph):
            raiseError("Graph must be of type Graph.")
        
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


    @accepts_config(GNSTrainingParams)
    @cr('GNS.fit')
    def fit(
        self,
        train_dataset: TorchDataset,
        eval_dataset: Optional[TorchDataset] = None,
        *,
        config: GNSTrainingParams,
        reset_state: bool = True
    ) -> Dict[str, list]:
        """
        Train the model using subgraph batching over both the input parameter space and the node space.

        Args:
            train_dataset: Dataset of input parameters and targets.
            eval_dataset: Optional validation dataset.
            config (GNSFitConfig): Training configuration dataclass.
            reset_state (bool): Whether to reset the optimizer and scheduler before training.

        Returns:
            Dict: Dictionary containing per-epoch train/test losses.
        """
        # --- Validate dataset shapes ---
        self._validate_shapes(train_dataset)
        if eval_dataset is not None:
            self._validate_shapes(eval_dataset)

        # --- Instantiate dataloaders ---
        input_dataloader = self._helpers.init_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=True,
            seed=self.seed,
        )

        eval_dataloader = None
        if eval_dataset is not None:
            eval_dataloader = self._helpers.init_dataloader(
                eval_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                shuffle=False,  # No se debe barajar el val/test
            )

        subgraph_loader = self._helpers.init_subgraph_loader(
            batch_size=config.node_batch_size,
            input_nodes=config.input_nodes,
            seed=self.seed,
        )

        # --- Initialize optimizer and scheduler ---)
        if reset_state or self.optimizer is None:
            self.optimizer = config.optimizer(self.parameters(), lr=config.lr)

        if reset_state or (self.scheduler is None and config.scheduler is not None):
            self.scheduler = config.scheduler(
                self.optimizer,
                step_size=config.lr_scheduler_step,
                gamma=config.lr_gamma
            )

        # --- Load training history if available ---
        state = self.state
        epoch_list      = state.get("epoch_list", [])
        train_loss_list = state.get("train_loss_list", [])
        test_loss_list  = state.get("test_loss_list", [])

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

            # Logging
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

            # Save state
            epoch_list.append(epoch)
            self.state = {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else {},
                "epoch_list": epoch_list,
                "train_loss_list": train_loss_list,
                "test_loss_list": test_loss_list,
            }
            self.last_training_config = config

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
                - If `return_loss` is True: average loss (float) over all batches (used in training/eval).
                - If `return_loss` is False: tensor of predictions with shape [B, N, F] (used in prediction mode).
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
    ) -> Union[Tuple[float, Data, Tensor, Tensor, Tensor], Tensor]:
        """
        Perform a single evaluation (validation or prediction) step.

        Args:
            subgraph (Data): Subgraph seed data from the loader.
            inputs_batch (Tensor): Input parameter batch of shape [B, D].
            targets_batch (Tensor or None): Ground truth targets if available.
            loss_fn (callable, optional): If provided, computes loss against targets.

        Returns:
            Union[Tuple[float, Data, Tensor, Tensor, Tensor], Tensor]:
                - If `loss_fn` is provided: a tuple (loss_value, graph, output, targets, loss_tensor) for evaluation.
                - If `loss_fn` is None: predictions on seed nodes as a Tensor of shape [S, F], where S is number of seed nodes.
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
            raiseError(f"Invalid dataset for {self.__class__.__name__}: {e}")

    
    def save(self, path: str) -> None:
        """
        Save the current model to a checkpoint file.
        """
        if self.graph_path is None:
            raiseError("Cannot save model without graph_path set.")

        checkpoint = {
            "graph_path": self.graph_path,
            "model_config": asdict(self.config),
            "state": self.state,
        }
        if self.last_training_config is not None:
            checkpoint["last_training_config"] = serialize_config(asdict(self.last_training_config))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n_epochs = len(self.state.get("epoch_list", [])) if self.state else 0
        filename = f"trained_model_{timestamp}_ep{n_epochs:03d}.pth"

        if os.path.isdir(path):
            path = os.path.join(path, filename)
        elif not path.endswith(".pth"):
            raiseError("Save path must end with '.pth' or be a directory.")

        torch.save(checkpoint, path)


    @classmethod
    def load(cls, path: str, device: Union[str, torch.device] = DEVICE) -> "GNS":
        """
        Load a GNS model from a checkpoint file.

        Args:
            path (str): Path to the .pth file.
            device (str or torch.device): Device to map the model to.

        Returns:
            GNS: Reconstructed model instance.
        """
        if not os.path.isfile(path):
            raiseError(f"Model file '{path}' not found.")
        if not path.endswith(".pth"):
            raiseError("Checkpoint file must have a '.pth' extension.")

        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raiseError("CUDA is not available. Use CPU instead.")

        checkpoint = torch.load(path, map_location=device)

        # --- Validate checkpoint structure ---
        required_keys = ["graph_path", "model_config", "state"]
        for key in required_keys:
            if key not in checkpoint:
                raiseError(f"Checkpoint is missing required key: '{key}'")

        # --- Reconstruct model ---
        model_config = GNSModelParams(deserialize_config(checkpoint["model_config"]))
        graph_path = checkpoint["graph_path"]
        model = cls.from_graph_path(config=model_config, graph_path=graph_path)

        # --- Load state ---
        state = checkpoint["state"]
        model.load_state_dict(state["model_state_dict"])
        model.state = state

        if checkpoint.get("last_training_config") is not None:
            last_training_config = GNSTrainingParams(
                **deserialize_config(checkpoint["last_training_config"])
            )
            model.last_training_config = last_training_config
            model.optimizer = last_training_config.optimizer(model.parameters(), lr=last_training_config.lr)
            model.optimizer.load_state_dict(state.get("optimizer_state_dict"))
            if last_training_config.scheduler is not None:
                model.scheduler = last_training_config.scheduler(
                    model.optimizer,
                    step_size=last_training_config.lr_scheduler_step,
                    gamma=last_training_config.lr_gamma
                )
                model.scheduler.load_state_dict(state.get("scheduler_state_dict"))

        model.eval()
        return model


    @classmethod
    def create_optimized_model(
        cls,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        optuna_optimizer: OptunaOptimizer,
    ) -> Tuple["GNS", Dict[str, Dict]]:
        """
        Create and train an optimized GNS model using Optuna hyperparameter search.

        The graph path must be specified in `optuna_optimizer.optimization_params['model']['graph_path']`.

        Args:
            train_dataset (Dataset): Training dataset.
            eval_dataset (Dataset): Validation dataset.
            optuna_optimizer (OptunaOptimizer): Configured Optuna optimizer with search space and options.

        Returns:
            Tuple[GNS, Dict[str, Dict]]: Final trained GNS model and best hyperparameters, structured as:
                {
                    "model": { ... },
                    "training": { ... }
                }
        """
        optimization_params = optuna_optimizer.optimization_params
        graph_path, search_space = cls.split_search_space(optimization_params)
        shared_graph = Graph.load(graph_path)

        @cr("GNS.optimization_function")
        def optimization_function(trial: Trial) -> float:
            # Sample hyperparameters
            model_params = {
                key: get_optimizing_value(key, val, trial)
                for key, val in search_space.get("model").items()
            }
            training_params = {
                key: get_optimizing_value(key, val, trial)
                for key, val in search_space.get("training").items()
            }

            model_params["graph_path"] = graph_path
            model = cls.from_graph(model_params, graph=shared_graph)

            pprint(0, f"\nTrial {trial.number + 1}/{optuna_optimizer.num_trials}:")
            pprint(0, f"  Model params: {json.dumps(model_params, default=hyperparams_serializer, indent=4)}")
            pprint(0, f"  Training params: {json.dumps(training_params, default=hyperparams_serializer, indent=4)}")

            try:
                if optuna_optimizer.pruner is not None:
                    # Pruning support: run one epoch at a time
                    val_loss = None
                    for epoch in range(training_params.get("epochs")):
                        epoch_params = training_params.copy()
                        epoch_params["epochs"] = 1
                        losses = model.fit(train_dataset, eval_dataset, **epoch_params, reset_state=False)
                        val_loss = losses["test_loss"][-1]
                        trial.report(val_loss, step=epoch)
                        if trial.should_prune():
                            with torch.no_grad():
                                del model
                                torch.cuda.empty_cache()
                            raise TrialPruned()
                else:
                    losses = model.fit(train_dataset, eval_dataset, **training_params)
                    val_loss = losses["test_loss"][-1]
                    trial.report(val_loss, step=9999)

            except RuntimeError as e:
                with torch.no_grad():
                    del model
                    torch.cuda.empty_cache()
                pprint(0, f"RuntimeError during trial {trial.number}: {str(e)}")
                raise e

            with torch.no_grad():
                del model
                torch.cuda.empty_cache()

            return val_loss

        # Run Optuna optimization
        best_params_flat = optuna_optimizer.optimize(objective_function=optimization_function)

        # Recover structured hyperparameters
        graph_path, search_space = cls.split_search_space(optuna_optimizer.optimization_params)
        best_model_params = search_space["model"].copy()
        best_training_params = search_space["training"].copy()

        # Fill with selected values
        for k in best_model_params:
            best_model_params[k] = best_params_flat[k]
        for k in best_training_params:
            best_training_params[k] = best_params_flat[k]
        best_model_params["graph_path"] = graph_path

        pprint(0, "\nBest hyperparameters:")
        pprint(0, f"  Model params: {json.dumps(best_model_params, indent=4, default=hyperparams_serializer)}")
        pprint(0, f"  Training params: {json.dumps(best_training_params, indent=4, default=hyperparams_serializer)}")

        final_model_config = GNSModelParams(**best_model_params)
        final_model = cls.from_graph_path(config=final_model_config, graph_path=graph_path)

        return final_model, best_training_params



    @staticmethod
    def split_search_space(optimization_params: Dict) -> Tuple[str, Dict]:
        """
        Splits the optimization_params dictionary into the shared graph_path
        and the actual search space (model.params + training).

        Args:
            optimization_params (Dict): Full dictionary with 'model' and 'training' sections.

        Returns:
            Tuple[str, Dict]: (graph_path, search_space)
        """
        if "model" not in optimization_params:
            raiseError("Missing 'model' section in optimization_params.")
        if "training" not in optimization_params:
            raiseError("Missing 'training' section in optimization_params.")

        model_section = optimization_params.get("model", {})
        graph_path = model_section.get("graph_path")
        model_params = model_section.get("params", {})
        training_params = optimization_params.get("training", {})

        if graph_path is None:
            raise ValueError("Missing required key 'graph_path' in optimization_params['model'].")

        return graph_path, {
            "model": model_params,
            "training": training_params
        }

    def __repr__(self):
        return (
            f"<GNSModel: {self.input_dim} → {self.latent_dim} → {self.output_dim}>\n"
            f" Layers: encoder({self.encoder_hidden_layers}), message({self.num_msg_passing_layers}), decoder({self.decoder_hidden_layers})\n"
            f" MLPs: message({self.message_hidden_layers}), update({self.update_hidden_layers})\n"
            f" Activation: {self.activation.__class__.__name__}, Dropout: {self.p_dropouts}, Device: {self.device}\n"
            f" Graph: {repr(self.graph)}\n"
            f" Params: {count_trainable_params(self):,} trainable\n"
            ")"

        )
