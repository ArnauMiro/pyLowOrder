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
from ..gns import GNSMLP, MessagePassingLayer, Graph, InputsInjector, _ShapeValidator, _GNSHelpers, GNSConfig, GNSTrainingConfig
from ..optimizer import OptunaOptimizer, TrialPruned
from ... import pprint, cr
from ...utils import raiseError
from ...utils.config_loader_factory import _model_from_config_path, load_gns_configs, _resolve_optuna_trial_params





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

    def __init__(self, config: Union[GNSConfig, dict], graph: Optional[Graph] = None) -> None:
        """
        Initialize the GNS model.

        Args:
            config (GNSConfig or dict): Model configuration. If a dict is provided, it will be resolved via config_loader_factory.
            graph (Graph, optional): If provided, overrides config.graph_path.

        Raises:
            RuntimeError: If neither graph nor config.graph_path is provided, or if both are.
        """
        super().__init__()

        # --- Fix: Ensure _graph is always defined early ---
        self._graph = None

        if isinstance(config, dict):
            config = load_gns_configs({"model": config})
        elif not isinstance(config, GNSConfig):
            raiseError("Invalid 'config' type passed to GNS(): must be GNSConfig or dict.")

        # --- Validation ---
        if not isinstance(config.activation, torch.nn.Module):
            raiseError("GNSConfig 'activation' must be a torch.nn.Module instance.")
        if not isinstance(config.device, torch.device):
            raiseError("GNSConfig 'device' must be a torch.device instance.")
        
        self.config = config
        self.device = torch.device(config.device)
        self.seed = config.seed
        self.p_dropouts = config.p_dropouts
        self.activation = config.activation

        # --- Device setup ---
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raiseError("CUDA is not available. Please use CPU instead.")
            torch.cuda.set_device(self.device)
        pprint(0, f"Using device: {self.device}", flush=True)

        # --- Reproducibility ---
        if self.seed is not None:
            set_seed(self.seed)

        # --- Graph loading logic ---
        if graph is not None and config.graph_path:
            raiseError(
                "Cannot provide both `graph` and `config.graph_path` to GNS(): this is ambiguous.\n"
                "Choose only one source:\n"
                "  • If you want GNS to load the graph, pass only config.graph_path.\n"
                "  • If you already have the graph loaded, pass it via `graph` and leave config.graph_path = None."
            )
        elif graph is not None:
            self.graph = graph  # Will validate and .to(device)
        elif config.graph_path:
            self.graph = Graph.load(config.graph_path)  # Setter ensures .validate() and .to(device)
        else:
            raiseError("GNS requires either a `graph` or a valid `config.graph_path`.")

        # --- Inputs injector ---
        self.injector = InputsInjector(device=self.device)

        # --- Training state ---
        self.state = {}
        self.optimizer = None
        self.scheduler = None
        self.checkpoint = None

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

        self.groupnorm = torch.nn.GroupNorm(
            num_groups=min(2, config.latent_dim),
            num_channels=config.latent_dim
        ).to(self.device)

        # --- Convenience aliases ---
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        self.output_dim = config.output_dim
        self.hidden_size = config.hidden_size
        self.num_msg_passing_layers = config.num_msg_passing_layers
        self.encoder_hidden_layers = config.encoder_hidden_layers
        self.decoder_hidden_layers = config.decoder_hidden_layers
        self.message_hidden_layers = config.message_hidden_layers
        self.update_hidden_layers = config.update_hidden_layers



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
            raiseError("Graph must be of type Graph.")
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
        eval_dataset: Optional[TorchDataset] = None,
        config: Union[GNSTrainingConfig, dict] = GNSTrainingConfig()
    ) -> Dict[str, list]:
        """
        Train the model using subgraph batching over both the input parameter space and the node space.

        Args:
            train_dataset: Dataset of input parameters and targets.
            eval_dataset: Optional validation dataset.
            config (GNSTrainingConfig or dict): Training configuration. If a dict is provided, it will be resolved via config_loader_factory.

        Returns:
            Dict: Dictionary containing per-epoch train/test losses.
        """
        if isinstance(config, dict):
            # Wrap in dummy config for loader to work
            _, config = load_gns_configs({"training": config}, with_training=True)
        elif not isinstance(config, GNSTrainingConfig):
            raiseError("Invalid 'config' passed to fit(): must be GNSTrainingConfig or dict.")
            
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

        Requires that `self.config.graph_path` is set; otherwise saving is disallowed
        to ensure reproducibility and correct reload behavior.

        Args:
            path (str): Either a full path to the checkpoint file (ending in .pth)
                        or a directory where the file will be saved with a timestamped name.

        Raises:
            raiseError: If config.graph_path is missing.
        """
        if not self.config.graph_path:
            raiseError(
                "Cannot save model: `config.graph_path` is missing.\n"
                "To save the model, instantiate GNS with a GNSConfig that includes `graph_path`, "
                "or reinstantiate the model after saving the graph to disk."
            )

        checkpoint = {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "state": self.state,
        }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n_epochs = len(self.state[2]) if self.state and len(self.state) > 2 else 0
        filename = f"trained_model_{timestamp}_ep{n_epochs:03d}.pth"

        if os.path.isdir(path):
            path = os.path.join(path, filename)
        elif not path.endswith(".pth"):
            raiseError("Save path must end with '.pth' or be a directory.")

        torch.save(checkpoint, path)


    @classmethod
    def load(cls, path: str, device: Union[str, torch.device] = DEVICE) -> "GNS":
        """
        Load a GNS model from a checkpoint file (.pth).

        Args:
            path (str): Path to the checkpoint file.
            device (Union[str, torch.device]): Target device.

        Returns:
            GNS: Loaded model instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            raiseError: If file extension is wrong or required config keys are missing.
        """
        if not os.path.isfile(path):
            raiseError(f"Model file '{path}' not found.")
        if not path.endswith(".pth"):
            raiseError("Checkpoint file must have a '.pth' extension.")

        device = torch.device(device or DEVICE)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raiseError("CUDA is not available. Use CPU instead.")
            torch.cuda.set_device(device)

        checkpoint = torch.load(path, map_location=device)

        # Validate checkpoint keys
        required_keys = ["config", "state_dict"]
        for key in required_keys:
            if key not in checkpoint:
                raiseError(f"Checkpoint is missing required key: '{key}'")

        config = GNSConfig(**checkpoint["config"])
        if not config.graph_path:
            raiseError(
                "Checkpoint config is missing 'graph_path'. Cannot reconstruct graph.\n"
                "If you created this model internally (e.g. via Optuna), make sure to instantiate it "
                "with a GNSConfig that includes graph_path before saving."
            )

        graph = Graph.load(config.graph_path).to(device)

        model = cls(config=config, graph=graph)
        model.load_state_dict(checkpoint["state_dict"])

        if "state" in checkpoint:
            model.state = checkpoint["state"]

        model.eval()
        return model


    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], with_training_config: bool = False):
        """
        Load a GNS model (and optionally training config) from a YAML configuration file.

        Args:
            yaml_path (str or Path): Path to the YAML config file.
            with_training_config (bool): If True, also returns the training config.

        Returns:
            GNS or (GNS, GNSTrainingConfig): The model and optionally the training configuration.
        """
        return _model_from_config_path(cls, yaml_path, model_type="gns", with_training=with_training_config)




    @classmethod
    def create_optimized_model(
        cls,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        optuna_optimizer: OptunaOptimizer,
    ) -> Tuple["GNS", Dict]:
        """
        Create and train an optimized GNS model using Optuna hyperparameter search.

        This method is compatible with pyLOM's Pipeline and does not require explicit graph_path as argument.
        The graph path must be included under `optuna_optimizer.optimization_params['graph_path']`. Note: 'graph_path' is not part of the search space and must be set externally.

        Args:
            train_dataset (Dataset): Training dataset.
            eval_dataset (Dataset): Validation dataset.
            optuna_optimizer (OptunaOptimizer): Configured Optuna optimizer with search space and options.

        Returns:
            Tuple[GNS, Dict]: A trained GNS model with the best-found hyperparameters, and the best parameters dict.
        """
        optimization_params = optuna_optimizer.optimization_params

        # Load the shared graph from the path provided in the config
        graph_path = optimization_params.get("graph_path")
        if not graph_path:
            raiseError("Missing 'graph_path' in optuna_optimizer.optimization_params.")
        
        shared_graph = Graph.load(graph_path)

        # Access the search space (which should NOT include graph_path)
        search_space = optimization_params.get("search_space", {})
        if not search_space:
            raiseError("Missing 'search_space' in optimization_params.")

        @cr("GNS.optimization_function")
        def optimization_function(trial: Trial) -> float:
            """
            Objective function for Optuna hyperparameter optimization.

            This function is called once per trial. It builds a GNS model using sampled hyperparameters,
            trains it, evaluates on validation set, and returns the validation loss.

            Args:
                trial (optuna.Trial): Optuna trial object used to sample hyperparameters.

            Returns:
                float: Final validation loss to be minimized.
            """

            # Sample hyperparameters
            hyperparams = {
                key: get_optimizing_value(key, val, trial)
                for section in ["model", "training"]
                for key, val in search_space.get(section, {}).items()
            }

            model_config, training_config = _resolve_optuna_trial_params(hyperparams, model_type="gns")
            model = cls(config=model_config, graph=shared_graph)

            pprint(0, f"\nTrial {trial.number + 1}/{optuna_optimizer.num_trials}. Training with:\n",
                json.dumps(hyperparams, indent=4, default=hyperparams_serializer), flush=True)

            try:
                if optuna_optimizer.pruner is not None:
                    for epoch in range(training_config.epochs):
                        epoch_cfg = GNSTrainingConfig(**{**asdict(training_config), "epochs": 1})
                        losses = model.fit(train_dataset, eval_dataset, config=epoch_cfg)
                        val_loss = losses["test_loss"][-1]
                        trial.report(val_loss, epoch)
                        if trial.should_prune():
                            torch.cuda.empty_cache()
                            del model
                            raise TrialPruned()
                else:
                    losses = model.fit(train_dataset, eval_dataset, config=training_config)
                    val_loss = losses["test_loss"][-1]
                    trial.report(val_loss, step=9999)

            except RuntimeError as e:
                torch.cuda.empty_cache()
                del model
                raise e

            torch.cuda.empty_cache()
            del model
            return val_loss

        # Run optimization
        best_params = optuna_optimizer.optimize(objective_function=optimization_function)

        pprint(0, f"\nBest hyperparameters found:\n{json.dumps(best_params, indent=4, default=hyperparams_serializer)}\n", flush=True)

        # Inject graph_path into final config so the model can be saved properly
        best_params["graph_path"] = graph_path

        final_model_config = GNSConfig(**{
            k: v for k, v in best_params.items() if k in GNSConfig.__annotations__
        })
        final_model = cls(config=final_model_config, graph=shared_graph)

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
