#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 30/03/2025

import os
import json
import getpass
import datetime
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, Callable
from dataclasses import asdict, replace

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from optuna import Trial

from dacite import from_dict, Config

from .. import DEVICE, set_seed
from ... import pprint, cr
from ...utils import (
    raiseError,
    get_git_commit,
)
from ..utils import (
    count_trainable_params,
    cleanup_tensors,
    sample_params,
    hyperparams_serializer,
)
from ..gns import (
    GNSMLP,
    MessagePassingLayer,
    Graph,
    InputsInjector,
    _ShapeValidator,
    _GNSHelpers,
)
from ..utils.wrappers import config_from_kwargs
from ..utils.config_schema import (
    GraphSpec,
    GNSModelConfig,
    GNSTrainingConfig,
    TorchDataloaderConfig,
    SubgraphDataloaderConfig,
)
from ..utils.resolvers import (
    resolve_import,
    resolve_device,
    resolve_activation,
    resolve_loss,
    resolve_optimizer,
    resolve_scheduler,
)
from ..utils.config_loader import GeneralTypeHooks
from ..optimizer import OptunaOptimizer, TrialPruned




class GNS(torch.nn.Module):
    r"""
    Graph Neural Solver class for predicting aerodynamic variables on RANS meshes.

    This model uses a message-passing GNN architecture with MLPs for the message and update functions.
    It supports subgraph batching and is optimized for training on large RANS meshes.

    Note:
        This constructor assumes a fully loaded `Graph` object and a validated `GNSModelConfig` config.
        For standard usage, prefer using `GNS.from_graph(...)` or `GNS.from_graph_path(...)`.
    """

    def __init__(self, *, config: GNSModelConfig, graph: Graph) -> None:
        """
        Internal constructor for GNS. Do not use directly unless you know exactly what you're doing.

        This method resolves the pure DTO configuration into runtime PyTorch objects
        (device, activation function, RNG generators, etc.) and initializes all core
        model components.

        Args:
            config (GNSModelConfig): Pure DTO model configuration.
            graph (Graph): Fully loaded in-memory Graph object.
        """
        super().__init__()

        # --- Basic validation ---
        if not isinstance(config, GNSModelConfig):
            raiseError("Expected config to be a GNSModelConfig.")
        if not isinstance(graph, Graph):
            raiseError("Expected graph to be a Graph instance.")

        # --- Resolve DTO config into runtime objects ---
        self.model_config = config
        self.device = resolve_device(config.device)
        self.seed = config.seed
        self.p_dropout = config.p_dropout
        self.activation = resolve_activation(config.activation)

        # --- Graph setup ---
        self.graph = graph  # property setter handles device transfer and validation
        self.graph_path = None  # Will be set if loaded from path

        # --- Inputs injector (strong dependency) ---
        self.injector = InputsInjector(device=self.device)

        # --- Training state (used by fit) ---
        self.last_training_config = None
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
            num_groups=config.groupnorm_groups,
            num_channels=config.latent_dim
        ).to(self.device)

        # --- Seed & RNG generators ---
        if self.seed is not None:
            # Global determinism for weight init, dropout, etc.
            set_seed(self.seed)

            # Runtime generators (CPU), decoupled streams
            self._generator = torch.Generator(device="cpu").manual_seed(self.seed)         # train inputs
            self._val_generator = torch.Generator(device="cpu").manual_seed(self.seed + 1) # eval/predict inputs

            self._sg_generator_train = torch.Generator(device="cpu").manual_seed(self.seed + 2)  # train subgraphs
            self._sg_generator_eval  = torch.Generator(device="cpu").manual_seed(self.seed + 3)  # eval/predict subgraphs
        else:
            self._generator = None
            self._val_generator = None
            self._sg_generator_train = None
            self._sg_generator_eval = None




    @classmethod
    # @config_from_kwargs(GNSModelConfig)
    def from_graph(cls, *, config: GNSModelConfig, graph: Graph) -> "GNS":
        """
        Construct a GNS model from a loaded Graph and a model config.

        This constructor is intended for scenarios where the Graph object
        is already loaded in memory. The config must not contain `graph_path`.

        Args:
            config (GNSModelConfig): Fully resolved model configuration.
            graph (Graph): In-memory Graph instance.

        Returns:
            GNS: Fully constructed model.

        Raises:
            RuntimeError: If `graph_path` is set in config (to avoid ambiguity).
        """
        return cls(config=config, graph=graph)


    @classmethod
    # @config_from_kwargs(GNSModelConfig)
    def from_graph_path(cls, *, config: GNSModelConfig, graph_path: Union[str, Path]) -> "GNS":
        """
        Construct a GNS model by loading the Graph from disk.

        This constructor is intended for cases where only the path to the graph
        is known (not the graph itself).

        Args:
            config (GNSModelConfig): Fully resolved model configuration.
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
            output_size=self.model_config.latent_dim,
            hidden_size=self.model_config.hidden_size,
            activation=self.activation,
            drop_p=self.p_dropout
        ).to(self.device)

    def _build_decoder(self):
        self.decoder = GNSMLP(
            input_size=self.model_config.latent_dim,
            output_size=self.model_config.output_dim,
            hidden_size=self.model_config.hidden_size,
            activation=self.activation,
            drop_p=self.p_dropout
        ).to(self.device)

    def _build_message_passing_layers(self):
        self.conv_layers_list = torch.nn.ModuleList([
            MessagePassingLayer(
                in_channels=2 * self.model_config.latent_dim + self._edge_dim,
                out_channels=self.model_config.latent_dim,
                hidden_size=self.model_config.hidden_size,
                message_hidden_layers=self.model_config.message_hidden_layers,
                update_hidden_layers=self.model_config.update_hidden_layers,
                activation=self.activation,
                drop_p=self.p_dropout,
            ) for _ in range(self.model_config.num_msg_passing_layers)
        ]).to(self.device)

    @graph.setter
    def graph(self, graph: Graph) -> None:
        """Graph property to set the graph object."""
        if hasattr(self, "_graph") and self._graph is not None:
            raiseError("Graph has already been set and cannot be reassigned.")
        if not isinstance(graph, Graph):
            raiseError("Graph must be of type Graph.")

        # Validate and move to device if needed
        graph.validate()
        if graph.device != self.device:
            graph = graph.to(self.device)

        # Update dependent dims (model_config must be set before assigning graph)
        self._encoder_input_dim = graph.x.shape[1] + self.model_config.input_dim
        self._edge_dim = graph.edge_attr.shape[1]

        # Assign and fingerprint
        self._graph = graph
        self.graph_fingerprint = graph.fingerprint()

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
    ) -> torch.Tensor:
        """
        Run inference on the model.

        Args:
            X (Tensor or Dataset): Operational inputs. If Tensor, shape [B, D].
            batch_size (int): Used only if X is a Dataset.
            node_batch_size (int): Number of seed nodes per subgraph batch.

        Returns:
            Tensor: Predictions of shape [B, N, F].
        """
        self._validate_shapes(X)

        input_cfg = TorchDataloaderConfig(
            batch_size=batch_size,
            shuffle=False,
        )
        subgraph_cfg = SubgraphDataloaderConfig(
            batch_size=node_batch_size,
            shuffle=False,
            input_nodes=None,
        )

        input_dataloader = self._helpers.init_dataloader(
            X, input_cfg, generator=None
        )
        subgraph_loader = self._helpers.init_subgraph_loader(
            subgraph_cfg, generator=None
        )

        self.eval()
        with torch.no_grad():
            return self._run_epoch(
                input_dataloader,
                subgraph_loader,
                is_train=False,
                return_loss=False
            )

    @config_from_kwargs(GNSTrainingConfig)
    @cr('GNS.fit')
    def fit(
        self,
        train_dataset: TorchDataset,
        eval_dataset: Optional[TorchDataset] = None,
        *,
        config: GNSTrainingConfig,
        reset_state: bool = True,
        on_epoch_end: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, list]:
        """
        Train the model using subgraph batching over both the input parameter space and the node space.
        """
        # --- Validate dataset shapes ---
        self._validate_shapes(train_dataset)
        if eval_dataset is not None:
            self._validate_shapes(eval_dataset)

        # --- Train loaders ---
        train_input_dl = self._helpers.init_dataloader(
            train_dataset,
            config.dataloader,
            generator=self._generator if config.dataloader.shuffle else None,
        )
        train_sg_loader = self._helpers.init_subgraph_loader(
            config.subgraph_loader,
            generator=self._sg_generator_train if config.subgraph_loader.shuffle else None,
        )

        # --- Eval loaders (optional) ---
        eval_dataloader = None
        eval_subgraph_loader = None
        if eval_dataset is not None:
            val_dl_cfg = replace(config.dataloader, shuffle=False)
            eval_dataloader = self._helpers.init_dataloader(
                eval_dataset,
                val_dl_cfg,
                generator=None,
            )
            val_sg_cfg = replace(config.subgraph_loader, shuffle=False)
            eval_subgraph_loader = self._helpers.init_subgraph_loader(
                val_sg_cfg,
                generator=None,
            )

        # --- Initialize optimizer and scheduler ---
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
                train_input_dl,
                train_sg_loader,
                return_loss=True,
                loss_fn=config.loss_fn,
                is_train=True
            )
            train_loss_list.append(train_loss)

            if eval_dataloader is not None:
                test_loss = self._run_epoch(
                    eval_dataloader,
                    eval_subgraph_loader,
                    return_loss=True,
                    loss_fn=config.loss_fn,
                    is_train=False
                )
                test_loss_list.append(test_loss)

            # Logging
            log_this_epoch = (
                config.print_every is not None
                and config.print_every > 0
                and epoch % config.print_every == 0
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

            if on_epoch_end is not None:
                on_epoch_end(epoch, train_loss)

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
            return torch.cat(outputs, dim=0).reshape(-1, self.graph.num_nodes, self.model_config.output_dim)


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

    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the current model to a checkpoint file.
        """
        if self.graph_path is None:
            raiseError("Cannot save model without graph_path set.")

        path = Path(path)

        checkpoint = {
            "graph_path": self.graph_path,
            "model_config": asdict(self.model_config),
            "state": self.state,
            "metadata": {
                "saved_at": datetime.datetime.now().isoformat(),
                "torch_version": torch.__version__,
                "user": getpass.getuser(),
                "git_commit": get_git_commit(),
            },
        }
        if self.last_training_config is not None:
            checkpoint["last_training_config"] = serialize_config_dict(asdict(self.last_training_config))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n_epochs = len(self.state.get("epoch_list", [])) if self.state else 0
        filename = f"trained_model_{timestamp}_ep{n_epochs:03d}.pth"

        if os.path.isdir(path):
            path = os.path.join(path, filename)
        elif path.suffix != ".pth":
            raiseError("Save path must end with '.pth' or be a directory.")

        torch.save(checkpoint, path)


    @classmethod
    def load(cls, path: Union[str, Path], device: Union[str, torch.device] = DEVICE) -> "GNS":
        """
        Load a GNS model from a checkpoint file.

        Args:
            path (str): Path to the .pth file.
            device (str or torch.device): Device to map the model to.

        Returns:
            GNS: Reconstructed model instance.
        """
        path = Path(path)
        if not os.path.isfile(path):
            raiseError(f"Model file '{path}' not found.")
        if not path.suffix == ".pth":
            raiseError(f"Checkpoint file must have a '.pth' extension. Extension was {path.suffix}.")

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
        model_config = GNSModelConfig(**resolve_config_dict(checkpoint["model_config"]))
        graph_path = checkpoint["graph_path"]
        model = cls.from_graph_path(config=model_config, graph_path=graph_path)

        # --- Load state ---
        state = checkpoint["state"]
        model.load_state_dict(state["model_state_dict"])
        model.state = state

        if checkpoint.get("last_training_config") is not None:
            last_training_config = GNSTrainingConfig(
                **resolve_config_dict(checkpoint["last_training_config"])
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
        optuna_optimizer: "OptunaOptimizer",  # quoted to avoid circular import issues
    ) -> Tuple["GNS", Dict[str, Dict]]:
        """
        Create and train an optimized GNS model using Optuna hyperparameter search.

        The graph path must be specified in `optuna_optimizer.optimization_params['model']['graph_path']`.

        Returns:
            A tuple (trained_model, best_hyperparams) with final model and config dictionaries.
        """
        optimization_params = optuna_optimizer.optimization_params
        graph_path = optimization_params["model"]["graph_path"]
        model_space = optimization_params["model"]["params"]
        training_space = optimization_params["training"]
        shared_graph = Graph.load(graph_path)

        # Prepare unified config resolution
        type_hooks = GeneralTypeHooks()
        dacite_cfg = Config(type_hooks=type_hooks, strict=True)

        @cr("GNS.optimization_function")
        def optimization_function(trial: Trial) -> float:
            # Sample model hyperparameters
            model_dict = resolve_config_dict(sample_params(model_space, trial))
            training_dict = resolve_config_dict(sample_params(training_space, trial))
            model_cfg = from_dict(data_class=GNSModelConfig, data=model_dict, config=dacite_cfg)
            training_cfg = from_dict(data_class=GNSTrainingConfig, data=training_dict, config=dacite_cfg)

            # Create and train model
            model = cls.from_graph(config=model_cfg, graph=shared_graph)

            pprint(0, f"\nTrial {trial.number + 1}/{optuna_optimizer.num_trials}:")
            pprint(0, f"  Model params: {json.dumps(model_cfg, default=hyperparams_serializer, indent=4)}")
            pprint(0, f"  Training params: {json.dumps(training_cfg, default=hyperparams_serializer, indent=4)}")

            def on_epoch(epoch: int, val_loss: float) -> None:
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    raise TrialPruned()

            try:
                if optuna_optimizer.pruner is not None:
                    losses = model.fit(
                        train_dataset,
                        eval_dataset,
                        config=training_cfg,
                        on_epoch_end=on_epoch,
                    )
                else:
                    losses = model.fit(train_dataset, eval_dataset, config=training_cfg)
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

        # Run optimization
        best_params_flat = optuna_optimizer.optimize(objective_function=optimization_function)

        # Reconstruct config dictionaries
        best_model_cfg = model_space.copy()
        best_training_cfg = training_space.copy()

        for k in best_model_cfg:
            if isinstance(best_model_cfg[k], dict) and "type" in best_model_cfg[k]:
                best_model_cfg[k] = best_params_flat[k]
        for k in best_training_cfg:
            if isinstance(best_training_cfg[k], dict) and "type" in best_training_cfg[k]:
                best_training_cfg[k] = best_params_flat[k]

        best_model_cfg["graph_path"] = graph_path

        pprint(0, "\nBest hyperparameters:")
        pprint(0, f"  Model params: {json.dumps(best_model_cfg, indent=4, default=hyperparams_serializer)}")
        pprint(0, f"  Training params: {json.dumps(best_training_cfg, indent=4, default=hyperparams_serializer)}")

        final_model_cfg = from_dict(data_class=GNSModelConfig, data=resolve_config_dict(best_model_cfg), config=dacite_cfg)
        final_model = cls.from_graph_path(config=final_model_cfg, graph_path=graph_path)

        return final_model, best_training_cfg



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
        model_cfg = model_section.get("params", {})
        training_cfg = optimization_params.get("training", {})

        if graph_path is None:
            raise ValueError("Missing required key 'graph_path' in optimization_params['model'].")

        return graph_path, {
            "model": model_cfg,
            "training": training_cfg
        }

    def __repr__(self):
        return (
            f"<GNSModel: {self.model_config.input_dim} → {self.model_config.latent_dim} → {self.model_config.output_dim}>\n"
            f" Layers: encoder({self.encoder_hidden_layers}), message({self.model_config.num_msg_passing_layers}), decoder({self.decoder_hidden_layers})\n"
            f" MLPs: message({self.model_config.message_hidden_layers}), update({self.model_config.update_hidden_layers})\n"
            f" Activation: {self.activation.__class__.__name__}, Dropout: {self.p_dropout}, Device: {self.device}\n"
            f" Graph: {repr(self.graph)}\n"
            f" Params: {count_trainable_params(self):,} trainable\n"
            ")"

        )
