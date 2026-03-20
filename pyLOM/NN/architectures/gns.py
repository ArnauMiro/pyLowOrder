#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Message passing graph neural network architecture for NN Module
#
# Last rev: 30/03/2025

import json
import getpass
import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional, Callable
from dataclasses import asdict, replace
import copy
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from optuna import Trial

from dacite import from_dict, Config

from .. import set_seed
from ... import pprint, cr
from ...utils import (
    raiseError,
    get_git_commit,
)
from ..utils import (
    count_trainable_params,
    cleanup_tensors,
)

from ..utils.optuna_utils import (
    sample_params,
    _materialize_space
)

from ..gns import (
    GNSMLP,
    MessagePassingLayer,
    Graph,
    InputsInjector,
    _ShapeValidator,
    _GNSHelpers,
)
from ..gns.trainer import _GNSTrainingLoop
from ..utils.wrappers import config_from_kwargs
from ..utils.config_schema import (
    GraphSpec,
    GNSModelConfig,
    GNSTrainingConfig,
    TorchDataloaderConfig,
    SubgraphDataloaderConfig,
)
from ...utils.config_resolvers import (
    resolve_device,
    resolve_activation,
    resolve_loss,
    resolve_optimizer,
    resolve_scheduler,
)
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

    def __init__(self, *, config: GNSModelConfig, graph: Graph, graph_spec: Optional[GraphSpec] = None) -> None:
        """
        Internal constructor for GNS. Do not use directly unless you know exactly what you're doing.

        This method resolves the pure DTO configuration into runtime PyTorch objects
        (device, activation function, RNG generators, etc.) and initializes all core
        model components.

        Parameters
        ----------
        config : GNSModelConfig
            Pure DTO with model hyperparameters.
        graph : Graph
            Fully loaded in-memory Graph object.
        graph_spec : Optional[GraphSpec]
            Optional provenance info (e.g., path/id). Not a hyperparameter.
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
        self.debug = config.debug

        # --- Graph & provenance ---
        self.graph_spec = graph_spec or GraphSpec()
        self.graph = graph  # setter validates, moves to device, updates dims, and sets self.graph_fingerprint

        # --- Inputs injector (strong dependency) ---
        self.injector = InputsInjector(device=self.device)

        # --- Training state (used by fit) ---
        self.last_training_config = None
        self.state: Dict[str, Any] = {}
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
        self.groupnorm_layers = torch.nn.ModuleList(
            [
                torch.nn.GroupNorm(
                    num_groups=config.groupnorm_groups,
                    num_channels=config.latent_dim,
                )
                for _ in range(config.num_msg_passing_layers)
            ]
        ).to(self.device)

        # --- Seed & RNG generators ---
        if self.seed is not None:
            set_seed(self.seed)  # global determinism

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


    def _debug_print(self, *args, rank: int = -1, **kwargs) -> None:
        if self.debug:
            pprint(rank, *args, **kwargs, flush=True)

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
    def from_graph_path(cls, *, config: GNSModelConfig, graph_path: Union[str, Path]) -> "GNS":
        """
        Construct a GNS model by loading the Graph from disk.

        Parameters
        ----------
        config : GNSModelConfig
            Fully resolved model configuration.
        graph_path : str or Path
            Path to the graph file. Must be a valid Graph file.

        Returns
        -------
        GNS
            Fully constructed model with provenance info attached.
        """
        path = Path(graph_path)
        graph = Graph.load(path)
        spec = GraphSpec(path=str(path))
        return cls(config=config, graph=graph, graph_spec=spec)


    def _build_encoder(self):
        self.encoder = GNSMLP(
            input_size=self._encoder_input_dim,
            output_size=self.model_config.latent_dim,
            hidden_size=self.model_config.hidden_size,
            num_hidden_layers=self.model_config.encoder_hidden_layers,
            activation=self.activation,
            drop_p=self.p_dropout,
        ).to(self.device)

    def _build_decoder(self):
        self.decoder = GNSMLP(
            input_size=self.model_config.latent_dim,
            output_size=self.model_config.output_dim,
            hidden_size=self.model_config.hidden_size,
            num_hidden_layers=self.model_config.decoder_hidden_layers,
            activation=self.activation,
            drop_p=self.p_dropout,
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

    @property
    def graph(self) -> Graph:
        """
        Loaded RANS graph (read‑only after first assignment).

        Returns
        -------
        Graph
            In‑memory graph already validated and moved to the model device.

        Raises
        ------
        RuntimeError
            If accessed before being set (should not happen in normal flow).
        """
        if self._graph is None:
            raiseError("Graph has not been set yet.")
        return self._graph


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
            pprint(0, f"Moving graph to device {self.device}.")
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
        self._debug_print(f"Initiating forward pass on device {self.device}...")
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr

        self._debug_print(f" - x.shape: {x.shape}, edge_index.shape: {edge_index.shape}, edge_attr.shape: {edge_attr.shape}")
        h = self.activation(self.encoder(x))
        self._debug_print(f" - Encoded node features h.shape: {h.shape}")
        for conv, norm in zip(self.conv_layers_list, self.groupnorm_layers):
            h = conv(h, edge_index, edge_attr)
            h = self.activation(h)
            h = norm(h)
        self._debug_print(f" - After message passing h.shape: {h.shape}. Running decoder...")
        y_hat = self.decoder(h)
        return y_hat

    @cr('GNS.predict')
    def predict(
        self,
        X: Union[Tensor, TorchDataset],
        batch_size: int = 1,
        node_batch_size: int = 256,
    ) -> np.ndarray:
        """
        Run inference on the model.

        Args:
            X (Tensor or Dataset): Operational inputs. If Tensor, shape [B, D].
            batch_size (int): Used only if X is a Dataset.
            node_batch_size (int): Number of seed nodes per subgraph batch.

        Returns:
            np.ndarray: Predictions of shape [B, N, F].
        """
        self._debug_print("Starting prediction...")
        try:
            self.validator.validate(X)
        except Exception as e:
            raiseError(f"Invalid dataset for {self.__class__.__name__}: {e}")

        self._debug_print(f"Creating dataloaders (batch_size={batch_size}, node_batch_size={node_batch_size})...")
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

        self._debug_print("Running evaluation epoch...")
        self.eval()
        runner = _GNSTrainingLoop(self)
        with torch.no_grad():
            return runner.run_epoch(
                input_dataloader=input_dataloader,
                subgraph_loader=subgraph_loader,
                loss_fn=None,
                return_loss=False,
                is_train=False,
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

        Notes
        -----
        - Delegates the epoch loop to the internal runner `_GNSTrainingLoop` (separation of concerns).
        - Keeps the external contract expected by `Pipeline` intact.
        - Config DTOs are resolved to runtime objects here (loss, optimizer, scheduler).
        - RNG generators are passed as runtime args to helpers; DTOs remain immutable.
        - Tracks and restores the best validation checkpoint at the end of training.
        """
        # --- Validate dataset shapes ---
        try:
            self.validator.validate(train_dataset)
            if eval_dataset is not None:
                self.validator.validate(eval_dataset)
        except Exception as e:
            raiseError(f"Invalid dataset for {self.__class__.__name__}: {e}")

        self._debug_print(f"Validated datasets. Train size: {len(train_dataset)}" +
                     (f", Eval size: {len(eval_dataset)}" if eval_dataset is not None else ""))

        # --- Resolve runtime components from DTOs ---
        loss_fn = resolve_loss(config.loss_fn)
        optimizer_cls = resolve_optimizer(config.optimizer)
        scheduler_cls = resolve_scheduler(config.scheduler)  # may be None

        # --- Train loaders ---
        self._debug_print(f"Creating training dataloaders with parameters: {asdict(config.dataloader)} and {asdict(config.subgraph_loader)}")
        train_input_dl = self._helpers.init_dataloader(
            train_dataset,
            config.dataloader,
            generator=self._generator if config.dataloader.shuffle else None,
        )
        sg_generator = self._sg_generator_train if config.subgraph_loader.shuffle else None
        # Materialize effective seed nodes once (input_nodes or seed_selector-derived)
        # so they are persisted in last_training_config and can be reproduced exactly.
        effective_input_nodes = config.subgraph_loader.input_nodes
        if effective_input_nodes is None:
            effective_input_nodes = self._helpers._resolve_seed_selector(
                config.subgraph_loader,
                generator=sg_generator,
            )
        effective_input_nodes = self._helpers._resolve_input_nodes(effective_input_nodes)
        train_sg_cfg = replace(config.subgraph_loader, input_nodes=effective_input_nodes.tolist())
        config = replace(config, subgraph_loader=train_sg_cfg)

        train_sg_loader = self._helpers.init_subgraph_loader(
            train_sg_cfg,
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
            val_sg_cfg = replace(config.subgraph_loader, shuffle=False, input_nodes=None)
            eval_subgraph_loader = self._helpers.init_subgraph_loader(
                val_sg_cfg,
                generator=None,
            )

        # --- Initialize optimizer and scheduler ---
        if reset_state or self.optimizer is None:
            # Support weight decay (regularization) for small datasets
            try:
                self.optimizer = optimizer_cls(self.parameters(), lr=config.lr, weight_decay=getattr(config, 'weight_decay', 0.0))
            except TypeError:
                # Fallback if optimizer does not accept weight_decay
                self.optimizer = optimizer_cls(self.parameters(), lr=config.lr)

        if reset_state or (self.scheduler is None and scheduler_cls is not None):
            # Default to StepLR-style kwargs; callers can change scheduler type via config.
            if scheduler_cls is not None:
                self.scheduler = scheduler_cls(
                    self.optimizer,
                    step_size=config.lr_scheduler_step,
                    gamma=config.lr_gamma
                )
            else:
                self.scheduler = None

        # --- Load training history if available ---
        state = self.state
        epoch_list      = state.get("epoch_list", [])
        train_loss_list = state.get("train_loss_list", [])
        test_loss_list  = state.get("test_loss_list", [])

        # --- Propagate config knobs consumed by the training loop via model attrs ---
        # Keep this explicit to avoid silent fallback to runner defaults.
        self.best_metric = config.best_metric
        self.best_metric_space = config.best_metric_space
        self.loss_weight_alpha = config.weighted_loss_alpha

        # Delegate epoch loop to the training runner (encapsulates train/val and best-checkpoint logic)
        runner = _GNSTrainingLoop(self)
        return runner.train(
            train_input_dl=train_input_dl,
            train_subgraph_dl=train_sg_loader,
            eval_input_dl=eval_dataloader,
            eval_subgraph_dl=eval_subgraph_loader,
            loss_fn=loss_fn,
            config=config,
            on_epoch_end=on_epoch_end,
            epoch_list=epoch_list,
            train_loss_list=train_loss_list,
            test_loss_list=test_loss_list,
        )
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the current model to a checkpoint file.

        Notes
        -----
        - Stores pure DTOs for model/training configs.
        - Stores graph provenance (GraphSpec) and graph fingerprint for mismatch detection.
        - Stores training state (including optimizer/scheduler states if available).
        """
        path = Path(path)

        # --- Minimal validation ---
        if not hasattr(self, "graph_spec"):
            raiseError("graph_spec is missing; cannot save provenance.")
        if not hasattr(self, "graph_fingerprint"):
            raiseError("graph_fingerprint is missing; cannot save provenance.")

        # --- State: ensure model_state_dict is present ---
        state_to_save = dict(self.state) if getattr(self, "state", None) else {}
        if "model_state_dict" not in state_to_save:
            state_to_save["model_state_dict"] = self.state_dict()

        # --- Build checkpoint (always) ---
        checkpoint = {
            "model_config": asdict(self.model_config),
            "state": state_to_save,
            "provenance": {
                "graph_spec": asdict(self.graph_spec),
                "graph_fingerprint": self.graph_fingerprint,
            },
            "metadata": {
                "saved_at": datetime.datetime.now().isoformat(),
                "torch_version": str(torch.__version__),  # normalize to str
                "user": getpass.getuser(),
                "git_commit": get_git_commit(obj=GNS)
            },
        }
        if getattr(self, "last_training_config", None) is not None:
            checkpoint["last_training_config"] = asdict(self.last_training_config)

        # --- Resolve final path ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n_epochs = len(state_to_save.get("epoch_list", []))
        filename = f"trained_model_{timestamp}_ep{n_epochs:03d}.pth"

        if path.is_dir():
            path = path / filename
        elif path.suffix != ".pth":
            raiseError("Save path must end with '.pth' or be a directory.")

        torch.save(checkpoint, path)


    @classmethod
    def load(
        cls,
        ckpt_path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        graph: Optional[Graph] = None,
        graph_spec: Optional[GraphSpec] = None,
        strict_fingerprint: bool = True,
    ) -> "GNS":
        ckpt = torch.load(ckpt_path, map_location="cpu")  # map to CPU first; we will move later

        # Rebuild pure DTOs from checkpoint
        saved_cfg = GNSModelConfig(**ckpt["model_config"])
        # Resolve target device
        target_device = torch.device(device) if device is not None else torch.device(saved_cfg.device)
        # Create a new config with the target device
        model_cfg = replace(saved_cfg, device=str(target_device))

        # Resolve graph and graph_spec
        prov = ckpt["provenance"]
        saved_spec_dict = prov.get("graph_spec", {}) or {}
        saved_fp = prov.get("graph_fingerprint")

        if graph is None:
            graph_path = saved_spec_dict.get("path")
            if graph_path is None:
                raiseError("Checkpoint has no graph path and no `graph` was provided.")
            graph = Graph.load(graph_path)
            graph_spec = GraphSpec(**saved_spec_dict)
        else:
            graph_spec = graph_spec or GraphSpec()

        # Instantiate the model
        model = cls(config=model_cfg, graph=graph, graph_spec=graph_spec)

        if strict_fingerprint and saved_fp is not None and model.graph_fingerprint != saved_fp:
            raiseError("Graph fingerprint mismatch between checkpoint and provided/loaded graph.")

        # Load model state dict
        state = ckpt["state"]
        model.load_state_dict(state["model_state_dict"])
        model.state = state

        # Rehydrate training state if available
        if "last_training_config" in ckpt and ckpt["last_training_config"] is not None:
            last_cfg = GNSTrainingConfig(**ckpt["last_training_config"])
            model._rehydrate_training_state(last_cfg, state)

        # Move model to target device
        model.to(target_device)
        model.eval()
        return model

    def _rehydrate_training_state(self, last_cfg: GNSTrainingConfig, state: Dict[str, Any]) -> None:
        """
        Rebuild optimizer and scheduler from the saved training config and state dicts.

        Parameters
        ----------
        last_cfg : GNSTrainingConfig
            Training DTO saved in the checkpoint.
        state : Dict[str, Any]
            State dict blob from the checkpoint, expected keys:
            - 'optimizer_state_dict'
            - 'scheduler_state_dict'
        """
        optimizer_cls = resolve_optimizer(last_cfg.optimizer)
        scheduler_cls = resolve_scheduler(last_cfg.scheduler)

        # Optimizer
        self.optimizer = optimizer_cls(self.parameters(), lr=last_cfg.lr)
        opt_sd = state.get("optimizer_state_dict")
        if opt_sd:
            self.optimizer.load_state_dict(opt_sd)

        # Scheduler (optional)
        if scheduler_cls is not None:
            self.scheduler = scheduler_cls(
                self.optimizer,
                step_size=last_cfg.lr_scheduler_step,
                gamma=last_cfg.lr_gamma
            )
            sch_sd = state.get("scheduler_state_dict")
            if sch_sd:
                self.scheduler.load_state_dict(sch_sd)
        else:
            self.scheduler = None

        self.last_training_config = last_cfg



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

        Returns
        -------
        (GNS, Dict[str, Any]):
            - trained_model (GNS): instancia del modelo (no entrenado).
            - fit_kwargs (dict): kwargs para GNS.fit, p.ej. {"config": GNSTrainingConfig}.
        """
        if eval_dataset is None:
            raiseError("Optuna optimization requires a validation dataset (eval_dataset).")

        # ---- Extract search space and load shared graph once ----
        optimization_params = optuna_optimizer.optimization_params
        graph_path, search_space = cls._split_search_space(optimization_params)
        model_space = search_space["model"]
        training_space = search_space["training"]

        shared_graph = Graph.load(graph_path)  # one in-memory graph for all trials

        # ---- Dacite config for DTOs (type hooks if you need them) ----
        dacite_cfg = Config(strict=True)

        @cr("GNS.optimization_function")
        def objective(trial: Trial) -> float:
            # 1) Sample hyperparameters (plain dicts)
            model_dict = sample_params(model_space, trial)
            training_dict = sample_params(training_space, trial)

            # 2) Build DTOs (pure dataclasses)
            model_cfg = from_dict(data_class=GNSModelConfig, data=model_dict, config=dacite_cfg)
            training_cfg = from_dict(data_class=GNSTrainingConfig, data=training_dict, config=dacite_cfg)

            # 3) Build model (no resolvers here; __init__ y fit se encargan internamente)
            model = cls(config=model_cfg, graph=shared_graph, graph_spec=GraphSpec(path=str(graph_path)))

            # 4) Logging (DTOs -> dicts)
            pprint(0, f"\nTrial {trial.number + 1}/{optuna_optimizer.num_trials}:")
            pprint(0, "  Model params: " + json.dumps(asdict(model_cfg), indent=4))
            pprint(0, "  Training params: " + json.dumps(asdict(training_cfg), indent=4))

            # 5) Optional pruning callback
            def on_epoch(epoch: int, val_loss: float) -> None:
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    raise TrialPruned()

            # 6) Train (fit se encarga de resolver loss/optimizer/scheduler)
            try:
                if optuna_optimizer.pruner is not None:
                    losses = model.fit(
                        train_dataset,
                        eval_dataset,
                        config=training_cfg,
                        on_epoch_end=on_epoch,
                    )
                else:
                    losses = model.fit(
                        train_dataset,
                        eval_dataset,
                        config=training_cfg
                    )
                    # report final metric explicitly for completeness
                    if losses["test_loss"]:
                        trial.report(losses["test_loss"][-1], step=9999)

            except TrialPruned:
                # Explicit cleanup on pruning
                with torch.no_grad():
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise
            except RuntimeError as e:
                with torch.no_grad():
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                pprint(0, f"RuntimeError during trial {trial.number}: {str(e)}")
                raise e

            # 7) Select final validation loss
            if not losses["test_loss"]:
                raiseError("Training returned no validation losses; cannot score the trial.")
            val_loss = float(losses["test_loss"][-1])

            # 8) Cleanup memoria
            with torch.no_grad():
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return val_loss

        # ---- Run optimization ----
        best_params_flat = optuna_optimizer.optimize(objective_function=objective)

        # ---- Reconstruct best dicts for model and training (solo claves buscadas) ----
        best_model_cfg_dict    = _materialize_space(model_space,    best_params_flat)
        best_training_cfg_dict = _materialize_space(training_space, best_params_flat)

        best_model_cfg = from_dict(GNSModelConfig,    best_model_cfg_dict,    config=dacite_cfg)
        best_training_cfg = from_dict(GNSTrainingConfig, best_training_cfg_dict, config=dacite_cfg)

        # ---- Logging best hyperparameters ---- (Already done by optuna)
        # pprint(0, "\nBest hyperparameters:")
        # pprint(0, "  Model params: " + json.dumps(best_model_cfg_dict, indent=4))
        # pprint(0, "  Training params: " + json.dumps(best_training_cfg_dict, indent=4))

        # ---- Build final DTOs & final model ----
        best_model_cfg = from_dict(data_class=GNSModelConfig, data=best_model_cfg_dict, config=dacite_cfg)
        best_training_cfg = from_dict(data_class=GNSTrainingConfig, data=best_training_cfg_dict, config=dacite_cfg)

        final_model = cls(
            config=best_model_cfg,
            graph=shared_graph,  # reuse the shared graph
            graph_spec=GraphSpec(path=str(graph_path)),
        )

        return final_model, {"config": best_training_cfg}



    @staticmethod
    def _split_search_space(optimization_params: Dict) -> Tuple[str, Dict]:
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

        model_section = optimization_params.get("model")
        graph_path = model_section.get("graph_path")
        model_cfg = model_section.get("config")
        training_cfg = optimization_params.get("training")

        if graph_path is None:
            raise ValueError("Missing required key 'graph_path' in optimization_params['model'].")

        return graph_path, {
            "model": model_cfg,
            "training": training_cfg
        }

    def __repr__(self):
        return (
            f"<GNSModel: {self.model_config.input_dim} → "
            f"{self.model_config.latent_dim} → {self.model_config.output_dim}>\n"
            f" Layers: encoder({self.model_config.encoder_hidden_layers}), "
            f"message({self.model_config.num_msg_passing_layers}), "
            f"decoder({self.model_config.decoder_hidden_layers})\n"
            f" MLPs: message({self.model_config.message_hidden_layers}), "
            f"update({self.model_config.update_hidden_layers})\n"
            f" Activation: {self.activation.__class__.__name__}, "
            f"Dropout: {self.p_dropout}, Device: {self.device}\n"
            f" Graph: {repr(self.graph)}\n"
            f" Params: {count_trainable_params(self):,} trainable\n"
            ")"
        )


# ------------------------------------------------------------
