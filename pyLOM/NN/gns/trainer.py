#!/usr/bin/env python
from __future__ import annotations

import copy
from typing import Optional, Callable, Tuple, Dict, Union

import torch
from torch import Tensor
from torch_geometric.data import Data

from ... import pprint
from ..utils import cleanup_tensors
from ..utils.config_schema import GNSTrainingConfig


class _GNSTrainingLoop:
    """
    Internal helper that encapsulates the GNS training and evaluation loops.

    - Runs full epochs (run_epoch) over provided dataloaders.
    - Implements train/val per-subgraph steps (_train_one_batch, _eval_one_batch).
    - Manages best checkpoint selection during validation via train(...).
    """

    def __init__(self, model: "GNS") -> None:
        self.model = model
        # Optional weighted MSE: enable by setting model.loss_weight_alpha > 0
        self.weight_alpha = float(getattr(model, "loss_weight_alpha", 0.0) or 0.0)

    def train(
        self,
        *,
        train_input_dl,
        train_subgraph_dl,
        eval_input_dl,
        eval_subgraph_dl,
        loss_fn: torch.nn.Module,
        config: GNSTrainingConfig,
        on_epoch_end: Optional[Callable[[int, float], None]],
        epoch_list: list,
        train_loss_list: list,
        test_loss_list: list,
    ) -> Dict[str, list]:
        model = self.model
        total_epochs = len(epoch_list) + config.epochs
        state = model.state
        best_val_loss = state.get("best_val_loss", float("inf"))
        best_epoch = state.get("best_epoch")
        best_checkpoint = None

        # Metric used to select best checkpoint on validation
        best_metric = getattr(model, "best_metric", "loss")

        for epoch in range(1 + len(epoch_list), 1 + total_epochs):
            train_loss = self.run_epoch(
                input_dataloader=train_input_dl,
                subgraph_loader=train_subgraph_dl,
                loss_fn=loss_fn,
                return_loss=True,
                is_train=True,
            )
            train_loss_list.append(train_loss)

            test_loss = None
            if eval_input_dl is not None:
                eval_result = self.run_epoch(
                    input_dataloader=eval_input_dl,
                    subgraph_loader=eval_subgraph_dl,
                    loss_fn=loss_fn,
                    return_loss=True,
                    metric=best_metric,
                    is_train=False,
                )
                if isinstance(eval_result, tuple):
                    test_loss, test_metric = eval_result
                else:
                    test_loss = eval_result
                    test_metric = test_loss
                test_loss_list.append(test_loss)

                if test_metric < best_val_loss:
                    best_val_loss = test_metric
                    best_epoch = epoch
                    best_checkpoint = {
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(model.optimizer.state_dict()) if model.optimizer is not None else {},
                        "scheduler_state_dict": copy.deepcopy(model.scheduler.state_dict()) if model.scheduler is not None else {},
                    }

            log_this_epoch = (
                config.print_every is not None and config.print_every > 0 and epoch % config.print_every == 0
            )
            if log_this_epoch:
                if test_loss is not None and best_metric != "loss":
                    test_log = f" | Eval loss: {test_loss:.4e} | Eval {best_metric}: {best_val_loss:.4e}"
                else:
                    test_log = f" | Eval loss: {test_loss:.4e}" if test_loss is not None else ""
                pprint(0, f"Epoch {epoch}/{total_epochs} | Train loss: {train_loss:.4e}{test_log}", flush=True)
                if model.device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(model.device) / 1024 ** 2
                    reserved = torch.cuda.memory_reserved(model.device) / 1024 ** 2
                    pprint(0, f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB", flush=True)

            epoch_list.append(epoch)
            model.state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
                "scheduler_state_dict": model.scheduler.state_dict() if model.scheduler is not None else {},
                "epoch_list": epoch_list,
                "train_loss_list": train_loss_list,
                "test_loss_list": test_loss_list,
                "best_val_loss": best_val_loss,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
            }
            model.last_training_config = config

            if on_epoch_end is not None:
                on_epoch_end(epoch, train_loss)

        if best_checkpoint is not None:
            model.load_state_dict(best_checkpoint["model_state_dict"])
            if model.optimizer is not None and best_checkpoint["optimizer_state_dict"]:
                model.optimizer.load_state_dict(best_checkpoint["optimizer_state_dict"])
            if model.scheduler is not None and best_checkpoint["scheduler_state_dict"]:
                model.scheduler.load_state_dict(best_checkpoint["scheduler_state_dict"])
            model.state.update({
                "model_state_dict": best_checkpoint["model_state_dict"],
                "optimizer_state_dict": best_checkpoint["optimizer_state_dict"],
                "scheduler_state_dict": best_checkpoint["scheduler_state_dict"],
            })

        return {
            "train_loss": train_loss_list,
            "test_loss": test_loss_list,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }

    def run_epoch(
        self,
        input_dataloader,
        subgraph_loader,
        *,
        loss_fn: Optional[torch.nn.Module] = None,
        return_loss: bool = False,
        metric: Optional[str] = None,
        is_train: bool = False,
    ) -> Union[float, Tensor]:
        model = self.model
        model._debug_print(f"{'Training' if is_train else 'Evaluating/Predicting'} epoch on device {model.device}...")
        model._debug_print(f" - Input dataloader len/batch size: {input_dataloader.__len__()}/{getattr(input_dataloader, 'batch_size', 'N/A')}")
        model.train() if is_train else model.eval()
        model._debug_print(f"Set model to {'train' if is_train else 'eval'} mode.")

        outputs = []
        total_loss = 0.0
        total_metric = 0.0
        input_batches = 0
        seed_batches_total = 0
        model._debug_print("Initializing context manager...")
        context = torch.enable_grad() if is_train else torch.no_grad()
        model._debug_print("Initialized context manager.")

        last_graph = None
        last_output = None
        last_targets = None
        last_loss = None
        num_losses = 0
        logged_first_batch_stats = False

        model._debug_print("Starting main loop over input batches...")
        with context:
            model._debug_print("Entering input dataloader loop...")
            for batch in input_dataloader:
                input_batches += 1
                model._debug_print("Processing new input batch...")
                inputs_batch = batch[0].to(model.device)
                try:
                    targets_batch = batch[1].to(model.device)
                except IndexError:
                    targets_batch = None
                model._debug_print(
                    f"Processing new input batch with inputs.shape={inputs_batch.shape}" +
                    (f", targets.shape={targets_batch.shape}" if targets_batch is not None else ", no targets")
                )

                for seed_batch in subgraph_loader:
                    seed_batches_total += 1
                    if isinstance(seed_batch, Data):
                        subgraph = seed_batch
                    else:
                        seed_nodes = seed_batch[0] if isinstance(seed_batch, (list, tuple)) else seed_batch
                        subgraph = model._helpers.build_subgraph(seed_nodes)
                    if is_train:
                        model._debug_print("Training on new batch...")
                        loss_val, G, out, targets, loss = self._train_one_batch(
                            subgraph=subgraph,
                            inputs_batch=inputs_batch,
                            targets_batch=targets_batch,
                            loss_fn=loss_fn,
                        )
                    elif return_loss:
                        model._debug_print("Evaluating on new batch...")
                        loss_val, G, out, targets, loss = self._eval_one_batch(
                            subgraph=subgraph,
                            inputs_batch=inputs_batch,
                            targets_batch=targets_batch,
                            loss_fn=loss_fn,
                        )
                    else:
                        model._debug_print("Predicting on new batch...")
                        out = self._eval_one_batch(
                            subgraph=subgraph,
                            inputs_batch=inputs_batch,
                            targets_batch=targets_batch,
                            loss_fn=None,
                        )
                        outputs.append(out)
                        continue  # Skip loss-related code

                    if return_loss:
                        total_loss += loss_val
                        if metric == "mae" and out is not None and targets is not None:
                            total_metric += torch.mean(torch.abs(out - targets)).item()
                        last_graph = G
                        last_output = out
                        last_targets = targets
                        last_loss = loss
                        if not logged_first_batch_stats and out is not None and targets is not None:
                            logged_first_batch_stats = True
                            try:
                                pred_mean = float(out.mean().item())
                                pred_std = float(out.std().item())
                                targ_mean = float(targets.mean().item())
                                targ_std = float(targets.std().item())
                                pprint(
                                    0,
                                    f"[diag] {'train' if is_train else 'eval'} batch stats: "
                                    f"pred_mean={pred_mean:.4f}, pred_std={pred_std:.4f}, "
                                    f"targ_mean={targ_mean:.4f}, targ_std={targ_std:.4f}",
                                    flush=True,
                                )
                            except Exception:
                                pass

                    num_losses += 1

            if is_train and model.scheduler is not None:
                model.scheduler.step()

        if return_loss:
            cleanup_tensors({
                "graph": last_graph,
                "output": last_output,
                "targets": last_targets,
                "loss": last_loss,
            })
            avg_loss = total_loss / max(1, num_losses)
            pprint(
                0,
                f"[diag] {'train' if is_train else 'eval'} epoch: input_batches={input_batches}, "
                f"seed_batches={seed_batches_total}, num_losses={num_losses}, avg_loss={avg_loss:.4e}",
                flush=True,
            )
            if metric == "mae":
                avg_metric = total_metric / max(1, num_losses)
                return avg_loss, avg_metric
            return avg_loss

        outputs_numpy = torch.cat(outputs, dim=0).cpu().numpy()
        outputs_numpy = outputs_numpy.reshape(-1, model.graph.num_nodes, model.model_config.output_dim)
        return outputs_numpy

    def _train_one_batch(
        self,
        *,
        subgraph: Data,
        inputs_batch: Tensor,
        targets_batch: Tensor,
        loss_fn: torch.nn.Module,
    ) -> Tuple[float, Data, Tensor, Tensor, Tensor]:
        model = self.model
        if not hasattr(model, "_counter"):
            model._counter = 0
        model._counter += 1
        G = model.injector.replicate_inject(subgraph, inputs_batch, targets_batch)

        model.optimizer.zero_grad()
        output = model.forward(G)[G.seed_mask]
        targets = G.y[G.seed_mask]

        if output.shape != targets.shape:
            subgraph_seed_count = int(getattr(subgraph, "seed_mask", torch.tensor([])).sum().item()) if getattr(subgraph, "seed_mask", None) is not None else None
            G_seed_count = int(G.seed_mask.sum().item()) if getattr(G, "seed_mask", None) is not None else None
            debug_info = {
                "inputs_batch.shape": tuple(inputs_batch.shape),
                "targets_batch.shape": tuple(targets_batch.shape) if targets_batch is not None else None,
                "subgraph.num_nodes": int(subgraph.num_nodes),
                "subgraph.seed_mask.sum": subgraph_seed_count,
                "subgraph.subset.shape": tuple(getattr(subgraph, "subset", torch.tensor([])).shape) if getattr(subgraph, "subset", None) is not None else None,
                "G.x.shape": tuple(G.x.shape),
                "G.y.shape": tuple(G.y.shape) if getattr(G, "y", None) is not None else None,
                "G.seed_mask.shape": tuple(G.seed_mask.shape) if getattr(G, "seed_mask", None) is not None else None,
                "G.seed_mask.sum": G_seed_count,
                "output.shape": tuple(output.shape),
                "targets.shape": tuple(targets.shape),
            }
            model._debug_print(f"[GNS] Shape mismatch detected in _train_one_batch: {debug_info}")

        assert output.shape == targets.shape, f"Output shape {output.shape} != target shape {targets.shape}"
        if model._counter % 100 == 0:
            model._debug_print(f" - Training batch {model._counter}: output.shape={output.shape}, targets.shape={targets.shape}")
        loss = self._compute_loss(output, targets, loss_fn)
        model._debug_print(f" - Computed loss: {loss.item():.4e}. Backpropagating...")
        loss.backward()
        model.optimizer.step()

        return loss.item(), G, output, targets, loss

    def _eval_one_batch(
        self,
        *,
        subgraph: Data,
        inputs_batch: Tensor,
        targets_batch: Optional[Tensor],
        loss_fn: Optional[torch.nn.Module],
    ) -> Union[Tuple[float, Data, Tensor, Tensor, Tensor], Tensor]:
        model = self.model
        model._debug_print("Evaluating on new batch...")
        G = model.injector.replicate_inject(subgraph, inputs_batch, targets_batch)
        output = model.forward(G)[G.seed_mask]

        if loss_fn is not None:
            model._debug_print("Computing evaluation loss...")
            targets = G.y[G.seed_mask]
            assert output.shape == targets.shape, f"Output shape {output.shape} != target shape {targets.shape}"
            loss = self._compute_loss(output, targets, loss_fn)
            return loss.item(), G, output, targets, loss

        return output

    def _compute_loss(self, output: Tensor, targets: Tensor, loss_fn: Optional[torch.nn.Module]) -> Tensor:
        """
        Compute loss with optional weighted MSE.
        If self.weight_alpha > 0, use weighted MSE. Otherwise use loss_fn if provided.
        """
        if self.weight_alpha > 0.0:
            residual = output - targets
            weights = 1.0 + self.weight_alpha * torch.abs(targets)
            return torch.mean(weights * residual * residual)
        if loss_fn is not None:
            return loss_fn(output, targets)
        return torch.mean((output - targets) ** 2)
