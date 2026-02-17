#!/usr/bin/env python
from __future__ import annotations

import copy
import json
from pathlib import Path
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
        self.nan_guard_enabled = True
        self.grad_clip_enabled = False
        self.grad_clip_max_norm = 1.0
        self.grad_clip_norm_type = 2.0
        self.best_metric_space = "scaled"
        self.debug_numerics = False
        self.debug_log_path = Path("gns_debug_numerics.jsonl")
        self.debug_every_n_steps = 200
        self.debug_overwrite_each_epoch = True
        self.debug_max_param_tensors = 12
        self._train_step_counter = 0

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
        self.best_metric_space = str(getattr(model, "best_metric_space", "scaled")).strip().lower()
        if self.best_metric_space not in {"scaled", "physical"}:
            raise RuntimeError(
                f"Invalid best_metric_space='{self.best_metric_space}'. Allowed: 'scaled', 'physical'."
            )
        self.nan_guard_enabled = bool(getattr(config, "nan_guard_enabled", True))
        self.grad_clip_enabled = bool(getattr(config, "grad_clip_enabled", False))
        self.grad_clip_max_norm = float(getattr(config, "grad_clip_max_norm", 1.0))
        self.grad_clip_norm_type = float(getattr(config, "grad_clip_norm_type", 2.0))
        self.debug_numerics = bool(getattr(config, "debug_numerics", False))
        dbg_path = getattr(config, "debug_log_path", None)
        self.debug_log_path = Path(dbg_path) if dbg_path else Path("gns_debug_numerics.jsonl")
        self.debug_every_n_steps = max(1, int(getattr(config, "debug_every_n_steps", 200)))
        self.debug_overwrite_each_epoch = bool(getattr(config, "debug_overwrite_each_epoch", True))
        self.debug_max_param_tensors = max(1, int(getattr(config, "debug_max_param_tensors", 12)))
        self._train_step_counter = 0

        # Optional epoch-0 diagnostics: evaluate losses before any optimizer step.
        # This is only recorded once for fresh runs (no previous trained epochs).
        self._maybe_record_epoch_zero_losses(
            train_input_dl=train_input_dl,
            train_subgraph_dl=train_subgraph_dl,
            eval_input_dl=eval_input_dl,
            eval_subgraph_dl=eval_subgraph_dl,
            loss_fn=loss_fn,
            best_metric=best_metric,
            epoch_list=epoch_list,
            train_loss_list=train_loss_list,
            test_loss_list=test_loss_list,
        )

        for epoch in range(1 + len(epoch_list), 1 + total_epochs):
            if self.debug_numerics and self.debug_overwrite_each_epoch:
                self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.debug_log_path.open("w") as f:
                    f.write("")
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
                    test_log = (
                        f" | Eval loss: {test_loss:.4e} | Eval {best_metric}"
                        f"({self.best_metric_space}): {best_val_loss:.4e}"
                    )
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
                "best_metric_space": self.best_metric_space,
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

    def _maybe_record_epoch_zero_losses(
        self,
        *,
        train_input_dl,
        train_subgraph_dl,
        eval_input_dl,
        eval_subgraph_dl,
        loss_fn: torch.nn.Module,
        best_metric: str,
        epoch_list: list,
        train_loss_list: list,
        test_loss_list: list,
    ) -> None:
        # Keep backward compatibility for resumed runs: do not prepend again.
        if len(epoch_list) > 0 or len(train_loss_list) > 0 or len(test_loss_list) > 0:
            return

        train_loss0 = self.run_epoch(
            input_dataloader=train_input_dl,
            subgraph_loader=train_subgraph_dl,
            loss_fn=loss_fn,
            return_loss=True,
            is_train=False,
        )
        train_loss_list.insert(0, train_loss0)

        if eval_input_dl is not None:
            eval_result0 = self.run_epoch(
                input_dataloader=eval_input_dl,
                subgraph_loader=eval_subgraph_dl,
                loss_fn=loss_fn,
                return_loss=True,
                metric=best_metric,
                is_train=False,
            )
            if isinstance(eval_result0, tuple):
                eval_loss0, _ = eval_result0
            else:
                eval_loss0 = eval_result0
            test_loss_list.insert(0, eval_loss0)

        pprint(
            0,
            f"[diag] Epoch 0 (no training) | Train loss: {train_loss_list[0]:.4e}" +
            (f" | Eval loss: {test_loss_list[0]:.4e}" if len(test_loss_list) > 0 else ""),
            flush=True,
        )

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
                            out_metric = self._to_metric_space(out)
                            tgt_metric = self._to_metric_space(targets)
                            total_metric += torch.mean(torch.abs(out_metric - tgt_metric)).item()
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

    def _to_metric_space(self, tensor: Tensor) -> Tensor:
        if self.best_metric_space == "scaled":
            return tensor
        model = self.model
        inverse_fn = getattr(model, "inverse_output_fn", None)
        if inverse_fn is None:
            raise RuntimeError(
                "best_metric_space='physical' requires model.inverse_output_fn to be set."
            )
        out = inverse_fn(tensor)
        if isinstance(out, torch.Tensor):
            return out.to(tensor.device)
        return torch.as_tensor(out, dtype=tensor.dtype, device=tensor.device)

    def _extract_seeded_output_targets(
        self,
        *,
        subgraph: Data,
        inputs_batch: Tensor,
        output_full: Tensor,
        targets_batch: Optional[Tensor],
        G: Data,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Build seeded output/target tensors from structured [B, N, *] views.

        This avoids relying on potentially inconsistent flattened masks in `G`
        and guarantees output/target alignment on the same seed selection.
        """
        if inputs_batch.ndim == 1:
            B = 1
        else:
            B = int(inputs_batch.size(0))

        N = int(subgraph.num_nodes)
        if output_full.ndim != 2:
            raise RuntimeError(f"Unexpected output_full.ndim={output_full.ndim}; expected 2.")
        if output_full.size(0) != B * N:
            raise RuntimeError(
                f"Unexpected output length {int(output_full.size(0))}; expected B*N={B}*{N}={B*N}."
            )

        seed_mask_local = getattr(subgraph, "seed_mask", None)
        if seed_mask_local is None:
            seed_mask_local = torch.ones(N, dtype=torch.bool, device=output_full.device)
        else:
            seed_mask_local = seed_mask_local.to(device=output_full.device, dtype=torch.bool)

        out_view = output_full.view(B, N, output_full.size(-1))
        output = out_view[:, seed_mask_local, :].reshape(-1, output_full.size(-1))

        targets: Optional[Tensor] = None
        if targets_batch is not None:
            tb = targets_batch
            if tb.ndim == 2:
                tb = tb.unsqueeze(0)
            subset = getattr(subgraph, "subset", None)
            if subset is not None:
                subset = subset.to(device=tb.device, dtype=torch.long)
                tb = tb.index_select(1, subset)
            targets = tb[:, seed_mask_local.to(tb.device), :].reshape(-1, tb.size(-1))
        elif getattr(G, "y", None) is not None and getattr(G, "seed_mask", None) is not None:
            targets = G.y[G.seed_mask]

        return output, targets

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
        self._train_step_counter += 1
        G = model.injector.replicate_inject(subgraph, inputs_batch, targets_batch)

        model.optimizer.zero_grad()
        output_full = model.forward(G)
        output, targets = self._extract_seeded_output_targets(
            subgraph=subgraph,
            inputs_batch=inputs_batch,
            output_full=output_full,
            targets_batch=targets_batch,
            G=G,
        )
        if targets is None:
            raise RuntimeError("Targets are required in training but were not provided.")

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
                "output_full.shape": tuple(output_full.shape),
                "output.shape": tuple(output.shape),
                "targets.shape": tuple(targets.shape),
            }
            raise RuntimeError(f"Shape mismatch detected in _train_one_batch: {debug_info}")

        if self.nan_guard_enabled:
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite values detected in model outputs during training.")
            if not torch.isfinite(targets).all():
                raise RuntimeError("Non-finite values detected in targets during training.")
        if model._counter % 100 == 0:
            model._debug_print(f" - Training batch {model._counter}: output.shape={output.shape}, targets.shape={targets.shape}")
        loss = self._compute_loss(output, targets, loss_fn)
        if self.debug_numerics and (self._train_step_counter % self.debug_every_n_steps == 0):
            self._debug_log(
                event="train_step",
                step=self._train_step_counter,
                payload={
                    "loss": self._tensor_stats(loss),
                    "output": self._tensor_stats(output),
                    "targets": self._tensor_stats(targets),
                },
            )
        if self.nan_guard_enabled and not torch.isfinite(loss):
            if self.debug_numerics:
                self._debug_log(
                    event="nonfinite_loss",
                    step=self._train_step_counter,
                    payload={
                        "loss": self._tensor_stats(loss),
                        "output": self._tensor_stats(output),
                        "targets": self._tensor_stats(targets),
                    },
                )
            raise RuntimeError(
                f"Non-finite training loss detected at batch {model._counter}. "
                "Aborting to prevent NaN propagation."
            )
        model._debug_print(f" - Computed loss: {loss.item():.4e}. Backpropagating...")
        loss.backward()

        if self.nan_guard_enabled:
            grad_snapshots = []
            grad_nonfinite_name = None
            for name, param in model.named_parameters():
                grad = param.grad
                if self.debug_numerics and grad is not None and len(grad_snapshots) < self.debug_max_param_tensors:
                    grad_snapshots.append({
                        "name": name,
                        "grad": self._tensor_stats(grad),
                    })
                if grad is not None and not torch.isfinite(grad).all():
                    grad_nonfinite_name = name
                    break
            if grad_nonfinite_name is not None:
                if self.debug_numerics:
                    self._debug_log(
                        event="nonfinite_grad",
                        step=self._train_step_counter,
                        payload={
                            "parameter": grad_nonfinite_name,
                            "loss": self._tensor_stats(loss),
                            "output": self._tensor_stats(output),
                            "targets": self._tensor_stats(targets),
                            "grads": grad_snapshots,
                        },
                    )
                raise RuntimeError(f"Non-finite gradients detected in parameter '{grad_nonfinite_name}'.")
            if self.debug_numerics and (self._train_step_counter % self.debug_every_n_steps == 0):
                self._debug_log(
                    event="grad_snapshot",
                    step=self._train_step_counter,
                    payload={"grads": grad_snapshots},
                )

        if self.grad_clip_enabled:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=self.grad_clip_max_norm,
                norm_type=self.grad_clip_norm_type,
            )
            if self.nan_guard_enabled and not torch.isfinite(total_norm):
                raise RuntimeError("Non-finite gradient norm after clipping.")

        model.optimizer.step()

        return loss.item(), G, output, targets, loss

    def _tensor_stats(self, t: Tensor) -> Dict[str, Union[str, float, int, list]]:
        td = t.detach()
        finite = torch.isfinite(td)
        finite_count = int(finite.sum().item())
        total = int(td.numel())
        out: Dict[str, Union[str, float, int, list]] = {
            "shape": list(td.shape),
            "dtype": str(td.dtype),
            "numel": total,
            "finite_count": finite_count,
            "nonfinite_count": total - finite_count,
        }
        if finite_count > 0:
            tf = td[finite]
            out.update({
                "min": float(tf.min().item()),
                "max": float(tf.max().item()),
                "mean": float(tf.mean().item()),
                "std": float(tf.std().item()) if tf.numel() > 1 else 0.0,
                "absmax": float(tf.abs().max().item()),
            })
        return out

    def _debug_log(self, *, event: str, step: int, payload: Dict[str, Union[dict, list, str, float, int]]) -> None:
        try:
            record = {"event": event, "step": step, **payload}
            self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.debug_log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # Debug logging must not break training flow.
            pass

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
        output_full = model.forward(G)
        output, targets = self._extract_seeded_output_targets(
            subgraph=subgraph,
            inputs_batch=inputs_batch,
            output_full=output_full,
            targets_batch=targets_batch,
            G=G,
        )

        if loss_fn is not None:
            model._debug_print("Computing evaluation loss...")
            if targets is None:
                raise RuntimeError("Targets are required for evaluation loss but were not provided.")
            assert output.shape == targets.shape, f"Output shape {output.shape} != target shape {targets.shape}"
            if self.nan_guard_enabled:
                if not torch.isfinite(output).all():
                    raise RuntimeError("Non-finite values detected in model outputs during evaluation.")
                if not torch.isfinite(targets).all():
                    raise RuntimeError("Non-finite values detected in targets during evaluation.")
            loss = self._compute_loss(output, targets, loss_fn)
            if self.nan_guard_enabled and not torch.isfinite(loss):
                raise RuntimeError("Non-finite evaluation loss detected.")
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
