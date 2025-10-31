#!/usr/bin/env python
"""
Minimal MLP training runner for scaling pipeline.
- Reads optional hparams.json from --resudir
- Trains single or DDP according to --ddp on|off
- Writes training_results_mlp_{single|ddp}.npy (via pyLOM Pipeline), predictions and scalers
- Optional true_vs_pred figure unless --no-figures
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.distributed as dist
import pyLOM, pyLOM.NN

seed = 19

def _is_main_process() -> bool:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    for k in ("RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v) == 0
            except Exception:
                break
    return True

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ddp", dest="ddp_mode", default="off", choices=["on","off"], help="DDP mode")
    p.add_argument("--resudir", default="MLP_DLR_airfoil", help="Results directory")
    p.add_argument("--basedir", default="/home/airbus/CETACEO_cp_interp/DATA/DLR_pylom/", help="Dataset base dir")
    p.add_argument("--casestr", default="NRL7301", help="Case string")
    p.add_argument("--no-figures", action="store_true", help="Disable figures (for batch runs)")
    p.add_argument("--batch-size", type=int, default=119, help="Per-process batch size override")
    return p.parse_known_args()[0]

def load_dataset(fname, inputs_scaler, outputs_scaler):
    """Robust dataset loader supporting multiple field/variable naming schemes.

    Tries combinations in this order:
      1) field=["CP"],          variables=["AoA","Mach"]
      2) field=["CP"],          variables=[]
      3) field=["CoefPressure"], variables=["AoA","Mach"]
      4) field=["CoefPressure"], variables=[]
    """
    # We prefer only the variables 'M' and 'aoa' when available
    combos = [
        (["CP"], ["M", "aoa"]),
        (["CoefPressure"], ["M", "aoa"]),
        (["CP"], ["AoA", "Mach"]),
        (["CoefPressure"], ["AoA", "Mach"]),
        (["CP"], []),
        (["CoefPressure"], []),
    ]
    last_err = None
    for fields, vars_ in combos:
        try:
            return pyLOM.NN.Dataset.load(
                fname,
                field_names=fields,
                add_mesh_coordinates=True,
                variables_names=vars_,
                inputs_scaler=inputs_scaler,
                outputs_scaler=outputs_scaler,
            )
        except KeyError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    # If all combos fail, raise the last error for visibility
    raise last_err if last_err is not None else RuntimeError("Unknown dataset load failure")

def _first_existing(*paths: str) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def true_vs_pred_plot(y_true, y_pred, path, mode: str | None = None):
    try:
        y_true_np = y_true.detach().cpu().numpy() if hasattr(y_true, 'detach') else np.asarray(y_true)
        y_pred_np = y_pred.detach().cpu().numpy() if hasattr(y_pred, 'detach') else np.asarray(y_pred)
        if y_true_np.ndim == 1: y_true_np = y_true_np.reshape(-1, 1)
        if y_pred_np.ndim == 1: y_pred_np = y_pred_np.reshape(-1, 1)
    except Exception:
        return
    import matplotlib.pyplot as plt
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass
    C = y_true_np.shape[1]
    fig, axes = plt.subplots(C, 1, figsize=(10, 5*C))
    if C == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        yt = y_true_np[:, j].ravel(); yp = y_pred_np[:, j].ravel()
        ax.scatter(yt, yp, s=1, c='tab:blue', alpha=0.5)
        try:
            a, b = np.polyfit(yt, yp, 1)
            xx = np.linspace(yt.min(), yt.max(), 100)
            yy = a*xx + b
            ss_res = np.sum((yp - (a*yt + b))**2)
            ss_tot = np.sum((yp - np.mean(yp))**2) + 1e-12
            r2 = 1.0 - ss_res/ss_tot
            rmse = float(np.sqrt(np.mean((yp - yt)**2)))
            ax.plot(xx, yy, 'r--', lw=1.5, label=f"y={a:.3f}x+{b:.3f}, R2={r2:.3f}")
        except Exception:
            rmse = float(np.sqrt(np.mean((yp - yt)**2)))
        mode_str = f" - {mode.upper()}" if mode else ""
        ax.set_title(f"True vs Pred{mode_str} (comp {j+1}, RMSE={rmse:.3e})")
        ax.set_xlabel("True"); ax.set_ylabel("Pred"); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"Saved: {path}")

def main():
    args = _parse_args()
    ddp = (args.ddp_mode == 'on')
    mode_tag = 'ddp' if ddp else 'single'

    resudir = os.environ.get('PYLOM_RESUDIR', '').strip() or args.resudir
    if _is_main_process():
        print(f"pyLOM NN: DDP={'ON' if ddp else 'OFF'}  resudir={resudir}")
    pyLOM.NN.create_results_folder(resudir)

    basedir = args.basedir; casestr = args.casestr

    # Device select (GPU if available, else CPU)
    device = pyLOM.NN.select_device("cuda" if torch.cuda.is_available() else "cpu")

    # Data (robust file discovery: supports *_TRAIN.h5 vs *_train.h5, etc.)
    input_scaler = pyLOM.NN.MinMaxScaler(); output_scaler = pyLOM.NN.MinMaxScaler()
    tr_path = _first_existing(
        os.path.join(basedir, f"{casestr}_TRAIN.h5"),
        os.path.join(basedir, f"{casestr}_train.h5"),
        os.path.join(basedir, f"{casestr}_TRAIN.hdf5"),
        os.path.join(basedir, f"{casestr}_train.hdf5"),
    )
    te_path = _first_existing(
        os.path.join(basedir, f"{casestr}_TEST.h5"),
        os.path.join(basedir, f"{casestr}_test.h5"),
        os.path.join(basedir, f"{casestr}_TEST.hdf5"),
        os.path.join(basedir, f"{casestr}_test.hdf5"),
    )
    va_path = _first_existing(
        os.path.join(basedir, f"{casestr}_VAL.h5"),
        os.path.join(basedir, f"{casestr}_val.h5"),
        os.path.join(basedir, f"{casestr}_VAL.hdf5"),
        os.path.join(basedir, f"{casestr}_val.hdf5"),
    )
    if tr_path is None or te_path is None:
        raise FileNotFoundError(f"Could not locate train/test files for casestr='{casestr}' in '{basedir}'")
    td_tr = load_dataset(tr_path, input_scaler, output_scaler)
    td_te = load_dataset(te_path, input_scaler, output_scaler)
    if va_path is not None:
        td_va = load_dataset(va_path, input_scaler, output_scaler)
    else:
        # No explicit VAL: split a small validation out of training
        g = torch.Generator().manual_seed(seed)
        td_tr, td_va = td_tr.get_splits([0.85, 0.15], shuffle=True, generator=g, return_views=True)

    # Hyperparameters (from hparams.json when present)
    hp = {}
    hp_path = os.path.join(resudir, 'hparams.json')
    try:
        if os.path.exists(hp_path):
            with open(hp_path, 'r') as f:
                hp = json.load(f) or {}
    except Exception:
        hp = {}
    n_layers = int(hp.get('n_layers', 6))
    hidden_size = int(hp.get('hidden_size', 129))
    p_drop = float(hp.get('p_dropouts', 0.15))
    epochs = int(hp.get('epochs', 50))
    lr = float(hp.get('lr', 8.38e-04))
    lr_gamma = float(hp.get('lr_gamma', 0.99))
    batch_size = int(hp.get('batch_size', getattr(args, 'batch_size', 119)))

    # Infer input feature dimension from a sample (robust across datasets)
    try:
        x0 = td_tr[0][0]
        n_in = int(x0.shape[-1]) if hasattr(x0, 'shape') and x0.ndim > 0 else 1
    except Exception:
        n_in = 4  # fallback to legacy

    model = pyLOM.NN.MLP(
        input_size=n_in, output_size=1,
        hidden_size=hidden_size, n_layers=n_layers, p_dropouts=p_drop,
        model_name=f"mlp_{mode_tag}",
    )

    training_params = {
        'epochs': epochs,
        'lr_scheduler_step': 1,
        'optimizer_class': torch.optim.Adam,
        'loss_fn': torch.nn.MSELoss(),
        'print_rate_epoch': 5,
        'num_workers': 0,
        'device': device,
        'lr': lr,
        'lr_gamma': lr_gamma,
        'batch_size': batch_size,
        'ddp': ddp,
        'ddp_backend': ('nccl' if torch.cuda.is_available() else 'gloo'),
        'save_logs_path': resudir,
    }
    if _is_main_process():
        try:
            print(f"pyLOM NN: DDP backend = {training_params['ddp_backend']}")
        except Exception:
            pass

    pipeline = pyLOM.NN.Pipeline(train_dataset=td_tr, test_dataset=td_te, valid_dataset=td_va,
                                  model=model, training_params=training_params)
    logs = pipeline.run()
    if _is_main_process():
        print(f"Training summary: {pyLOM.NN.MLP.summarize_results(logs)}")

    # Persist model and scalers
    if _is_main_process():
        pipeline.model.save(os.path.join(resudir, 'model.pth'))
        input_scaler.save(os.path.join(resudir, 'input_scaler.json'))
        output_scaler.save(os.path.join(resudir, 'output_scaler.json'))

    # Predictions on test set, inverse transform, save arrays
    if _is_main_process():
        preds = model.predict(td_te, batch_size=2048)
        scaled_preds = output_scaler.inverse_transform(preds)
        scaled_y = output_scaler.inverse_transform(td_te[:][1])
        np.save(os.path.join(resudir, f'scaled_preds_{mode_tag}.npy'), scaled_preds)
        np.save(os.path.join(resudir, 'scaled_y.npy'), scaled_y)
        # Eval
        evalr = pyLOM.NN.RegressionEvaluator()
        evalr(scaled_y, scaled_preds)
        evalr.print_metrics()
        if not args.no_figures:
            true_vs_pred_plot(scaled_y, scaled_preds, os.path.join(resudir, f'true_vs_pred_{mode_tag}.png'), mode=mode_tag)
        pyLOM.cr_info()

if __name__ == '__main__':
    main()
