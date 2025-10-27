#!/usr/bin/env python
#
# Example of MLP.
#
# Last revision: 14/11/2024

import os, numpy as np, torch, matplotlib.pyplot as plt
try:
    plt.switch_backend("Agg")
except Exception:
    pass
import torch.distributed as dist
import argparse
import pyLOM, pyLOM.NN

seed = 19

def _is_main_process():
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    # Fallback to env vars before dist init (torchrun/mpi/slurm)
    for k in ("RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v) == 0
            except Exception:
                break
    return True

def load_dataset(fname,inputs_scaler,outputs_scaler):
    '''
    Auxiliary function to load a dataset into a pyLOM
    NN dataset
    '''
    return pyLOM.NN.Dataset.load(
        fname,
        field_names=["CP"],
        add_mesh_coordinates=True,
        variables_names=["AoA", "Mach"],
        inputs_scaler=inputs_scaler,
        outputs_scaler=outputs_scaler,
    )

def print_dset_stats(name,td):
    '''
    Auxiliary function to print the statistics
    of a NN dataset
    '''
    x, y = next(iter(torch.utils.data.DataLoader(td, batch_size=len(td))))
    pyLOM.pprint(0,f'name={name} ({len(td)}), x ({x.shape}) = [{x.min(dim=0)},{x.max(dim=0)}], y ({y.shape}) = [{y.min(dim=0)},{y.max(dim=0)}]')

def true_vs_pred_plot(y_true, y_pred, path, mode: str = None):
    """
    Auxiliary function to plot the true vs predicted values
    """
    # Coerce to numpy arrays (N, C)
    try:
        if hasattr(y_true, 'detach'):
            y_true_np = y_true.detach().cpu().numpy()
        else:
            y_true_np = np.asarray(y_true)
        if hasattr(y_pred, 'detach'):
            y_pred_np = y_pred.detach().cpu().numpy()
        else:
            y_pred_np = np.asarray(y_pred)
        if y_true_np.ndim == 1:
            y_true_np = y_true_np.reshape(-1, 1)
        if y_pred_np.ndim == 1:
            y_pred_np = y_pred_np.reshape(-1, 1)
    except Exception:
        return
    num_plots = y_true_np.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):
        plt.subplot(num_plots, 1, j + 1)
        yt = y_true_np[:, j].ravel()
        yp = y_pred_np[:, j].ravel()
        # Scatter
        plt.scatter(yt, yp, s=1, c="b", alpha=0.5, label="Data")
        # Regression line yp ≈ a*yt + b
        try:
            a, b = np.polyfit(yt, yp, 1)
            y_fit = a * yt + b
            ss_res = np.sum((yp - y_fit) ** 2)
            ss_tot = np.sum((yp - np.mean(yp)) ** 2) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            x_line = np.linspace(yt.min(), yt.max(), 100)
            y_line = a * x_line + b
            plt.plot(x_line, y_line, 'r-', lw=2, label=f"Reg: y={a:.3f}x+{b:.3f}, R2={r2:.4f}")
        except Exception:
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            # If fit fails, still show RMSE
            pass
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        mode_str = f" - Mode: {mode.upper()}" if mode else ""
        plt.title(f"Scatterplot for Component {j+1}{mode_str} (RMSE={rmse:.4e})")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"Saved: {path}", flush=True)

## removed unused plot_train_test_loss (kept only minimal plotting utilities)

## removed unused plot_compare_test_loss (replaced by all-in-one figure)

## removed unused plot_compare_curves (replaced by all-in-one figure)

def plot_compare_all_in_one(res_single_path, res_ddp_path, out_path):
    """
    Plot all four curves (Single-Train, Single-Test, DDP-Train, DDP-Test)
    overlaid in a single axes for quick comparison.
    """
    try:
        rs = np.load(res_single_path, allow_pickle=True).item()
        rd = np.load(res_ddp_path, allow_pickle=True).item()
        tr_s = rs.get('train_loss', [])
        tr_d = rd.get('train_loss', [])
        ts_s = rs.get('test_loss', [])
        ts_d = rd.get('test_loss', [])
        plt.figure(figsize=(8, 5))
        if len(tr_s) > 0:
            plt.plot(range(1, len(tr_s)+1), tr_s, label='Single - Train', linestyle='-')
        if len(ts_s) > 0:
            plt.plot(range(1, len(ts_s)+1), ts_s, label='Single - Test', linestyle='--')
        if len(tr_d) > 0:
            plt.plot(range(1, len(tr_d)+1), tr_d, label='DDP - Train', linestyle='-')
        if len(ts_d) > 0:
            plt.plot(range(1, len(ts_d)+1), ts_d, label='DDP - Test', linestyle='--')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss (Single vs DDP)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        # no-op
    except Exception as e:
        print(f"Could not create all-in-one curves plot: {e}")

def generate_comparisons(res_dir: str):
    res_single = os.path.join(res_dir, 'training_results_mlp_single.npy')
    res_ddp    = os.path.join(res_dir, 'training_results_mlp_ddp.npy')
    exists_single = os.path.exists(res_single)
    exists_ddp = os.path.exists(res_ddp)
    if exists_single and exists_ddp:
        # Single all-in-one figure
        try:
            out3 = os.path.join(res_dir, 'curves_all_in_one_single_vs_ddp.png')
            plot_compare_all_in_one(res_single, res_ddp, out3)
        except Exception:
            pass
        # JSON report
        try:
            rs = np.load(res_single, allow_pickle=True).item()
            rd = np.load(res_ddp, allow_pickle=True).item()
            # No external sidecars; rely on values embedded in NPY (cr_total_time_s, cr_channels)
            preds_s_path = os.path.join(res_dir, 'scaled_preds_single.npy')
            preds_d_path = os.path.join(res_dir, 'scaled_preds_ddp.npy')
            y_path = os.path.join(res_dir, 'scaled_y.npy')
            rmse_s = rmse_d = None
            if os.path.exists(y_path) and os.path.exists(preds_s_path):
                y_true_np = np.load(y_path)
                y_pred_s = np.load(preds_s_path)
                rmse_s = float(np.sqrt(np.mean((y_pred_s - y_true_np) ** 2)))
            if os.path.exists(y_path) and os.path.exists(preds_d_path):
                y_true_np = np.load(y_path)
                y_pred_d = np.load(preds_d_path)
                rmse_d = float(np.sqrt(np.mean((y_pred_d - y_true_np) ** 2)))
            report_path = os.path.join(res_dir, 'comparison_report.json')
            pyLOM.NN.MLP.write_comparison_report(rs, rd, report_path, rmse_single=rmse_s, rmse_ddp=rmse_d, notes='Auto-generated comparison report')

            # Bar chart comparing timing/throughput and final loss (uses CR timing when available)
            try:
                import matplotlib.pyplot as _plt
                # Timing (prefer CR times to avoid inconsistencies)
                def get_total_time(d):
                    if 'cr_total_time_s' in d and d['cr_total_time_s'] is not None:
                        return float(d['cr_total_time_s'])
                    et = d.get('epoch_time_s', [])
                    return float(np.sum(et)) if len(et) else None
                def get_avg_epoch(d):
                    et = d.get('epoch_time_s', [])
                    return float(np.mean(et)) if len(et) else None
                def get_avg_thr(d):
                    thr = d.get('throughput_samples_per_sec_global', [])
                    return float(np.mean(thr)) if len(thr) else None
                def get_final_test(d):
                    tl = d.get('test_loss', [])
                    return float(tl[-1]) if len(tl) else None

                total_time_s = get_total_time(rs)
                total_time_d = get_total_time(rd)
                avg_epoch_s = get_avg_epoch(rs)
                avg_epoch_d = get_avg_epoch(rd)
                avg_thr_s = get_avg_thr(rs)
                avg_thr_d = get_avg_thr(rd)
                final_test_s = get_final_test(rs)
                final_test_d = get_final_test(rd)

                labels = ['Single', 'DDP']
                fig, axes = _plt.subplots(2, 2, figsize=(10, 7))

                # Total time (lower is better). Speedup = single/ddp
                bars00 = axes[0,0].bar(labels, [total_time_s, total_time_d], color=['tab:blue', 'tab:orange'])
                axes[0,0].set_title('Total Training Time (s)')
                axes[0,0].grid(True, axis='y', linestyle='--', alpha=0.4)
                if total_time_s and total_time_d and total_time_d > 0:
                    sp = total_time_s / total_time_d
                    axes[0,0].text(0.5, 0.95, f"Speedup DDP: ×{sp:.2f}", transform=axes[0,0].transAxes, ha='center', va='top', fontsize=9)

                # Avg epoch time (lower is better). Speedup = single/ddp
                bars01 = axes[0,1].bar(labels, [avg_epoch_s, avg_epoch_d], color=['tab:blue', 'tab:orange'])
                axes[0,1].set_title('Avg Epoch Time (s)')
                axes[0,1].grid(True, axis='y', linestyle='--', alpha=0.4)
                if avg_epoch_s and avg_epoch_d and avg_epoch_d > 0:
                    sp = avg_epoch_s / avg_epoch_d
                    axes[0,1].text(0.5, 0.95, f"Speedup DDP: ×{sp:.2f}", transform=axes[0,1].transAxes, ha='center', va='top', fontsize=9)

                # Avg throughput (higher is better). Speedup = ddp/single
                bars10 = axes[1,0].bar(labels, [avg_thr_s, avg_thr_d], color=['tab:blue', 'tab:orange'])
                axes[1,0].set_title('Avg Throughput (samples/s)')
                axes[1,0].grid(True, axis='y', linestyle='--', alpha=0.4)
                if avg_thr_s and avg_thr_d and avg_thr_s > 0:
                    sp = avg_thr_d / avg_thr_s
                    axes[1,0].text(0.5, 0.95, f"Speedup DDP: ×{sp:.2f}", transform=axes[1,0].transAxes, ha='center', va='top', fontsize=9)

                # Final test loss, annotate RMSE
                bars = axes[1,1].bar(labels, [final_test_s, final_test_d], color=['tab:blue', 'tab:orange'])
                axes[1,1].set_title('Final Test Loss')
                axes[1,1].grid(True, axis='y', linestyle='--', alpha=0.4)
                for bar, txt in zip(bars, [rmse_s, rmse_d]):
                    if txt is not None:
                        h = bar.get_height()
                        axes[1,1].text(bar.get_x() + bar.get_width()/2, h, f"RMSE={txt:.3e}", ha='center', va='bottom', fontsize=8)
                if final_test_s and final_test_d and final_test_d > 0:
                    # Improvement factor (>1 means DDP lower loss)
                    imp = final_test_s / final_test_d
                    axes[1,1].text(0.5, 0.95, f"Factor DDP (loss): ×{imp:.2f}", transform=axes[1,1].transAxes, ha='center', va='top', fontsize=9)

                fig.tight_layout()
                fig.savefig(os.path.join(res_dir, 'comparison_bars.png'), dpi=300)
                _plt.close(fig)
            except Exception:
                pass
        except Exception:
            pass

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", dest="ddp_mode", default="off", choices=["on","off"], help="DDP mode: on|off")
    parser.add_argument("--resudir", dest="resudir", default="MLP_DLR_airfoil", help="Results output directory")
    parser.add_argument("--basedir", dest="basedir", default="/home/airbus/CETACEO_cp_interp/DATA/DLR_pylom/", help="Base directory of datasets")
    parser.add_argument("--casestr", dest="casestr", default="NRL7301", help="Case string prefix for dataset files")
    parser.add_argument("--postprocess-only", dest="postprocess_only", action="store_true", help="Only generate comparison plots/report from existing NPY files and exit")
    return parser.parse_known_args()[0]

# Parse CLI/env before any path-dependent work
args = _parse_args()
ddp_enabled = (args.ddp_mode == "on")
print(f"pyLOM NN: DDP mode is {'ON' if ddp_enabled else 'OFF'}")

# Resolve results directory from CLI/env
resudir_env = os.environ.get("PYLOM_RESUDIR", "").strip()
RESUDIR = resudir_env if len(resudir_env) > 0 else args.resudir
if _is_main_process():
    print(f"pyLOM NN: Results directory is {RESUDIR}")
pyLOM.NN.create_results_folder(RESUDIR)

# Fast path: only postprocess (no training)
if args.postprocess_only and _is_main_process():
    generate_comparisons(RESUDIR)
    import sys
    sys.exit(0)

# Dataset location
BASEDIR = args.basedir
CASESTR = args.casestr

## Set device
device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Load datasets and set up the results output

input_scaler  = pyLOM.NN.MinMaxScaler()
output_scaler = pyLOM.NN.MinMaxScaler()
td_train = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_TRAIN.h5'),input_scaler,output_scaler)
td_test  = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_TEST.h5'),input_scaler,output_scaler)
td_val   = load_dataset(os.path.join(BASEDIR,f'{CASESTR}_VAL.h5'),input_scaler,output_scaler)

# if we want to split by flight conditions instead of using the provided split, we can do the following
dataset = td_train + td_test + td_val
generator = torch.Generator().manual_seed(seed) # set seed for reproducibility
td_train, td_test, td_val = dataset.get_splits([0.7, 0.15, 0.15], shuffle=True, generator=generator, return_views=True)

if _is_main_process():
    print_dset_stats('train',td_train)
    print_dset_stats('test', td_test)
    print_dset_stats('val',  td_val)

## functions moved to top: _is_main_process and _parse_args

mode_tag = 'ddp' if ddp_enabled else 'single'

model = pyLOM.NN.MLP(
    input_size=4, # x, y, AoA, Mach
    output_size=1, # CP
    hidden_size=129,
    n_layers=6,
    p_dropouts=0.15,
    model_name=f"mlp_{mode_tag}",
)

training_params = {
    "epochs": 5,
    'lr_scheduler_step': 1,
    "optimizer_class": torch.optim.Adam,
    "loss_fn": torch.nn.MSELoss(),
    "print_rate_epoch": 5,
    "num_workers": 0,
    "device": device,
    "lr": 0.000838, 
    "lr_gamma": 0.99, 
    "batch_size": 119,
    # DDP switch: auto|on|off controlled by CLI/env; here resolved to boolean
    "ddp": ddp_enabled,
    "save_logs_path": RESUDIR,
}

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    valid_dataset=td_val,
    model=model,
    training_params=training_params,
)
training_logs = pipeline.run()

# Print a compact training summary (mode, final losses, throughput) only on main process
if _is_main_process():
    summary = pyLOM.NN.MLP.summarize_results(training_logs)
    print(f"Training summary: {summary}")

# check saving and loading the model
if _is_main_process():
    pipeline.model.save(os.path.join(RESUDIR,"model.pth"))
    model = pyLOM.NN.MLP.load(RESUDIR + "/model.pth")

# check saving and loading the scalers
if _is_main_process():
    input_scaler.save(os.path.join(RESUDIR,"input_scaler.json"))
    output_scaler.save(os.path.join(RESUDIR,"output_scaler.json"))
    input_scaler = pyLOM.NN.MinMaxScaler.load(os.path.join(RESUDIR,"input_scaler.json"))
    output_scaler = pyLOM.NN.MinMaxScaler.load(os.path.join(RESUDIR,"output_scaler.json"))

# to predict from a dataset
if _is_main_process():
    preds = model.predict(td_test, batch_size=2048)

# to predict from a tensor
# preds = model(torch.tensor(dataset_test[:][0], device=model.device)).cpu().detach().numpy()

if _is_main_process():
    scaled_preds = output_scaler.inverse_transform(preds)
    scaled_y     = output_scaler.inverse_transform(td_test[:][1])
    # Save predictions/targets for cross-run comparison if needed
    np.save(os.path.join(RESUDIR, f"scaled_preds_{mode_tag}.npy"), scaled_preds)
    np.save(os.path.join(RESUDIR, f"scaled_y.npy"), scaled_y)

# check that the scaling is correct
if _is_main_process():
    pass

if _is_main_process():
    evaluator = pyLOM.NN.RegressionEvaluator()
    evaluator(scaled_y, scaled_preds)
    evaluator.print_metrics()

    true_vs_pred_plot(scaled_y, scaled_preds, os.path.join(RESUDIR, f'true_vs_pred_{mode_tag}.png'), mode=mode_tag)
    # No CR sidecars: timing comes embedded in NPY via pyLOM.nn (cr_total_time_s)
    # Generate a single curves plot overlaying the 4 curves (train/test, single/DDP)
    generate_comparisons(RESUDIR)

    pyLOM.cr_info()

# No extra hooks; comparisons already attempted above
