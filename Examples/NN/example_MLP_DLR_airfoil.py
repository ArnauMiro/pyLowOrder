#!/usr/bin/env python
#
# Example of MLP.
#
# Last revision: 14/11/2024

import os, numpy as np, torch, matplotlib.pyplot as plt
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

def true_vs_pred_plot(y_true, y_pred, path):
    """
    Auxiliary function to plot the true vs predicted values
    """
    num_plots = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):
        plt.subplot(num_plots, 1, j + 1)
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.5)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"Scatterplot for Component {j+1}")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=300)

def plot_train_test_loss(train_loss, test_loss, path):
    """
    Auxiliary function to plot the training and test loss
    """
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    total_epochs = len(test_loss) # test loss is calculated at the end of each epoch
    total_iters = len(train_loss) # train loss is calculated at the end of each iteration/batch
    iters_per_epoch = total_iters // total_epochs
    plt.plot(np.arange(iters_per_epoch, total_iters+1, step=iters_per_epoch), test_loss, label="Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=300)

def plot_compare_test_loss(res_single_path, res_ddp_path, out_path):
    """
    Plot an overlay of test loss per epoch for single vs DDP runs
    loading the saved numpy result files.
    """
    try:
        rs = np.load(res_single_path, allow_pickle=True).item()
        rd = np.load(res_ddp_path, allow_pickle=True).item()
        ts_s = rs.get('test_loss', [])
        ts_d = rd.get('test_loss', [])
        plt.figure()
        if len(ts_s) > 0:
            plt.plot(range(1, len(ts_s)+1), ts_s, label='Single - Test Loss')
        if len(ts_d) > 0:
            plt.plot(range(1, len(ts_d)+1), ts_d, label='DDP - Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Test Loss Comparison (Single vs DDP)')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.savefig(out_path, dpi=300)
    except Exception as e:
        print(f"Could not create comparison plot: {e}")

## Set device
device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device


## Load datasets and set up the results output
BASEDIR = '/home/airbus/CETACEO_cp_interp/DATA/DLR_pylom/'
CASESTR = 'NRL7301'

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

## functions moved to top: _detect_ddp and _is_main_process

# Manual DDP mode control via CLI/env
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", dest="ddp_mode", default="off", choices=["on","off"], help="DDP mode: on|off")
    parser.add_argument("--resudir", dest="resudir", default="MLP_DLR_airfoil", help="Results output directory")
    return parser.parse_known_args()[0]

args = _parse_args()
ddp_mode_env = os.environ.get("PYLOM_DDP", "").strip().lower()
ddp_mode = ddp_mode_env if ddp_mode_env in ("on","off") else args.ddp_mode
ddp_enabled = (ddp_mode == "on")
print(f"pyLOM NN: DDP mode is {'ON' if ddp_enabled else 'OFF'}")

# Resolve results directory from CLI/env
resudir_env = os.environ.get("PYLOM_RESUDIR", "").strip()
RESUDIR = resudir_env if len(resudir_env) > 0 else args.resudir
if _is_main_process():
    print(f"pyLOM NN: Results directory is {RESUDIR}")
pyLOM.NN.create_results_folder(RESUDIR)

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
    "epochs": 50,
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
    print(scaled_y.min(), scaled_y.max())

if _is_main_process():
    evaluator = pyLOM.NN.RegressionEvaluator()
    evaluator(scaled_y, scaled_preds)
    evaluator.print_metrics()

    true_vs_pred_plot(scaled_y, scaled_preds, os.path.join(RESUDIR, f'true_vs_pred_{mode_tag}.png'))
    plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], os.path.join(RESUDIR, f'train_test_loss_{mode_tag}.png'))

    # If both runs exist, generate comparison plot of test loss
    res_single = os.path.join(RESUDIR, 'training_results_mlp_single.npy')
    res_ddp    = os.path.join(RESUDIR, 'training_results_mlp_ddp.npy')
    if os.path.exists(res_single) and os.path.exists(res_ddp):
        plot_compare_test_loss(res_single, res_ddp, os.path.join(RESUDIR, 'test_loss_compare_single_vs_ddp.png'))

    pyLOM.cr_info()
    plt.show()
