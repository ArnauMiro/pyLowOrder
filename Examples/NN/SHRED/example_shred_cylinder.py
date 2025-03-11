import numpy as np
import torch
import sys
import pyLOM, pyLOM.NN

sys.path.append('/gpfs/scratch/bsc21/bsc021893/parametrize/SHRED-ROM') ## Maybe change for pySHRED or add directly in pyLOM?
from utils.processdata     import Padding, TimeSeriesDataset

## Set device
device = pyLOM.NN.select_device() # Force CPU for this example, if left in blank it will automatically select the device

def split_reconstruct(Nt):
    ## Splitting into train, test and validation for reconstruction mode of SHRED
    tridx       = np.sort(np.random.choice(Nt, size=2500, replace=False))
    mask        = np.ones(Nt)
    mask[tridx] = 0
    vate_idx    = np.arange(0, Nt)[np.where(mask!=0)[0]]
    vaidx       = vate_idx[::2]
    teidx       = vate_idx[1::2]
    return tridx, vaidx, teidx

## Input parameters
# Data paths
podpath  = '/gpfs/scratch/bsc21/bsc021893/parametrize/latent_noncompiled_fp32_3_64_5.00e-03.npy' # Path to reduced POD basis
senspath = '/gpfs/scratch/bsc21/bsc021893/parametrize/data_points.npz'                           # Path to sensor measurements
sensvar  = 'velox'                                                                               # Variable from the sensor measurements we'll be working with

# SHRED sensor configurations for uncertainty quantification
sensxconfig = 3  # number of snesors per configuration
nconfigs    = 3 # number of configurations

# SHRED parameters (ask!!)
lags   = 50

## Import sensor measurements
data_points = np.load(senspath, allow_pickle=True)
sens_vals   = data_points[sensvar] # shape (m, Nt)
nsens       = sens_vals.shape[0]
Nt          = sens_vals.shape[1]

## Split between train, test and validation: TODO: In the POD part
tridx, vaidx, teidx = split_reconstruct(Nt)

## Import POD coefficients. Stack and rescale them
pod_coeff   = np.load(podpath).T
Nmodes      = [pod_coeff.shape[0]]
stacked_pod = pod_coeff[:sum(Nmodes)].T.reshape(1, Nt, sum(Nmodes))
pod_scaler  = pyLOM.NN.MinMaxScaler()
pod_scaler.fit(stacked_pod.reshape(-1, sum(Nmodes)))
rescaled_pod = pod_scaler.transform(stacked_pod.reshape(-1, sum(Nmodes))).reshape(stacked_pod.shape)
data_out     = Padding(torch.from_numpy(rescaled_pod), 1).squeeze(1).to(device)
output_size  = data_out.shape[-1]

## Build shred architecture
shred   = pyLOM.NN.SHRED(sensxconfig, output_size, device, nsens, nconfigs=nconfigs)

## Fit all SHREDs for uncertainty quantification
for kk, mysensors in enumerate(shred.configs):
    # Get the values and scale them
    myvalues = sens_vals[mysensors,:].T
    myscaler = pyLOM.NN.MinMaxScaler()
    scalpath = 'out/scaler_%i.json' % kk
    myscaler.fit(myvalues)
    myscaler.save(scalpath)
    vals_config = myscaler.transform(myvalues)[np.newaxis,:,:]
    data_in = Padding(torch.from_numpy(vals_config), lags).to(device)
    # Generate training validation and test datasets both for reconstruction of states
    train_dataset = TimeSeriesDataset(data_in[tridx], data_out[tridx])
    valid_dataset = TimeSeriesDataset(data_in[vaidx], data_out[vaidx])
    test_dataset  = TimeSeriesDataset(data_in[teidx], data_out[teidx])
    # Fit SHRED
    shred.fit(train_dataset, valid_dataset, batch_size=64, epochs=500, lr=1e-3, verbose=False, patience=100)
    shred.save('out/shred_%i' % kk, scalpath, mysensors)

pyLOM.cr_info()