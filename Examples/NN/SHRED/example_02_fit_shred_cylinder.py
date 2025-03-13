import numpy as np
import torch
import pyLOM, pyLOM.NN

import matplotlib.pyplot as plt

pyLOM.style_plots(legend_fsize=14)

class TimeSeriesDatasetMine(torch.utils.data.Dataset):
    '''
    Input: sequence of input measurements with shape (ntrajectories, ntimes, ninput) and corresponding measurements of high-dimensional state with shape (ntrajectories, ntimes, noutput)
    Output: Torch dataset
    '''

    def __init__(self, X, Y):
        self.X = torch.tensor(X).permute(1,2,0)
        self.Y = torch.tensor(Y).T
        self.len = X.shape[1]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len

## Set device
device = pyLOM.NN.select_device() # Force CPU for this example, if left in blank it will automatically select the device

## Input parameters
# Data paths
sensvar  = 'VELOX'      # Variable from the sensor measurements we'll be working with
podvar   = 'VELOX'      # Variable from the POD we'll be working with

# Output paths
inscaler = 'out/scalers/config_'
outscale = 'out/scalers/pod.json'
shreds   = 'out/shreds/config_'

# SHRED sensor configurations for uncertainty quantification
nconfigs = 1

## Import sensor measurements
# Training
data_trai = pyLOM.Dataset.load('sensors_trai.h5')
sens_trai = data_trai[sensvar].astype(np.float32)
nsens     = sens_trai.shape[0]
mask_trai = data_trai.get_variable('mask')
time_trai = data_trai.get_variable('time')
# Validation
data_vali = pyLOM.Dataset.load('sensors_vali.h5')
sens_vali = data_vali[sensvar].astype(np.float32)
mask_vali = data_vali.get_variable('mask')
time_vali = data_vali.get_variable('time')
# Test
data_test = pyLOM.Dataset.load('sensors_test.h5')
sens_test = data_test[sensvar].astype(np.float32)
mask_test = data_test.get_variable('mask')
time_test = data_test.get_variable('time')
# Compute total timesteps
ntimeG = np.max(np.hstack((mask_trai,mask_vali,mask_test)))+1
time   = np.zeros((ntimeG,), dtype=time_trai.dtype)
time[mask_trai] = time_trai
time[mask_vali] = time_vali
time[mask_test] = time_test

## Import POD coefficients and rescale them.
# Training
mymodes     = np.array([0,1,3,4,5])
S, pod_trai = pyLOM.POD.load('POD_trai_%s.h5' % podvar, vars=['S','V'])
pod_trai    = pod_trai.astype(np.float32)
pod_scaler  = pyLOM.NN.MinMaxScaler()
pod_scaler.fit(pod_trai.T)
pod_scaler.save(outscale)
trai_out    = pod_scaler.transform(pod_trai.T).T
output_size = trai_out.shape[0]
Sscale      = S/np.sum(S)
# Validation
pod_vali = pyLOM.POD.load('POD_vali_%s.h5' % podvar, vars='V')[0].astype(np.float32)
vali_out = pod_scaler.transform(pod_vali.T).T
# Test
pod_test = pyLOM.POD.load('POD_test_%s.h5' % podvar, vars='V')[0].astype(np.float32)
test_out = pod_scaler.transform(pod_test.T).T
# Full POD
full_pod = np.zeros((output_size,ntimeG), dtype=pod_trai.dtype)
full_pod[:,mask_trai] = pod_trai
full_pod[:,mask_vali] = pod_vali
full_pod[:,mask_test] = pod_test

## Build SHRED architecture
shred   = pyLOM.NN.SHRED(output_size, device, nsens, nconfigs=nconfigs)

## Fit all SHRED configurations using the data from the sensors
for kk, mysensors in enumerate(shred.configs):
    # Get the sensor values for the training and validation
    mytrai = sens_trai[mysensors,:]
    myvali = sens_vali[mysensors,:]
    mytest = sens_test[mysensors,:]
    # Scale the data
    myscaler = pyLOM.NN.MinMaxScaler()
    scalpath = '%s%i.json' % (inscaler, kk)
    myscaler.fit(mytrai.T)
    myscaler.save(scalpath)
    trai_sca = myscaler.transform(mytrai.T).T
    vali_sca = myscaler.transform(myvali.T).T
    test_sca = myscaler.transform(mytest.T).T
    embedded = np.zeros((trai_sca.shape[0],ntimeG), dtype=trai_sca.dtype)
    embedded[:,mask_trai] = trai_sca
    embedded[:,mask_vali] = vali_sca
    embedded[:,mask_test] = test_sca
    delayed = pyLOM.math.time_delay_embedding(embedded, dimension=50)
    # Generate training validation and test datasets both for reconstruction of states
    train_dataset = TimeSeriesDatasetMine(delayed[:,mask_trai,:], trai_out) #TODO: use the pyLOM dataset or torch tensor dataset
    valid_dataset = TimeSeriesDatasetMine(delayed[:,mask_vali,:], vali_out) #TODO: use the pyLOM dataset
    # Fit SHRED
    shred.fit(train_dataset, valid_dataset, epochs=1500, patience=100, verbose=False, mod_scale=torch.tensor(Sscale))
    shred.save('%s%i' % (shreds,kk), scalpath, mysensors)

output = shred(torch.from_numpy(delayed).permute(1,2,0).to(device)).cpu().detach().numpy()
outres = pod_scaler.inverse_transform(output).T
MRE    = pyLOM.math.columnwise_mre(full_pod, outres)

# Plot error bars
indices = np.arange(len(MRE))+1
cmap    = plt.cm.jet
colors  = cmap(np.linspace(0.1, 0.9, len(MRE)))
fig, ax = plt.subplots(figsize=(20, 3))
bars = ax.bar(indices, Sscale*MRE, capsize=5, color=colors, edgecolor='black')
ax.set_xlabel("Rank", fontsize=14)
ax.set_ylabel("Average Relative Error", fontsize=14)
ax.set_xticks(indices[24::25])
ax.set_xticklabels([f"{i}" for i in indices[24::25]], fontsize=12)
ax.tick_params(axis='both', labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)
fig.savefig('errorbars_scale.pdf', dpi=300, bbox_inches='tight')

fig, axs = plt.subplots(output_size,1, figsize=(20, 24))
axs = axs.flatten()
for rr in range(len(axs)):
    if rr == 0:
        axs[rr].plot(time, outres[rr], 'r-.', label='SHRED')
        axs[rr].plot(time, full_pod[rr], 'b--', label='Original')
    else:
        axs[rr].plot(time, outres[rr], 'r-.')
        axs[rr].plot(time, full_pod[rr], 'b--')
    axs[rr].set_ylabel('Mode %i' % rr)
fig.legend()
fig.tight_layout()
fig.savefig('output_modes.pdf', dpi=600)


pyLOM.cr_info()
