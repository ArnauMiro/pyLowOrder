import numpy as np
import torch
import pyLOM, pyLOM.NN

## Set plot styling
pyLOM.style_plots(legend_fsize=14)

## Set device
device = pyLOM.NN.select_device() # Force CPU for this example, if left in blank it will automatically select the device

## Input parameters
# Data paths
sensvar = 'VELOX'              # Variable from the sensor measurements we'll be working with
podvar  = 'VELOX'              # Variable from the POD we'll be working with
shreds  = 'out/shreds/config_' # Where the SHREDS models are saved

# Select which sensor configurations have to be postprocessed
nconfigs  = 20
configIDs = np.arange(nconfigs, dtype=int)

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
S, pod_trai = pyLOM.POD.load('POD_trai_%s.h5' % podvar, vars=['S','V'])
pod_trai    = pod_trai.astype(np.float32)
Sscale      = S/np.sum(S)
output_size = pod_trai.shape[0]
# Validation
pod_vali = pyLOM.POD.load('POD_vali_%s.h5' % podvar, vars='V')[0].astype(np.float32)
# Test
pod_test = pyLOM.POD.load('POD_test_%s.h5' % podvar, vars='V')[0].astype(np.float32)
# Full POD
full_pod = np.zeros((output_size,ntimeG), dtype=pod_trai.dtype)
full_pod[:,mask_trai] = pod_trai
full_pod[:,mask_vali] = pod_vali
full_pod[:,mask_test] = pod_test

## Build SHRED architecture
shred = pyLOM.NN.SHRED(output_size, device, nsens)

## Load all SHRED configurations and evaluate them using the data from the corresponding sensors
outres = np.zeros((output_size, ntimeG, nconfigs), dtype=full_pod.dtype)
MRE    = np.zeros((output_size, nconfigs), dtype=full_pod.dtype)
for kk in range(nconfigs):
    # Load the SHRED model for this configuration
    load_dict = torch.load('%s%i.pth' % (shreds, kk))
    mysensors = load_dict['sensors']
    shred.load_state_dict(load_dict['model_state_dict'])
    # Get the sensor values for the training, validation and test
    mytrai = sens_trai[mysensors,:]
    myvali = sens_vali[mysensors,:]
    mytest = sens_test[mysensors,:]
    # Load the appropiate scaler
    myscaler = pyLOM.NN.MinMaxScaler()
    myscaler = myscaler.load(load_dict['scaler_path'])
    trai_sca = myscaler.transform(mytrai.T).T
    vali_sca = myscaler.transform(myvali.T).T
    test_sca = myscaler.transform(mytest.T).T
    # Concatenate train, test and validation data to generate the embeddings correctly
    embedded = np.zeros((trai_sca.shape[0],ntimeG), dtype=trai_sca.dtype)
    embedded[:,mask_trai] = trai_sca
    embedded[:,mask_vali] = vali_sca
    embedded[:,mask_test] = test_sca
    delayed = pyLOM.math.time_delay_embedding(embedded, dimension=50)
    # Generate training validation and test datasets both for reconstruction of states
    output = shred(torch.from_numpy(delayed).permute(1,2,0).to(device)).cpu().detach().numpy()
    # Load the POD scaler of that SHRED to transform POD data
    podscale = pyLOM.NN.MinMaxScaler()
    podscale = podscale.load(load_dict['podscale_path'])
    outres[:,:,kk] = podscale.inverse_transform(output).T
    # Compute mean relative error
    MRE[:,kk] = pyLOM.math.columnwise_mre(full_pod, outres[:,:,kk])

## Compute reconstruction statistics
meanout = np.mean(outres, axis=2)
stdout  = np.std(outres, axis=2)
meanMRE = np.mean(MRE, axis=1)
stdMRE  = np.std(MRE, axis=1)

print('The mean MRE of the %i configurations is %.2f' % (nconfigs, np.mean(meanMRE)))

## Plot error bars
#Non-scaled
fig, _ = pyLOM.utils.plotModalErrorBars(meanMRE)
fig.savefig('errorbars.pdf', dpi=300, bbox_inches='tight')
#Scaled
fig, _ = pyLOM.utils.plotModalErrorBars(Sscale*meanMRE)
fig.savefig('errorbars_scaled.pdf', dpi=300, bbox_inches='tight')

## Plot POD modes reconstruction
fig, _ = pyLOM.utils.plotTimeSeries(time, full_pod, meanout, stdout)
fig.savefig('output_modes.pdf', dpi=600)

pyLOM.cr_info()