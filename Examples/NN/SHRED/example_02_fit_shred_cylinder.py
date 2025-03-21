import numpy as np
import torch
import pyLOM, pyLOM.NN

pyLOM.style_plots(legend_fsize=14)

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
nconfigs = 20

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
pod_scaler  = pyLOM.NN.MinMaxScaler(column=True)
pod_scaler.fit(pod_trai)
pod_scaler.save(outscale)
trai_out    = pod_scaler.transform(pod_trai)
output_size = trai_out.shape[0]
Sscale      = S/np.sum(S)
# Validation
pod_vali = pyLOM.POD.load('POD_vali_%s.h5' % podvar, vars='V')[0].astype(np.float32)
vali_out = pod_scaler.transform(pod_vali)
# Test
pod_test = pyLOM.POD.load('POD_test_%s.h5' % podvar, vars='V')[0].astype(np.float32)
test_out = pod_scaler.transform(pod_test)
# Full POD
full_pod = np.zeros((output_size,ntimeG), dtype=pod_trai.dtype)
full_pod[:,mask_trai] = pod_trai
full_pod[:,mask_vali] = pod_vali
full_pod[:,mask_test] = pod_test

## Build SHRED architecture
shred   = pyLOM.NN.SHRED(output_size, device, nsens, nconfigs=nconfigs)

## Fit all SHRED configurations using the data from the sensors
for kk, mysensors in enumerate(shred.configs):
    # Get the sensor values for the training, validation and test
    mytrai = sens_trai[mysensors,:]
    myvali = sens_vali[mysensors,:]
    mytest = sens_test[mysensors,:]
    # Scale the data
    myscaler = pyLOM.NN.MinMaxScaler()
    scalpath = '%s%i.json' % (inscaler, kk)
    myscaler.fit(mytrai)
    myscaler.save(scalpath)
    trai_sca = myscaler.transform(mytrai)
    vali_sca = myscaler.transform(myvali)
    test_sca = myscaler.transform(mytest)
    # Concatenate train, test and validation data to generate the embeddings correctly
    embedded = np.zeros((trai_sca.shape[0],ntimeG), dtype=trai_sca.dtype)
    embedded[:,mask_trai] = trai_sca
    embedded[:,mask_vali] = vali_sca
    embedded[:,mask_test] = test_sca
    delayed = pyLOM.math.time_delay_embedding(embedded, dimension=50)
    # Generate training validation and test datasets both for reconstruction of states
    train_dataset = pyLOM.NN.Dataset(trai_out, variables_in=delayed[:,mask_trai,:]) 
    valid_dataset = pyLOM.NN.Dataset(vali_out, variables_in=delayed[:,mask_vali,:])
    # Fit SHRED
    shred.fit(train_dataset, valid_dataset, epochs=1500, patience=100, verbose=False, mod_scale=torch.tensor(Sscale))
    shred.save('%s%i' % (shreds,kk), scalpath, outscale, mysensors)

pyLOM.cr_info()