import numpy as np
import torch
import pyLOM, pyLOM.NN

import matplotlib.pyplot as plt

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
inscaler = 'out/scaler_'
ouscaler = 'out/scaler_pod.json'
shreds   = 'out/shred_'

# SHRED sensor configurations for uncertainty quantification
nconfigs    = 1 

## Import sensor measurements
# Training
data_trai = pyLOM.Dataset.load('sensors_trai.h5')
sens_trai = data_trai[sensvar].astype(np.float32)
nsens     = sens_trai.shape[0]
mask_trai = data_trai.get_variable('mask')
# Validation
data_vali = pyLOM.Dataset.load('sensors_vali.h5')
sens_vali = data_vali[sensvar].astype(np.float32)
mask_vali = data_vali.get_variable('mask')
# Test
data_test = pyLOM.Dataset.load('sensors_test.h5')
sens_test = data_test[sensvar].astype(np.float32)
mask_test = data_test.get_variable('mask')
# Compute total timesteps
ntimeG    = np.max(np.hstack((mask_trai,mask_vali,mask_test)))+1

## Import POD coefficients and rescale them.
# Training
pod_train   = pyLOM.POD.load('POD_trai_%s.h5' % podvar, vars='V')[0].astype(np.float32)
pod_scaler  = pyLOM.NN.MinMaxScaler()
pod_scaler.fit(pod_train.T)
#pod_scaler.save(ouscaler)
trai_out    = pod_scaler.transform(pod_train.T).T
output_size = trai_out.shape[0]
# Validation
pod_vali   = pyLOM.POD.load('POD_vali_%s.h5' % podvar, vars='V')[0].astype(np.float32)
vali_out   = pod_scaler.transform(pod_vali.T).T

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
    shred.fit(train_dataset, valid_dataset, epochs=1500, patience=100, verbose=False)
    shred.save('%s%i' % (shreds,kk), scalpath, mysensors)

pyLOM.cr_info()
