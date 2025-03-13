import numpy as np
import torch
import pyLOM, pyLOM.NN

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
senspath = 'sensors.h5' # Path to sensor measurements
sensvar  = 'VELOX'      # Variable from the sensor measurements we'll be working with
podvar   = 'VELOX'      # Variable from the sensor measurements we'll be working with

# Output paths
inscaler = 'out/scaler_'
ouscaler = 'out/scaler_pod.json'
shreds   = 'out/shred_'

# SHRED sensor configurations for uncertainty quantification
nconfigs    = 1 

## Import sensor measurements
data_points = pyLOM.Dataset.load(senspath)
sens_vals   = data_points[sensvar].astype(np.float32) # shape (m, Nt)
nsens       = sens_vals.shape[0]
Nt          = sens_vals.shape[1]

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

## Import POD coefficients. Stack and rescale them.
pod_coeff   = pyLOM.POD.load('POD_modes_%s.h5' % podvar, vars='V')[0].astype(np.float32)
pod_scaler  = pyLOM.NN.MinMaxScaler()
pod_scaler.fit(pod_coeff)
pod_scaler.save(ouscaler)
data_out    = pod_scaler.transform(pod_coeff)
output_size = data_out.shape[0]

## Build SHRED architecture
shred   = pyLOM.NN.SHRED(output_size, device, nsens, nconfigs=nconfigs)

## Fit all SHRED configurations using the data from the sensors
for kk, mysensors in enumerate(shred.configs):
    # Get the values and scale them
    myvalues = sens_vals[mysensors,:]
    myscaler = pyLOM.NN.MinMaxScaler()
    scalpath = '%s%i.json' % (inscaler, kk)
    myscaler.fit(myvalues)
    myscaler.save(scalpath)
    vals_config = myscaler.transform(myvalues)[np.newaxis,:,:]
    rescaled = myscaler.transform(myvalues)
    data_in = pyLOM.math.time_delay_embedding(rescaled)
    # Generate training validation and test datasets both for reconstruction of states
    train_dataset = TimeSeriesDatasetMine(data_in[:,mask_trai,:], data_out[:,mask_trai]) #TODO: use the pyLOM dataset or torch tensor dataset
    valid_dataset = TimeSeriesDatasetMine(data_in[:,mask_vali,:], data_out[:,mask_vali]) #TODO: use the pyLOM dataset
    # Fit SHRED
    shred.fit(train_dataset, valid_dataset, epochs=1500, patience=100, verbose=False)
    shred.save('%s%i' % (shreds,kk), scalpath, mysensors)

pyLOM.cr_info()
