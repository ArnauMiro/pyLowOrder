import numpy as np
import torch
import pyLOM, pyLOM.NN

def split_reconstruct(Nt):
    ## Splitting into train, test and validation for reconstruction mode of SHRED
    np.random.seed(0)
    tridx       = np.sort(np.random.choice(Nt, size=int(0.7*Nt), replace=False))
    mask        = np.ones(Nt)
    mask[tridx] = 0
    vate_idx    = np.arange(0, Nt)[np.where(mask!=0)[0]]
    vaidx       = vate_idx[::2]
    teidx       = vate_idx[1::2]
    return tridx, vaidx, teidx

class TimeSeriesDatasetMine(torch.utils.data.Dataset):
    '''
    Input: sequence of input measurements with shape (ntrajectories, ntimes, ninput) and corresponding measurements of high-dimensional state with shape (ntrajectories, ntimes, noutput)
    Output: Torch dataset
    '''

    def __init__(self, X, Y):
        self.X = X.permute(1,2,0)
        self.Y = Y.T
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

## Split between train, test and validation: TODO: In the data extraction script part
tridx, vaidx, teidx = split_reconstruct(Nt)

## Import POD coefficients. Stack and rescale them.
pod_coeff   = pyLOM.POD.load('POD_modes_%s.h5' % podvar, vars='V')[0].astype(np.float32)
pod_scaler  = pyLOM.NN.MinMaxScaler()
pod_scaler.fit(pod_coeff)
pod_scaler.save(ouscaler)
rescaled_pod = pod_scaler.transform(pod_coeff)
data_out     = torch.from_numpy(rescaled_pod)
output_size  = data_out.shape[0]

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
    data_del = torch.from_numpy(pyLOM.math.time_delay_embedding(rescaled))
    # Generate training validation and test datasets both for reconstruction of states
    train_dataset = TimeSeriesDatasetMine(data_del[:,tridx,:], data_out[:,tridx]) #TODO: use the pyLOM dataset or torch tensor dataset
    valid_dataset = TimeSeriesDatasetMine(data_del[:,vaidx,:], data_out[:,vaidx]) #TODO: use the pyLOM dataset
    # Fit SHRED
    shred.fit(train_dataset, valid_dataset, epochs=1500, patience=100, verbose=False)
    shred.save('%s%i' % (shreds,kk), scalpath, mysensors)

pyLOM.cr_info()
