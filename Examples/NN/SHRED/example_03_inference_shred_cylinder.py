import numpy as np
import torch
import pyLOM, pyLOM.NN

pyLOM.style_plots()


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
sensors   = pyLOM.Dataset.load('sensors.h5')
mask_trai = sensors.get_variable('training_time')
mask_vali = sensors.get_variable('validation_time')
mask_test = sensors.get_variable('test_time')
# Training
sens_trai = sensors.mask_field(sensvar, mask_trai)
nsens     = sens_trai.shape[0]
# Validation
sens_vali = sensors.mask_field(sensvar, mask_vali)
# Test
sens_test = sensors.mask_field(sensvar, mask_test)
# Compute total timesteps
time   = sensors.get_variable('time')
ntimeG = time.shape[0]


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
full_pod = np.zeros((output_size,ntimeG), dtype=np.float32)
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
    myscaler = pyLOM.NN.MinMaxScaler(column=True)
    myscaler = myscaler.load(load_dict['scaler_path'])
    trai_sca = myscaler.transform(mytrai)
    vali_sca = myscaler.transform(myvali)
    test_sca = myscaler.transform(mytest)
    # Concatenate train, test and validation data to generate the embeddings correctly
    embedded = np.zeros((trai_sca.shape[0],ntimeG), dtype=np.float32)
    embedded[:,mask_trai] = trai_sca
    embedded[:,mask_vali] = vali_sca
    embedded[:,mask_test] = test_sca
    delayed = pyLOM.math.time_delay_embedding(embedded, dimension=50)
    # Generate training validation and test datasets both for reconstruction of states
    output = shred(torch.from_numpy(delayed).permute(1,2,0).to(device)).cpu().detach().numpy().T
    # Load the POD scaler of that SHRED to transform POD data
    podscale = pyLOM.NN.MinMaxScaler(column=True)
    podscale = podscale.load(load_dict['podscale_path'])
    outres[:,:,kk] = podscale.inverse_transform(output)
    # Compute mean relative error
    MRE[:,kk] = pyLOM.math.MRE_array(full_pod, outres[:,:,kk])


## Compute reconstruction statistics
meanout = np.mean(outres, axis=2)
stdout  = np.std(outres, axis=2)
meanMRE = np.mean(MRE, axis=1)
stdMRE  = np.std(MRE, axis=1)
print('The mean MRE of the %i configurations is %.2f' % (nconfigs, np.mean(meanMRE)))


## Save the mean output between configurations as the POD coefficients for reconstruction
pyLOM.POD.save('POD_predicted_%s.h5'% podvar,None,None,meanout,sensors.partition_table,nvars=1)


## Plot error bars
#Non-scaled
fig, _ = pyLOM.NN.plotModalErrorBars(meanMRE)
fig.savefig('errorbars.pdf', dpi=300, bbox_inches='tight')
#Scaled
fig, _ = pyLOM.NN.plotModalErrorBars(Sscale*meanMRE)
fig.savefig('errorbars_scaled.pdf', dpi=300, bbox_inches='tight')

## Plot POD modes reconstruction
fig, _ = pyLOM.NN.plotTimeSeries(time, full_pod, meanout, stdout)
fig.savefig('output_modes.pdf', dpi=600)


pyLOM.cr_info()