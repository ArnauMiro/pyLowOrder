#!/usr/bin/env python
#
# PYLOM Testsuite
# Run SHRED on the cylinder dataset: random sensors, POD of the training
# snapshots, a short fit of two sensor configurations, and a save/load round trip.
#
# Last revision: 25/11/2026
import sys, os, numpy as np, torch
import pyLOM, pyLOM.NN

DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]
SEED      = 42

pyLOM.NN.set_seed(SEED)
device = pyLOM.NN.select_device('cpu')
RESUDIR = os.path.join(OUTDIR,'SHRED_%s' % os.path.basename(DATAFILE).split('.')[0])
pyLOM.NN.create_results_folder(RESUDIR,verbose=False)


## Data loading and split in training, validation and test
m = pyLOM.Mesh.load(DATAFILE)
d = pyLOM.Dataset.load(DATAFILE,ptable=m.partition_table)
tridx, vaidx, teidx = d.split_data('time', mode='reconstruct', seed=SEED)
ntime = d.get_variable('time').shape[0]


## Random sensors inside the wake
nsens   = 10
bounds  = np.array([0.5,8.,-1.,1.])
sensors = d.select_random_sensors(nsens, bounds, VARIABLES, seed=SEED)
sens_trai = sensors.mask_field(VARIABLES[0], tridx)
sens_vali = sensors.mask_field(VARIABLES[0], vaidx)
sens_test = sensors.mask_field(VARIABLES[0], teidx)


## POD of the training snapshots, validation and test projected on its modes
PSI,S,V = pyLOM.POD.run(d.mask_field(VARIABLES[0], tridx),remove_mean=True,randomized=True,r=8,q=3,seed=SEED)
def project(X):
	return pyLOM.math.matmul(pyLOM.math.diag(1/S), pyLOM.math.matmulp(pyLOM.math.transpose(PSI), X))
pod_trai = V.astype(np.float32)
pod_vali = project(d.mask_field(VARIABLES[0], vaidx)).astype(np.float32)
pod_test = project(d.mask_field(VARIABLES[0], teidx)).astype(np.float32)
pod_scaler = pyLOM.NN.MinMaxScaler(column=True)
pod_scaler.fit(pod_trai)
podpath  = os.path.join(RESUDIR,'pod_scaler.json')
pod_scaler.save(podpath)
trai_out = pod_scaler.transform(pod_trai)
vali_out = pod_scaler.transform(pod_vali)
Sscale   = S/np.sum(S)
pyLOM.pprint(0,'TSUITE S            =',S.min(),S.max(),S.mean())


## Fit two sensor configurations, save and load each one
shred = pyLOM.NN.SHRED(trai_out.shape[0], device, nsens, nconfigs=2, seed=SEED)
for kk, mysensors in enumerate(shred.configs):
	myscaler = pyLOM.NN.MinMaxScaler(column=True) # one variable per sensor (rows)
	myscaler.fit(sens_trai[mysensors,:])
	scalpath = os.path.join(RESUDIR,'scaler_%i.json' % kk)
	myscaler.save(scalpath)
	embedded = np.zeros((len(mysensors),ntime), dtype=np.float32)
	embedded[:,tridx] = myscaler.transform(sens_trai[mysensors,:])
	embedded[:,vaidx] = myscaler.transform(sens_vali[mysensors,:])
	embedded[:,teidx] = myscaler.transform(sens_test[mysensors,:])
	delayed  = pyLOM.math.time_delay_embedding(embedded, dimension=20)
	train_dataset = pyLOM.NN.Dataset(trai_out, variables_in=delayed[:,tridx,:])
	valid_dataset = pyLOM.NN.Dataset(vali_out, variables_in=delayed[:,vaidx,:])
	shred.fit(train_dataset, valid_dataset, epochs=30, patience=100, verbose=False, mod_scale=torch.tensor(Sscale))
	fname = os.path.join(RESUDIR,'shred_%i' % kk)
	shred.save(fname, scalpath, podpath, mysensors)
	# Output of the fitted model and of the loaded one on the test snapshots
	xtest = torch.from_numpy(delayed[:,teidx,:]).permute(1,2,0).to(device)
	shred.eval()
	with torch.no_grad():
		out = shred(xtest).cpu().numpy().T
	loaded, saved = pyLOM.NN.SHRED.load(fname, device)
	with torch.no_grad():
		out_loaded = loaded(xtest).cpu().numpy().T
	pred = pod_scaler.inverse_transform(out)
	pyLOM.pprint(0,'TSUITE pred_%i       =' % kk,pred.min(),pred.max(),pred.mean())
	pyLOM.pprint(0,'TSUITE load_%i       =' % kk,np.abs(out_loaded-out).max(),int(np.array_equal(saved['sensors'],mysensors)),len(loaded.configs))


pyLOM.cr_info()
pyLOM.pprint(0,'End of output')
