#!/bin/env python
#
# Example how to perform in-situ postprocessing
# with Lamine module
#
# Last rev: 24/10/2025

import pyQvarsi
import pyLOM
import numpy as np

from   pyQvarsi.utils import MPI_SIZE

## Case parameters
BASEDIR = '/gpfs/scratch/bsc21/bsc021893/tests/lamine_vishal'
CASJSON = "BluffBodySolverIncomp"
nGPUpn  = 4
nCPUpn  = 76


## Checking that the size matches the one declared
nnodes = int(MPI_SIZE/nCPUpn)
nGPUs  = nnodes*nGPUpn


## Initialize Redis clients on each rank
client = pyQvarsi.lamine.init_client()
pyQvarsi.pprint(0, "All Python clients initialized", flush=True)


## Actual CPU numbering (it changes when running with nnodes > 1)
rerank, CPUlist, data_key = pyQvarsi.lamine.cpu_renumbering(nCPUpn, nGPUpn, nGPUs, result_type='ins') #result_type='avg'


## Read from SOD2D json file
CASESTR, ntime, nsend, isrest, iiload = pyQvarsi.lamine.read_json(CASJSON, basedir=BASEDIR)
nmodes = int(ntime/nsend)
iisave  = 1 if iiload == 2 else 1
loadQBY = False

## Read postprocessing mesh
meshpost = pyQvarsi.MeshSOD2D.read(CASESTR, basedir=BASEDIR)
pyQvarsi.pprint(0, "Mesh read", flush=True)


## Find the interaction between CPUs when using a postprocessing mesh with more partitions than the running mesh
who2send, partsend, ranks, myranksG, map = pyQvarsi.lamine.repartition_mesh(CASESTR, nGPUs, meshpost.partition_table, basedir=BASEDIR)


## Convert postprocessing mesh to pyLOM
qp = meshpost.partition_table
p  = pyLOM.PartitionTable.from_pyQvarsi(meshpost.partition_table, porder=meshpost.porder, has_master=False)
m  = pyLOM.Mesh.from_pyQvarsi(meshpost, ptable=p, sod=True)
pyQvarsi.pprint(0, "pylom conversion is done", flush=True)
m.save('mesh.h5', nopartition=True, mode='w')
pyQvarsi.pprint(0, "pylom mesh is saved", flush=True)

## Deallocate postprocessing mesh as we don't needed anymore
del meshpost


## Pull data from simulation
istep = -1 # initialize the simulation step number to -1
step_list = []     # initialize an empty list containing all the steps sent
counter   = 0
uu = np.zeros((m.xyz.shape[0],nmodes),dtype=np.float32)
vv = np.zeros((m.xyz.shape[0],nmodes),dtype=np.float32)
while True:
	if (client.poll_tensor("step",0,1)):
		tmp = client.get_tensor('step').astype('int32')
	else:
		continue
	if (istep != tmp[0]): 
		istep = tmp[0]	
		step_list.append(istep)
		# We dont return rho because it is incompressible case
		varDict = pyQvarsi.lamine.pull_data(client, rerank, data_key, nGPUs, CPUlist, m.xyz.shape[0], ranks, myranksG, map, who2send, partsend, varList=['u','pr']) 
		field   = pyQvarsi.FieldSOD2D(xyz=m.xyz, ptable=qp,**varDict)
		pyQvarsi.pprint(0,"Timestep %i downloaded successfully" % istep, flush=True)
		uu[:,counter] = field['u'][:,0]  # Storing only the x-velocity component
		vv[:,counter] = field['u'][:,1]  # Storing only the x-velocity component
		counter = counter+1
	else:
		continue
		
	## Use field as wished to do any postprocessing!!!
	if(len(step_list) == nmodes):
		dataDict = {'u':uu, 'v':vv}
		loadQBY, iiload, iisave = pyLOM.LAMINE.QR(dataDict, nmodes, p, loadQBY, iiload, iisave, basedir=BASEDIR)
		
		counter = 0
		step_list = []
			
		pyQvarsi.cr_info()
		pyLOM.cr_info()