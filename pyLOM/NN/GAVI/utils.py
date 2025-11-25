#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# GAVI (Geometry Agnostic Variational-autoencoders Integration) general utilities.
#
# Eiximeno, B., A., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. (2025). 
# On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models. 
# Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797
#
# Last rev: 24/10/2025

import numpy as np
import torch

from ..                 import DEVICE
from ..utils            import Dataset
from ...partition_table import PartitionTable
from ...utils.cr        import cr
from ...utils.errors    import raiseWarning
from ...utils.gpu       import gpu_to_cpu, cpu_to_gpu
from ...inp_out.io_h5   import h5_save_QR, h5_load_QR, h5_load_compressed


@cr('GAVI.save_QR')
def save(fname:str,Q:np.ndarray,B:np.ndarray,ptable:PartitionTable,pointData:bool=True,mode:str='w'):
	r'''
	Store the QR factorization results results in serial or parallel according to the partition used to compute it. It will be saved on a h5 file.

	Args:
		fname (str): path to the .h5 file in which the POD will be saved
		Q (np.ndarray): Q columns to save.
		B (np.ndarray): R matrix to save.
		ptable (PartitionTable): partition table used to compute the QR
		pointData (bool, optional): bool to specify if the POD was performed either on point data or cell data (default ``True``)
		mode (str, optional): mode in which the HDF5 file is opened, 'w' stands for write mode and 'a' stands for append mode. Write mode will overwrite the file and append mode will add the informaiton at the end of the current file, choose with great care what to do in your case (default ``w``).
	'''
	h5_save_QR(fname,gpu_to_cpu(Q),None,gpu_to_cpu(B),ptable,nvars=1,pointData=pointData,mode=mode)

## Load the QR factorization
@cr('GAVI.load_QR')
def load(fname:str,vars:list=['Q','B'],ptable:PartitionTable=None):
	r'''
	Load the QR factorization results results in serial or parallel according to the partition given..

	Args:
		fname (str): path to the .h5 file in which the POD will be saved
		Q (np.ndarray): Q columns to load.
		B (np.ndarray): R matrix to load.
		ptable (PartitionTable): partition table used to compute the QR (default, None)

	Returns:
		list: list of the np.ndarray requested to load.
	'''
	return h5_load_QR(fname,vars,ptable=ptable)

@cr('GAVI.load_compressed')
def load_compressed(fname:str, ptable:PartitionTable, nelxAE:int=1, basedir:str='./'):
	r"""
	Function to load the compressed data from Q arrays and move the necessary values to the GPU if available
	
	Args:
		fname (str): name of the file to load
		ptable (PartitionTable): partition table that we are using to decompress (should be the one of the visualizing mesh)
		nelxAE (int, optional): how many elements of the mesh are compressed per autoencoder (default is ``1``)
		basedir (str, optional): folder where the compressed file is located (default ``"./``)
		
	Returns:
		[np.ndarray, np.ndarray, torch.tensor, torch.tensor, cp.ndarray, cp.ndarray]: arrays of the mean and standard deviation values on the CPU, tensors of the weight and biases of the decoder on the GPU and arrays of the Q and B of the latent spaces in the GPU (if available)	
	"""
	Qmeans, Qstds, weights, biases, Q, B = h5_load_compressed(fname, basedir, ptable, nelxAE)
	return Qmeans, Qstds, torch.tensor(weights, device=DEVICE), torch.tensor(biases, device=DEVICE), cpu_to_gpu(Q), cpu_to_gpu(B)

## Create dataset
@cr('GAVI.create_NNdataset')
def create_dataset(matrix:np.ndarray, scale:str='max', device:torch.device=DEVICE):
	r'''
	Create the pyLOM.NN dataset for neural network training of the GAVI autoencoders
	
	Args:
		matrix (np.ndarray): data matrix that will be added to the dataset with shape (number of modes, number of variables, number of samples).
		scale (str, optional): type of scaler applied to the data, 'max' is recommended for the autoencoder on the R matrix and 'meanstd' is recommended for the autoencoder on the Q matrix (default ``'max'``).
		device (torch.device, optional): device in which the data will be loaded (default: CUDA if available)

	Returns:
		[Dataset, np.ndarray]: pyLOM.NN.Dataset with the scaled data and the scalers used to scale it
	'''
	if scale == 'max':
		matmax = np.max(np.abs(matrix))
		matsca = matrix/matmax
		scaler = matmax
	elif scale == 'meanstd':
		scaler = np.zeros((matrix.shape[1], 2), dtype=np.float32)
		matsca = np.zeros(matrix.shape, dtype=np.float32)
		for ivar in range(matrix.shape[1]):
			matstd  = np.std(matrix[:,ivar,:])
			matmean = np.mean(matrix[:,ivar,:])
			matsca[:,ivar,:] = (matrix[:,ivar,:]-matmean)/matstd
			scaler[ivar]     = np.array([matmean,matstd])
	else:
		matsca = matrix
		scaler = None
		raiseWarning('Scaling method not implemented, setting scaler to None and adding the non-scaled data to the dataset')
	matsca = torch.tensor((matsca).astype(np.float32), device=device)
	return Dataset(tuple(matsca[:, i, :] for i in range(matsca.shape[1])), mesh_shape=(matsca.shape[0],), snapshots_by_column=True), scaler