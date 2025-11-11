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

from pyLOM.utils   import cr
from pyLOM.utils   import gpu_to_cpu, pprint
from pyLOM.inp_out import h5_save_QR, h5_load_QR
from pyLOM         import PartitionTable
from ..            import Dataset, select_device

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

## Save the autoencoder weights and latent space

## Load the autoencoder weights and latent space

## Create dataset
def create_dataset(matrix, scale='max', device=select_device()):
	if scale == 'max':
		matmax = np.max(np.abs(matrix))
		matsca = matrix/matmax
		scaler = matmax
	elif scale == 'meanstd':
		matmean = np.mean(matrix)
		matstd  = np.std(matrix)
		matsca  = (matrix-matmean)/matstd
		scaler  = np.array([matmean,matstd])
	else:
		matsca = matrix
	matsca = torch.tensor((matsca).astype(np.float32), device=device)
	return Dataset((matsca,), mesh_shape=(matsca.shape[0],), snapshots_by_column=True), scaler