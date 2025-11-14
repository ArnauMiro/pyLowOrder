#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# LAMINE: Live Analysis Multicore Integration for Numerical Engineering module.
# 
# Functions to compute modal decomposition of data that is being generated live. 
# An example of it is a CFD simulation that shares data to a python process through SMARTREDIS
#
# Last rev: 14/11/2025
from __future__ import print_function

from ..             import PartitionTable
from ..vmmath       import update_qr_streaming, init_qr_streaming, svd, matmul
from ..inp_out      import h5_load_QR, h5_save_QR
from ..utils.cr     import cr_nvtx as cr

FMT_QR_FILE = '%s/QR_%s-%i.h5'

## Live QR
@cr('LAMINE.qr')
def QR(vardict:dict, nmodes:int, ptable:PartitionTable, loadQBY:bool, iiload:int, iisave:int, basedir:str='.'):
	r"""
	Function to compute the randomized QR factorization on data which is being generated runtime. Once the of each variable is computed, it is saved in a .h5 file. Only 2 .h5 files are written, *-1.h5 and *-2.h5, as they get overwritten every odd or even QR computation, respectively.
	The saved QR results can be used to compute POD or GAVI.

	Args:
		vardict (dict): dictionary containing all the variables to decompose. It has the following structure: {'varname1':valuesvar1, 'varname2':valuesvar2}
		nmodes (int): how many modes should be recovered by the QR factorization
		ptable (PartitionTable): partition in which the data has to be saved (should be the one from the original dataset)
		loadQBY (bool): if we need to load previous QR data. Might be false on the first iteration of the code but then will always be True
		iiload (int): previous QR iteration to load
		iisave (int): which QR iteration we compute now
		basedir (str, optional): directory where the QR results are saved and loaded from (default: ``'.'``)

	Returns:
		[Bool, int, int]: the new values for loadQBY, iiload and iisave.
	"""
	for var in vardict:
		if loadQBY:
			Qvars = h5_load_QR(FMT_QR_FILE % (basedir,var,iiload), ['Q','B','Y'], ptable=ptable)
			Q,B,Y = update_qr_streaming(vardict[var], Qvars[0], Qvars[2], Qvars[1], nmodes, 1)
			del Qvars
		else:
			Q, B, Y = init_qr_streaming(vardict[var], nmodes, 1)			
		h5_save_QR(FMT_QR_FILE % (basedir,var,iisave), Q, Y, B, ptable)
		del Q, B, Y

	loadQBY = True
	iiload  = 2 if iiload == 1 else 1
	iisave  = 2 if iisave == 1 else 1

	return loadQBY, iiload, iisave

## Recover POD modes
@cr('LAMINE.POD')
def QR2POD(varname:str, ptable:PartitionTable, iiload:int, basedir:str='.'):
	r"""
	Function to compute POD from the QR results computed live and saved after a LAMINE process

	Args:
		varname (str): name of the variable to compute the POD. Must be consistent with the name set in QR factorization computation as we'll look for a QR file with this varname
		ptable (PartitionTable): partition of the original data
		iiload (int): which QR result we want to save (1 or 2)
		basedir (str, optional): directory where the QR results must be loaded from (default ``'.'``)

	Returns
		[np.ndarray, np.ndarray, np.ndarray]: left singular vectors, singular values and right singular vectors
	"""
	Q, B   = h5_load_QR(FMT_QR_FILE % (basedir,varname,iiload), ['Q','B'], ptable=ptable)
	Ur,S,V = svd(B)
	U      = matmul(Q,Ur)
	del U, S, V