#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# IO Module
#
# Last rev: 20/10/2021

# Pickle and HDF5 exchange format
from .io_pkl  import pkl_load, pkl_save
from .io_h5   import h5_load_dset, h5_save_dset, h5_append_dset, h5_save_mesh, h5_load_mesh, h5_save_QR, h5_load_QR, h5_save_POD, h5_load_POD, h5_save_DMD, h5_load_DMD, h5_save_SPOD, h5_load_SPOD, h5_save_VAE

# VTK HDF5 3D format
from .io_vtkh5 import vtkh5_save_mesh, vtkh5_link_mesh, vtkh5_save_field

# Output to paraview
from .io_paraview import pv_writer

## File name formats
FMT_QR_FILE = 'QR_%s-%i.h5' # Live QR factorization

del io_pkl, io_h5, io_paraview
