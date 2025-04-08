#%%
import numpy as np
import h5py
import pyLOM

file_path = 'path-to-/clean.h5'

#%%
m = pyLOM.Mesh.load(file_path)
print(m)

#%%
# Compute necessary geometrical information (this may take a few minutes)
cell_nodes = m.xyz # Cell nodes
surf_norms = m.normal
edge_norms = m.edge_normal
cell_connectivity = m.cell_connectivity

print("surf_norms: ", surf_norms.shape)
print("edge_norms: ", edge_norms.shape)
print("cell_connectivity: ", cell_connectivity.shape)

#%%
# Save the new data in an h5 file

save_file = "/home/p.yeste/CETACEO_DATA/CLEAN/clean-mesh-vtk.hdf"

with h5py.File(save_file, 'r+') as f:
    if f['VTKHDF'].require_group('CellData'):
        del(f['VTKHDF/CellData'])
    f['VTKHDF'].create_group('CellData')

    f['VTKHDF/CellData'].create_dataset('SurfNorms', data=surf_norms)
    f['VTKHDF/CellData'].create_dataset('EdgeNorm1', data=edge_norms[:,:3])
    f['VTKHDF/CellData'].create_dataset('EdgeNorm2', data=edge_norms[:,3:6])
    f['VTKHDF/CellData'].create_dataset('EdgeNorm3', data=edge_norms[:,6:])
    f['VTKHDF/CellData'].create_dataset('CellConnectivity', data=cell_connectivity)

print("Data saved in", save_file)
