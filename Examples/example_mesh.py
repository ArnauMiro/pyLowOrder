#%%
import numpy as np
import h5py
import pyLOM

file_path = '/home/p.yeste/pyLowOrder/Testsuite/DATA/CYLINDER.h5'

#%%
m = pyLOM.Mesh.load(file_path)
print(m)

#%%
# Compute necessary geometrical information (this may take a while depending on your mesh size)
cell_nodes = m.xyz # Cell nodes
surf_norms = m.normal
edge_norms = m.edge_normal
cell_connectivity = m.cell_connectivity

print("surf_norms: ", surf_norms.shape)
print("edge_norms: ", edge_norms.shape)
print("cell_connectivity: ", cell_connectivity.shape)

#%%
# Save the new data in an h5 file

save_file = "/home/p.yeste/CETACEO_DATA/CYLINDER_extra.h5"

# copy the original file and add the new data
with h5py.File(file_path, 'r') as f:
    with h5py.File(save_file, 'w') as f_save:
        for key in f.keys():
            f.copy(key, f_save)
        f_save['MESH'].create_dataset('SurfNorms', data=surf_norms)
        f_save['MESH'].create_dataset('EdgeNorm_1', data=edge_norms[:,:3])
        f_save['MESH'].create_dataset('EdgeNorm_2', data=edge_norms[:,3:6])
        f_save['MESH'].create_dataset('EdgeNorm_3', data=edge_norms[:,6:9])
        f_save['MESH'].create_dataset('EdgeNorm_4', data=edge_norms[:,9:])
        f_save['MESH'].create_dataset('CellConnectivity', data=cell_connectivity)

print("Data saved in", save_file)
