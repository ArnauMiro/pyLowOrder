# %%
import os
import numpy as np
from torch.utils.data import DataLoader

import pyLOM
from pyLOM.NN import Dataset
#%%

datapath = '/home/p.yeste/CETACEO_DATA/'

npoints = 100
nsamples = 39
ptable = pyLOM.PartitionTable.new(1, npoints, npoints)

xyz = np.random.rand(npoints, 2).astype(np.float32)

var_1 = np.random.rand(nsamples,).astype(np.float32)
var_2 = np.random.rand(nsamples,).astype(np.float32)
opt_1 = np.random.rand(nsamples, npoints).astype(np.float32)
opt_2 = np.random.rand(nsamples, npoints).astype(np.float32)
# opt_3 = np.random.rand(nsamples, npoints, 1).astype(np.float32) NN.Dataset only allows ndim=2 for now
# opt_4 = np.random.rand(nsamples, npoints, 2).astype(np.float32)


ds = pyLOM.Dataset(
    xyz=xyz,
    ptable=ptable,
    order=np.arange(npoints),
    point=True,
    vars={
        'Var1': {'idim': 0, 'value': var_1},
        'Var2': {'idim': 0, 'value': var_2},
    },
    # 'value' must have shape (npoints*ndim, nsamples)
    OPT1={'ndim': 1, 'value': opt_1.transpose(1,0).reshape(npoints*1, nsamples)},
    OPT2={'ndim': 1, 'value': opt_2.transpose(1,0).reshape(npoints*1, nsamples)},
    # OPT3={'ndim': 1, 'value': opt_3.transpose(1,2,0).reshape(npoints*1, nsamples)},
    # OPT4={'ndim': 2, 'value': opt_4.transpose(1,2,0).reshape(npoints*2, nsamples)},
)

print(ds)

out_path = os.path.join(datapath, "test_dataset.h5")
if os.path.isfile(out_path):
    print(f"File {out_path} already exists. Overwriting.")
    os.remove(out_path)
ds.save(out_path, append=False)
print(f"Dataset saved to {out_path}")

#%%
ds_nn = pyLOM.NN.Dataset.load(
    out_path,
    field_names=["OPT1", "OPT2"],
    add_variables=True,
    add_mesh_coordinates=False,
    variables_names=["Var1", "Var2"],
    inputs_scaler=None,
    outputs_scaler=None,
)
print(ds_nn)

for i in range(len(ds_nn)):
    x, y = ds_nn[i]
    print(f"x: {x.shape}, y: {y.shape}")
    print(f"x: {x}, y: {y}")
    if i == 0:
        break  # Just to show the first sample

dl = DataLoader(ds_nn, batch_size=1, shuffle=True)
for batch in dl:
    x, y = batch
    print(f"x: {x.shape}, y: {y.shape}")
    print(f"x: {x}, y: {y}")
    break  # Just to show the first batch

# %%
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

x = np.random.rand(2,)

# Create a PyTorch dataset
dataset = TensorDataset(torch.tensor(x))
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in dataloader:
    x = batch[0]
    print(f"x: {x.shape}")
# %%

import numpy as np
import torch


B = 3  # Batch size
N = 5  # Number of nodes per batch
edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])  # Example edge index

num_edges = edge_index.size(1)

# Expande edge_index: [2, num_edges] → [B, 2, num_edges]
edge_index = edge_index.unsqueeze(0).expand(B, -1, -1)
print(f"Expanded edge_index shape: {edge_index.shape}")  # Should be [B, 2, num_edges]
print(f"Edge index: {edge_index}")

# Crea offset por batch: [B, 1, 1] para sumar solo a las filas de nodos
offset = (torch.arange(B) * N).view(B, 1, 1)
print(f"Offset shape: {offset.shape}")  # Should be [B, 1, 1]
print(f"Offset: {offset}")

# Aplica el offset solo a la segunda fila (source and target nodes)
edge_index = edge_index + offset  # [B, 2, num_edges]
print(f"Offset edge_index shape: {edge_index.shape}")  # Should be [B, 2, num_edges]
print(f"Offset edge_index: {edge_index}")

# Reorganiza a [2, B * num_edges]
edge_index_batch = edge_index.permute(1, 0, 2).reshape(2, B * num_edges)
print(f"Final edge_index_batch shape: {edge_index_batch.shape}")  # Should be [2, B * num_edges]
print(f"Final edge_index_batch: {edge_index_batch}")


# %%
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader

x = torch.rand(10, 3)  # Node features for 10 nodes
y = torch.rand(10, 2)  # Node labels for 10 nodes
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5, 6, 7, 8, 9]])  # Example edge index
edge_attr = torch.rand(edge_index.shape[1], 2)  # Edge features for 10 edges

g = Data(
    x=x,
    node_features=x,
    edge_index=edge_index,
    edge_features=edge_attr,
    y=y,
    pepe=x,
    paco=y,
)
print(g)

subg_loader = NeighborLoader(
    g,
    num_neighbors=[2, 2],  # Number of neighbors to sample at each layer
    batch_size=2,  # Batch size
    subgraph_type="induced"
)
print(subg_loader)
for batch in subg_loader:
    print(f"Batch: {batch}")
    print(f"Batch node features: {batch.x.shape}, Batch edge index: {batch.edge_index.shape}, Batch labels: {batch.y.shape}")
    break  # Just to show the first batch
# %%
