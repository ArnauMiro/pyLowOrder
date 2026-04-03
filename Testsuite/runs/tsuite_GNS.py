#!/usr/bin/env python
#
# PYLOM Testsuite
# Run GNS on the cylinder testsuite dataset
#
# Last revision: 03/04/2026
from __future__ import print_function, division

import os
import sys
from pathlib import Path

import h5py
from dacite import Config as DaciteConfig, from_dict
import pyLOM, pyLOM.NN
from pyLOM import Mesh
from pyLOM.NN import Graph
from pyLOM.NN.utils.config_schema import GNSModelConfig, GNSTrainingConfig


DATAFILE  = sys.argv[1]
VARIABLES = eval(sys.argv[2])
OUTDIR    = sys.argv[3]


## Set device
device = pyLOM.NN.select_device("cpu")


## Data loading
d = pyLOM.Dataset.load(DATAFILE)
y = d.X(*VARIABLES)
print(d)


## Load pyLOM dataset and set up results output
RESUDIR = os.path.join(OUTDIR, f"GNS_{DATAFILE}")
pyLOM.NN.create_results_folder(RESUDIR, verbose=False)


## Build (or reuse) graph in train file
train_path = Path(DATAFILE).resolve()
with h5py.File(str(train_path), "r") as f:
    has_graph = "GRAPH" in f

if has_graph:
    graph = Graph.load(str(train_path), device="cpu")
    mesh_shape = (int(graph.num_nodes),)
else:
    mesh = Mesh.load(str(train_path), mpio=False)
    graph = Graph.from_pyLOM_mesh(mesh=mesh, device="cpu")
    graph.save(str(train_path), mode="a")
    mesh_shape = (int(mesh.ncells),)


## Generate torch dataset
input_scaler  = pyLOM.NN.MinMaxScaler()
output_scaler = pyLOM.NN.MinMaxScaler()

variables_names = list(getattr(d, "varnames", [])) or ["time"]

nn_dataset = pyLOM.NN.Dataset.load(
    str(train_path),
    field_names=VARIABLES,
    variables_names=variables_names,
    add_variables=True,
    add_mesh_coordinates=False,
    mesh_shape=mesh_shape,
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    squeeze_last_dim=False,
    channels_last=True,
)

td_train, td_test = nn_dataset.get_splits([0.8, 0.2])


## Generate model
model_cfg = GNSModelConfig(
    input_dim=len(variables_names),
    output_dim=1,
    latent_dim=16,
    hidden_size=128,
    num_msg_passing_layers=1,
    encoder_hidden_layers=2,
    decoder_hidden_layers=1,
    message_hidden_layers=1,
    update_hidden_layers=1,
    groupnorm_groups=1,
    activation="torch.nn.ELU",
    p_dropout=0.0,
    seed=42,
    device="cpu",
    debug=False,
)

train_cfg_dict = {
    "epochs": 2,
    "lr": 1.0e-3,
    "weight_decay": 0.0,
    "lr_gamma": 0.98,
    "lr_scheduler_step": 1,
    "loss_fn": "torch.nn.MSELoss",
    "optimizer": "torch.optim.Adam",
    "scheduler": "torch.optim.lr_scheduler.StepLR",
    "print_every": 1,
    "dataloader": {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False,
    },
    "subgraph_loader": {
        "batch_size": 2048,
        "shuffle": True,
        "input_nodes": None,
        "mode": "nodes",
        "seed_selector": {"type": "all", "frac": None, "nodes_path": None},
    },
}

dcfg = DaciteConfig(strict=True)
train_cfg = from_dict(GNSTrainingConfig, train_cfg_dict, config=dcfg)

model = pyLOM.NN.GNS.from_graph_path(config=model_cfg, graph_path=str(train_path))

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    model=model,
    training_params={"config": train_cfg},
)

training_logs = pipeline.run()


## check saving and loading the model
pipeline.model.save(os.path.join(RESUDIR, "model.pth"))
model = pyLOM.NN.GNS.load(RESUDIR + "/model.pth")
preds = model.predict(td_test)

scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y     = output_scaler.inverse_transform([td_test[:][1].numpy()])[0]


## Testsuite output
pyLOM.pprint(0, "TSUITE y            =", y.min(), y.max(), y.mean())
#pyLOM.pprint(0, "TSUITE scaled_y     =", scaled_y.min(), scaled_y.max(), scaled_y.mean())
#pyLOM.pprint(0, "TSUITE scaled_preds =", scaled_preds.min(), scaled_preds.max(), scaled_preds.mean())

pyLOM.cr_info()
pyLOM.pprint(0, "End of output")
