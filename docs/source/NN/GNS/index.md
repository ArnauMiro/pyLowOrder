# GNS in pyLOM

The Graph Neural Simulator (GNS) stack in `pyLOM.NN` provides:

- graph construction from `pyLOM.Mesh`
- graph-based surrogate models (`GNS`)
- training orchestration (`Pipeline`)
- reproducible experiment artifacts (`save_experiment_artifacts`)

This section documents the current implementation in this repository.

## Scope

The documented modules are:

- `pyLOM.NN.architectures.gns.GNS`
- `pyLOM.NN.gns.graph.Graph`
- `pyLOM.NN.pipeline.Pipeline`
- `pyLOM.NN.utils.config_schema` (internal config schema)
- `pyLOM.NN.utils.experiment` (artifacts and plotting helpers)

```{toctree}
:maxdepth: 2

usage
api
```
