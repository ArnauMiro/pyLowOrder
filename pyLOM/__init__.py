#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Last rev: 09/07/2021

__version__ = '3.2.5'

# Import essential tools
from .                import inp_out as io, vmmath as math
from .dataset         import Dataset
from .partition_table import PartitionTable
from .mesh            import Mesh

# Import utilities
from .utils.cr     import cr, cr_nvtx, cr_start, cr_stop, cr_reset, cr_info
from .utils.nvtxp  import nvtxp
from .utils.parall import pprint
from .utils.plots  import show_plots, close_plots, style_plots
from .utils.gpu    import gpu_device

# Import Low Order Models
from . import POD, PCA, DMD, SPOD, MANIFOLD, GPOD

# Import AI Models
# The NN module overloads the memory when loaded
# to load the NN module use `import pyLOM.NN` instead

del dataset, partition_table, mesh
