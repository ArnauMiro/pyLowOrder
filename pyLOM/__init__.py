#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Last rev: 09/07/2021

__version__ = '3.0.0'

# Import essential tools
from .                import inp_out as io, vmmath as math
from .dataset         import Dataset
from .partition_table import PartitionTable
from .mesh            import Mesh

# Import utilities
from .utils.cr     import cr, cr_nvtx, cr_start, cr_stop, cr_reset, cr_info
from .utils.nvtxp  import nvtxp
from .utils.parall import pprint
from .utils.gpu    import gpu_device
from .utils.plots  import show_plots, close_plots

# Import Low Order Models
from . import POD, DMD, SPOD, MANIFOLD, GPOD

# Import AI Models
# Leaving this commented as the NN module overloads the memory when loaded
# to load the NN module use `import pyLOM.NN` instead
# Show a warning in case the modules cannot be loaded
#try:
#    from . import NN
#except:
#	from .utils.errors import raiseWarning
#	raiseWarning('Import - Cannot load NN! Ensure that the correct dependencies are installed',allranks=False)
#	del raiseWarning


del dataset, partition_table, mesh
