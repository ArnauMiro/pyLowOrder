#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Last rev: 09/07/2021

__VERSION__ = '1.0.0'

# Import Low Order Models
from . import POD#, DMD

# Import essential tools
from .        import inp_out as io, vmmath as math
from .dataset import Dataset

# Import utilities
from .utils.cr     import cr_start, cr_stop, cr_reset, cr_info
from .utils.parall import pprint, is_rank_or_serial
from .utils.plots  import show_plots, close_plots


del utils, dataset
