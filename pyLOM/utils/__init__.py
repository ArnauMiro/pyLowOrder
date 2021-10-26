#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Utils Module
#
# Last rev: 19/07/2021

__VERSION__ = '1.0.0'

from .errors import raiseError, raiseWarning
from .cr     import cr_start, cr_stop, cr_info
from .matrix import transpose, norm
from .parall import MPI_RANK, MPI_SIZE, worksplit, is_rank_or_serial, pprint
from .parall import mpi_scatter, mpi_gather, mpi_reduce

del errors, cr
