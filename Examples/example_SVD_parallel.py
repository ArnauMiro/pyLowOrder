#!/usr/bin/env python
#
# Example of SVD utility so that it returns
# the same output as MATLAB.
#
# Last revision: 09/07/2021
from __future__ import print_function, division

import numpy as np
import pyLOM
from pyLOM.utils.parall import mpi_scatter, mpi_gather


## Define matrix A 8x2
A = np.array([[1,2],[3,4],[5,6],[7,8],[1,2],[3,4],[5,6],[7,8]],dtype=np.double,order='C') if pyLOM.is_rank_or_serial(0) else None

# Scatter A among the processors
Ai = mpi_scatter(A,root=0,do_split=True)


## Run parallel SVD
Ui, S, V = pyLOM.math.tsqr_svd(Ai)

# Gather Ui to processor 0
U = mpi_gather(Ui,root=0)

pyLOM.pprint(0,'pyLOM:')
pyLOM.pprint(0,'U',U.shape if not U is None else None,U)
pyLOM.pprint(0,'S',S.shape,S)
pyLOM.pprint(0,'V',V.shape,V)


pyLOM.cr_info()