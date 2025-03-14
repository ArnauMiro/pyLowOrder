#!/usr/bin/env python
#
# Example of several statistical metrics
# compared with scikit-learn
#
# Last revision: 09/07/2021
from __future__ import print_function, division

import numpy as np
import pyLOM

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


## Vectorial fields 
pyLOM.pprint(0,'Vectorial field:')
y_true = np.array([3, -0.5, 2, 7],np.double)
y_pred = np.array([2.5, 0.0, 2, 8],np.double)

pyLOM.pprint(0,'pyLOM RMSE:  ', pyLOM.math.RMSE(y_true,y_pred,relative=False))
pyLOM.pprint(0,'pyLOM MAE:   ', pyLOM.math.MAE(y_true,y_pred))
pyLOM.pprint(0,'pyLOM r2:    ', pyLOM.math.r2(y_true,y_pred))
pyLOM.pprint(0,'scikit RMSE: ', root_mean_squared_error(y_true, y_pred))
pyLOM.pprint(0,'scikit MAE:  ', mean_absolute_error(y_true, y_pred))
pyLOM.pprint(0,'scikit r2:   ', r2_score(y_true, y_pred))


## Matricial fields
pyLOM.pprint(0,'Matricial field:')
y_true =  np.array([[0.5, 1],[-1, 1],[7, -6]],np.double)
y_pred =  np.array([[0, 2],[-1, 2],[8, -5]],np.double)

pyLOM.pprint(0,'pyLOM RMSE:  ', pyLOM.math.RMSE(y_true,y_pred,relative=False))
pyLOM.pprint(0,'pyLOM MAE:   ', pyLOM.math.MAE(y_true,y_pred))
pyLOM.pprint(0,'pyLOM r2:    ', pyLOM.math.r2(y_true,y_pred))
pyLOM.pprint(0,'scikit RMSE: ', root_mean_squared_error(y_true, y_pred))
pyLOM.pprint(0,'scikit MAE:  ', mean_absolute_error(y_true, y_pred))
pyLOM.pprint(0,'scikit r2:   ', r2_score(y_true, y_pred))

pyLOM.cr_info()