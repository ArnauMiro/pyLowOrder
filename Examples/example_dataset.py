#!/bin/env python
import numpy as np
import pyLOM


## Parameters
OUTFILE = "./example_dataset.h5" 


## Synthetic data
#
# This is the data you want to perform ML on, i.e., the cooling function
# or the Cp on the airfoil example
K= np.array([[[1,2,3,4],[5, 6, 7, 8],[9,10,11,12]], [[13, 14, 15, 16],[17,18,19,20],[21,22,23,24]]],np.float32).T
print('K:',K.shape) # (4, 3, 2)

# So the first dimension, 4, are the xyz positions
# then xyz must have length (4,3) or (4,2) in 2D
xyz = np.array([[1,19,12],[1,20,12],[2,5,11],[3,6,11]],np.float32)
print('xyz:',xyz.shape) # (4, 3)

# For the point order array we simply numerate our
# points consecutively
pointO = np.arange(xyz.shape[0])
print('pointO:',pointO.shape) # (4,)

# Now we are missing two variables of size (3,) and (2,)
# for this example we will call them `var1` and `var2`
var1 = np.array([20,30,40],np.float32)
print('var1:',var1.shape) # (3,)
var2 = np.array([200,300],np.float32)
print('var2:',var2.shape) # (2,)
print()


## Create a serial partition table
# The number of points is simply the first dimension
# of K and xyz
npoints = xyz.shape[0]
# Here we create a table with a single partition with the
# same number of points and elements (i.e., a cloud of points)
ptable  = pyLOM.PartitionTable.new(1,npoints,npoints)
print(ptable)


## Create a pyLOM dataset
# Now we can arrange the dataset
# In this example, it is indiferent if we treat the data as
# points or cells. For simplicity we will assume the data to
# be points.
d = pyLOM.Dataset(xyz=xyz,ptable=ptable,order=pointO,point=True,
    # Add the variables as a dictionary associated with
    # their dimensions on the matrix K
    vars  = {
        # We associate var1 to the first dimension of K, i.e.,
        # idim = 0 as dimension 0 of K is always the number of
        # points
        'var1' : {'idim':0,'value':var1},
        # We associate var2 to the second dimension of K, i.e.,
        # idim = 1        
        'var2' : {'idim':1,'value':var2},
    },
    # Now we add the fields, i.e., the actual data to compute
    # things. So K has ndim = 2 as there are two extra dimensions
    # other than the number of points
    K = {'ndim':2,'value':K}
)
print(d)

# Now we store the dataset
# we make sure to activate mode = 'w' to overwrite
# any possible early results
d.save(OUTFILE,mode='w')


pyLOM.cr_info()