import numpy as np

from ..utils     import raiseError, mpi_reduce, MPI_RANK


def data_splitting(Nt:int, mode:str, seed:int=-1):
	r'''
	Generate random training, validation and test masks for a dataset of Nt samples.

	Args:
		Nt (int): number of data samples.
		mode (str): type of splitting to perform. In reconstruct mode all three datasets have samples along all the data range.
		seed (int, optional): (default: ``-1``).

	Returns:
		[(np.ndarray), (np.ndarray), (np.ndarray)]: List of arrays containing the identifiers of the training, validation and test samples.
	'''
	np.random.seed(0) if seed < 0 else np.random.seed(seed)
	if mode =='reconstruct':
		tridx       = np.sort(np.random.choice(Nt, size=int(0.7*(Nt)), replace=False))
		mask        = np.ones(Nt)
		mask[tridx] = 0
		mask[0]     = 0
		mask[-1]    = 0
		tridx       = np.argwhere(mask==0)[:,0]
		vate_idx    = np.arange(0, Nt)[np.where(mask!=0)[0]]
		vaidx       = vate_idx[::2]
		teidx       = vate_idx[1::2]
	else:
		raiseError('Data split mode not implemented yet')
	return tridx, vaidx, teidx

def time_delay_embedding(X, dimension=50):
	r'''
	Extract time-series of lenght equal to lag from longer time series in data, whose dimension is (number of time series, sequence length, data shape)
	Inputs: [Points, Time]
	Output: [Points, Time, Delays]
	'''
	
	X_delay = np.zeros((X.shape[0], X.shape[1], dimension), dtype=X.dtype)
	for i in range(X.shape[0]):
		for j in range(1,X.shape[1]+1):
			if j < dimension:
				X_delay[i,j-1,-j:] = X[i,:j]
			else:
				X_delay[i,j-1,:] = X[i,j-dimension:j]

	return X_delay

def find_random_sensors(bounds:np.ndarray, xyz:np.ndarray, nsensors:int):
	r'''
	Generate a set of random points inside a bounding box and find the closest grid points to them

	Args:
		bounds (np.ndarray): bounds of the box in the following format: np.array([xmin, xmax, ymin, ymax, zmin, zmax])
		xyz (np.ndarray): coordinates of the grid
		nsensors(int): number of sensors to generate

	Returns:
		np.ndarray: array with the indices of the points
	'''
	# Generate random points using numpy's uniform distribution
	x = np.random.uniform(bounds[0], bounds[1], nsensors)
	y = np.random.uniform(bounds[2], bounds[3], nsensors)
	z = np.random.uniform(bounds[4], bounds[5], nsensors) if len(bounds) > 4 else None
	# Stack them into an Nxndim
	randcoords  = np.vstack((x, y, z)).T if z is not None else np.vstack((x, y)).T 
	
	# Find which rank has the closest point to the sensors
	mysensors = list()
	for ii, sensor in enumerate(randcoords):
		dist       = np.sum((sensor-xyz)**2, axis=1)
		mindist    = np.min(dist)
		_,minrank  = mpi_reduce((mindist, MPI_RANK), all=True, op='argmin')
		if minrank == MPI_RANK:
			mysensors.append(np.argmin(dist))
	
	return np.array(mysensors)