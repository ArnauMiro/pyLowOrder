import numpy as np

from ..utils.mpi import MPI_RANK, mpi_bcast, mpi_reduce
from ..utils     import raiseError, is_rank_or_serial 


def data_splitting(Nt:int, mode:str, seed:int=-1, root:int=0):
	r'''
	Generate random training, validation and test masks for a dataset of Nt samples.

	Args:
		Nt (int): number of data samples.
		mode (str): type of splitting to perform. In reconstruct mode all three datasets have samples along all the data range.
		seed (int, optional): (default: ``-1``).
		root (int,optional): (default: ``0``).

	Returns:
		[(np.ndarray), (np.ndarray), (np.ndarray)]: List of arrays containing the identifiers of the training, validation and test samples.
	'''
	# Setup seed
	if seed >= 0: np.random.seed(seed)

	# Mask should be the same for all ranks, we thus generate it at rank=0
	if mode =='reconstruct':
		tridx, vaidx, teidx = [], [], []
		if is_rank_or_serial(root):
			# Here we explicitly avoid the start and end of the mask
			tridx       = np.sort(np.random.choice(Nt-2, size=int(0.7*(Nt))-2, replace=False)+1)
			mask        = np.ones(Nt,dtype=bool)
			mask[tridx] = 0
			mask[0]     = 0
			mask[-1]    = 0
			tridx       = np.argwhere(mask==0)[:,0]
			vate_idx    = np.arange(0, Nt)[np.where(mask!=0)[0]]
			vaidx       = vate_idx[::2]
			teidx       = vate_idx[1::2]
		# Broadcast to all ranks
		tridx = mpi_bcast(tridx,root=root)
		vaidx = mpi_bcast(vaidx,root=root)
		teidx = mpi_bcast(teidx,root=root)
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

def find_random_sensors(bounds:np.ndarray, xyz:np.ndarray, nsensors:int, root:int=0):
	r'''
	Generate a set of random points inside a bounding box and find the closest grid points to them

	Args:
		bounds (np.ndarray): bounds of the box in the following format: np.array([xmin, xmax, ymin, ymax, zmin, zmax])
		xyz (np.ndarray): coordinates of the grid
		nsensors(int): number of sensors to generate
		root(int): rank that generates the sensors

	Returns:
		np.ndarray: array with the indices of the points
	'''
	# Generate random points using numpy's uniform distribution
	# Here we build the random points at a global domain box in a single processor
	# that is later broadcasted to every rank
	xyz_sensors = []
	if is_rank_or_serial(root):
		x = np.random.uniform(bounds[0], bounds[1], nsensors)
		y = np.random.uniform(bounds[2], bounds[3], nsensors)
		z = np.random.uniform(bounds[4], bounds[5], nsensors) if len(bounds) > 4 else None
		# Stack them into an Nxndim
		xyz_sensors = np.vstack((x,y,z)).T if z is not None else np.vstack((x,y)).T
	# Broadcast to all ranks
	xyz_sensors = mpi_bcast(xyz_sensors,root=root)
	
	# At this point all ranks should have xyz_sensors, now find which rank contains
	# the point and relate it with the closest point
	ranklist, idxlist = [], []
	for ii, xyz_sensor in enumerate(xyz_sensors):
		# Find the square of the distance
		# If the partition is empty, i.e., doesn't have any xyz assign
		# a randomly high value
		dist2      = np.sum((xyz_sensor-xyz)**2,axis=1) if xyz.shape[0] > 0 else 1e99
		mindist    = np.min(dist2)
		# Now find which rank has the minimum distance
		_,minrank  = mpi_reduce((mindist,MPI_RANK),all=True,op='argmin')
		if is_rank_or_serial(minrank):
			ranklist.append(minrank)
			idxlist.append(np.argmin(dist2))
	
	# Return arrays ensuring correct integer type
	return np.array(idxlist,np.int32), np.array(ranklist,np.int32)