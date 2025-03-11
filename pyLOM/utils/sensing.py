import numpy as np

def generate_random_sensors(N, x0, x1, y0, y1, z0, z1, seed=-1):
	np.random.seed(0) if seed == -1 else np.random.seed(seed)
	# Generate random points using numpy's uniform distribution
	x = np.random.uniform(x0, x1, N)
	y = np.random.uniform(y0, y1, N)
	z = np.random.uniform(z0, z1, N)
	points = np.vstack((x, y, z)).T  # Stack them into an Nx3 array (x, y, z)
	return points

def nearest_neighbour2sensor(randcoords, meshcoords, dataset):
    N         = randcoords.shape[0]
    senscoord = np.zeros((N, meshcoords.shape[1]))                # Sensor real coordinates
    sensdata  = np.zeros((N, dataset.shape[1])) # Sensor data
    for ii, sensor in enumerate(randcoords):
        dist = np.sum((sensor-meshcoords)**2, axis=1)
        imin = np.argmin(dist)
        senscoord[ii,:] = meshcoords[imin]
        sensdata[ii,:]  = dataset[imin]
    return senscoord, sensdata