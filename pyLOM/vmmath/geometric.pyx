#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - geometry.
#
# Last rev: 27/10/2021

cimport cython
cimport numpy as np

import numpy as np
from collections import defaultdict, deque

from .cfuncs       cimport real
from .cfuncs       cimport c_scellCenters, c_snormals, c_seuclidean_d
from .cfuncs       cimport c_dcellCenters, c_dnormals, c_deuclidean_d

from ..utils.cr     import cr
from ..utils.errors import raiseError


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _scellCenters(float[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] xyz_cen = np.zeros((nel,ndim),dtype = np.float32)
	# Call C function
	c_scellCenters(&xyz_cen[0,0],&xyz[0,0],&conec[0,0],nel,ndim,ncon)
	# Return
	return xyz_cen

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dcellCenters(double[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] xyz_cen = np.zeros((nel,ndim),dtype = np.double)
	# Call C function
	c_dcellCenters(&xyz_cen[0,0],&xyz[0,0],&conec[0,0],nel,ndim,ncon)
	# Return
	return xyz_cen

@cr('math.cellCenters')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cellCenters(real[:,:] xyz, int[:,:] conec):
	r'''
	Compute the cell centers given a list 
	of elements.

	Args:
		xyz (np.ndarray):   node positions
		conec (np.ndarray): connectivity array

	Returns:
		np.ndarray: center positions
	'''
	if real is double:
		return _dcellCenters(xyz,conec)
	else:
		return _scellCenters(xyz,conec)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _snormals(float[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] normals = np.zeros((nel,ndim),dtype = np.float32)
	# Call C function
	c_snormals(&normals[0,0],&xyz[0,0],&conec[0,0],nel,ndim,ncon)
	# Return
	return normals

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dnormals(double[:,:] xyz, int[:,:] conec):
	'''
	Compute the cell centers given a list 
	of elements (internal function).
	'''
	cdef int nel = conec.shape[0], ndim = xyz.shape[1], ncon = conec.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] normals = np.zeros((nel,ndim),dtype = np.double)
	# Call C function
	c_dnormals(&normals[0,0],&xyz[0,0],&conec[0,0],nel,ndim,ncon)
	# Return
	return normals

@cr('math.normals')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def normals(real[:,:] xyz, int[:,:] conec):
	r'''
	Compute the cell normals given a list 
	of elements.

	Args:
		xyz (np.ndarray):   node positions
		conec (np.ndarray): connectivity array

	Returns:
		np.ndarray: cell normals
	'''
	if real is double:
		return _dnormals(xyz,conec)
	else:
		return _snormals(xyz,conec)

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float32_t,ndim=2] _seuclidean_d(float[:,:] X):
	'''
	Compute Euclidean distances between simulations.

	In:
		- X: NxM Data matrix with N points in the mesh for M simulations
	Returns:
		- D: MxM distance matrix 
	'''
	# Initialize
	cdef int n = X.shape[0], m = X.shape[1]
	cdef np.ndarray[np.float32_t,ndim=2] D = np.zeros((m,m),dtype=np.float32)
	# Call C function
	c_seuclidean_d(&D[0,0],&X[0,0],n,m);
	# Return the distance matrix
	return D

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _deuclidean_d(double[:,:] X):
	'''
	Compute Euclidean distances between simulations.

	In:
		- X: NxM Data matrix with N points in the mesh for M simulations
	Returns:
		- D: MxM distance matrix 
	'''
	# Initialize
	cdef int n = X.shape[0], m = X.shape[1]
	cdef np.ndarray[np.double_t,ndim=2] D = np.zeros((m,m),dtype=np.double)
	# Call C function
	c_deuclidean_d(&D[0,0],&X[0,0],n,m);
	# Return the distance matrix
	return D

@cr('math.euclidean_d')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def euclidean_d(real[:,:] X):
	r'''
	Compute the Euclidean distances between simulations.

	Args:
		X (np.ndarray): NxM Data matrix with N points in the mesh for M simulations

	Returns:
		np.ndarray: MxM distance matrix 
	'''
	if real is double:
		return _deuclidean_d(X)
	else:
		return _seuclidean_d(X)


@cr('math.edge_to_cells')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def edge_to_cells(int[:,:] conec):
	r'''
	Build a dictionary that maps each edge to the cells that share it.

	Args:
		conec (np.ndarray): connectivity array

	Returns:
		defaultdic: edges to cells connectivity dictionary
	'''
	cdef int i, cid, v1, v2, nnodcells, ncells = conec.shape[0], 
	cdef int[:] cell_nodes
	cdef object edge_to_cells = defaultdict(set)

	for cid in range(ncells):
		# Get the nodes of the cell
		cell_nodes = conec[cid]
		nnodcells  = len(cell_nodes)
		for i in range(nnodcells):
			# We are assuming the nodes are ciclically ordered.
			v1, v2 = sorted([cell_nodes[i], cell_nodes[(i+1) % nnodcells]]) # Sort IDs
			edge_to_cells[(v1, v2)].add(cid)  # Associate the cell with the edge

	return edge_to_cells


@cr('math.neighbors_dict')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def neighbors_dict(object edge_dict):
	'''
	Build a dictionary that maps each cell to its neighbors.

	Args:
		edge_dict (dict): Dictionary mapping edges to cells sharing that edge.

	Returns:
		dict: cell to neighbours dictionary
	'''
	cdef int c1, c2
	cdef object cells, neighbors_dict = defaultdict(set)

	for _, cells in edge_dict.items():
		cells = list(cells)
		if len(cells) == 2:  # If there are two cells sharing the edge
			c1, c2 = cells
			neighbors_dict[c1].add(c2)
			neighbors_dict[c2].add(c1)

	return neighbors_dict


@cr('math.fix_coherence')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def fix_normals_coherence(real[:,:] normals, object edge_dict, object adjacency, int num_cells):
	'''
	Ensure the coherence of the normals of the cells.
	'''
	cdef int i, j, k, current, neighbor, count = 0, n_border
	cdef list faces, border_cells = set(), queue
	cdef np.ndarray[np.double_t,ndim=1] border_normals, avg_internal_normal = np.zeros((normals.shape[1],),np.double)
	cdef np.ndarray[np.npy_bool,ndim=1] visited = np.zeros(num_cells, dtype=bool)

    # Find the cells that are on the border
	for _, faces in edge_dict.items():
		if len(faces) == 1:  # If the edge is on the border
			border_cells.add(faces[0])
	n_border = len(border_cells)

    # Propagate the normals using a BFS algorithm
	queue   = deque([next(iter(border_cells))])  # Start from a border cell
	visited[queue[0]] = True

	while queue:
		current = queue.popleft()
		for neighbor in adjacency[current]:
			if not visited[neighbor]:
				# Check if the normals are consistent
				if np.dot(normals[current], normals[neighbor]) < 0:
					for j in normals.shape[1]:
						normals[neighbor,j] *= -1  # Invert the normal

				visited[neighbor] = True
				queue.append(neighbor)

	# Adjust the normals of the border cells
	border_normals = np.zeros((n_border,),np.double)
	for k,i in enumerate(border_cells):
		for j in normals.shape[1]:
			border_normals[k,j] = normals[i,j]
			avg_internal_normal[j] += normals[i,j]
		count += 1
	
	for j in normals.shape[1]:
		avg_internal_normal[j] /= <double>(count)

    # If the average normal of the border cells is pointing inwards, invert all the normals
	if np.dot(np.mean(border_normals, axis=0), avg_internal_normal) < 0:
		for i in border_cells:
			for j in normals.shape[1]:
				normals[i,j] *= -1

	return normals


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.float_t,ndim=2] _sedge_normals(float[:,:] xyz, float[:] cell_normal, int num_nodes):
	'''
	Compute the edge normals (pointing outwards) of a cell given the nodes of the cell, the number of nodes and the cell normal.

	In:
		- xyz: Array of the node coordinates of the cell
		- num_nodes: Number of nodes of the cell
		- cell_normal: Normal to the plane of the cell

	Returns:
		- edge_normals: List of the edge normals of the cell
	'''
	cdef int i, j, nnodes = xyz.shape[0], ndim = cell_normal.shape[0]
	cdef float[:] edge_normal
	cdef np.ndarray[np.float_t,ndim=1] v1           = np.zeros((ndim,),np.float)
	cdef np.ndarray[np.float_t,ndim=1] v2           = np.zeros((ndim,),np.float)
	cdef np.ndarray[np.float_t,ndim=1] edge         = np.zeros((ndim,),np.float)
	cdef np.ndarray[np.float_t,ndim=1] midpoint     = np.zeros((ndim,),np.float)
	cdef np.ndarray[np.float_t,ndim=2] edge_normals = np.zeros((nnodes,ndim),np.float)
	# Iterate over each edge of the cell
	for i in range(nnodes):
		for j in range(ndim):
			v1[j]   = xyz[i,j]
			v2[j]   = xyz[(i + 1) % num_nodes,j]  # Get the edge vertices
			edge[j] = v2[j] - v1[j]  # Get the edge vector

		edge_normal  = np.cross(edge, cell_normal)  # Compute the edge normal
		edge_normal /= np.linalg.norm(edge_normal)  # Normalize the edge normal

		# Ensure the edge normal is pointing outwards (assumes convex polygon)
		for j in range(ndim):
			midpoint[j] = (v1[j] + v2[j])/2. - xyz[(i+2) % num_nodes,j]

		if np.dot(midpoint, edge_normal) < 0:
			for j in range(ndim):
				edge_normal[j] *= -1.

		edge_normals[i,:] = edge_normal

	return edge_normals

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef np.ndarray[np.double_t,ndim=2] _dedge_normals(double[:,:] xyz, double[:] cell_normal, int num_nodes):
	'''
	Compute the edge normals (pointing outwards) of a cell given the nodes of the cell, the number of nodes and the cell normal.

	In:
		- xyz: Array of the node coordinates of the cell
		- num_nodes: Number of nodes of the cell
		- cell_normal: Normal to the plane of the cell

	Returns:
		- edge_normals: List of the edge normals of the cell
	'''
	cdef int i, j, nnodes = xyz.shape[0], ndim = cell_normal.shape[0]
	cdef double[:] edge_normal
	cdef np.ndarray[np.float_t,ndim=1] v1            = np.zeros((ndim,),np.float)
	cdef np.ndarray[np.float_t,ndim=1] v2            = np.zeros((ndim,),np.float)
	cdef np.ndarray[np.double_t,ndim=1] edge         = np.zeros((ndim,),np.double)
	cdef np.ndarray[np.double_t,ndim=1] midpoint     = np.zeros((ndim,),np.double)
	cdef np.ndarray[np.double_t,ndim=2] edge_normals = np.zeros((nnodes,ndim),np.double)
	# Iterate over each edge of the cell
	for i in range(nnodes):
		for j in range(ndim):
			v1[j]   = xyz[i,j]
			v2[j]   = xyz[(i + 1) % num_nodes,j]  # Get the edge vertices
			edge[j] = v2[j] - v1[j]  # Get the edge vector

		edge_normal  = np.cross(edge, cell_normal)  # Compute the edge normal
		edge_normal /= np.linalg.norm(edge_normal)  # Normalize the edge normal

		# Ensure the edge normal is pointing outwards (assumes convex polygon)
		for j in range(ndim):
			midpoint[j] = (v1[j] + v2[j])/2. - xyz[(i+2) % num_nodes,j]

		if np.dot(midpoint, edge_normal) < 0:
			for j in range(ndim):
				edge_normal[j] *= -1.

		edge_normals[i,:] = edge_normal

	return edge_normals

@cr('math.edge_normals')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def edge_normals(real[:,:] xyz, real[:] cell_normal, int num_nodes):
	'''
	Compute the edge normals (pointing outwards) of a cell given the nodes of the cell, the number of nodes and the cell normal.

	In:
		- xyz: Array of the node coordinates of the cell
		- num_nodes: Number of nodes of the cell
		- cell_normal: Normal to the plane of the cell

	Returns:
		- edge_normals: List of the edge normals of the cell
	'''
	if real is double:
		return _dedge_normals(xyz, cell_normal, num_nodes)
	else:
		return _sedge_normals(xyz, cell_normal, num_nodes)