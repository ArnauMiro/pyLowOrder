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


@cr('math.cell_adjacency')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def cell_adjacency(object edge_dict):
	'''
	Build a dictionary that maps each cell to its neighbors.

	Args:
		edge_dict (dict): Dictionary mapping edges to cells sharing that edge.

	Returns:
		dict: cell to neighbours dictionary
	'''
	cdef int c1, c2
	cdef object cells, cell_adjacency = defaultdict(set)

	for _, cells in edge_dict.items():
		cells = list(cells)
		if len(cells) == 2:  # If there are two cells sharing the edge
			c1, c2 = cells
			cell_adjacency[c1].add(c2)
			cell_adjacency[c2].add(c1)

	return cell_adjacency


@cr('math.fix_coherence')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def fix_normals_coherence(real[:,:] normals, object edge_dict, object adjacency, int num_cells):
	'''
	Ensure that the normals of the cells are coherent. (i.e. they point all in the same direction).
	In:
		- normals: Array of normals of the cells
		- edge_dict: Dictionary mapping edges to cells sharing that edge.
		- adjacency: Dictionary mapping cells to their neighbors.
		- num_cells: Number of cells in the mesh
	Returns:
		- normals: Array of normals of the cells
	'''
	cdef int i, j, k, current, neighbor, count = 0, n_border, ndim = normals.shape[1]
	cdef set faces, border_cells = set()
	cdef object queue
	cdef np.ndarray[np.double_t,ndim=2] border_normals
	cdef np.ndarray[np.double_t,ndim=1] avg_internal_normal = np.zeros((ndim,),np.double)
	cdef np.ndarray[np.npy_bool,ndim=1] visited = np.zeros(num_cells, dtype=bool)

    # Find the cells that are on the border
	for _, faces in edge_dict.items():
		if len(faces) == 1:  # If the edge is on the border
			border_cells.add(next(iter(faces)))  # Add the cell to the border cells
	n_border = len(border_cells)

    # Propagate the normals using a BFS algorithm
	queue = deque([next(iter(border_cells))])  # Start from a border cell
	visited[queue[0]] = True

	while queue:
		current = queue.popleft()
		for neighbor in adjacency[current]:
			if not visited[neighbor]:
				# Check if the normals are consistent
				if np.dot(normals[current], normals[neighbor]) < 0:
					for j in range(ndim):
						normals[neighbor,j] *= -1  # Invert the normal

				visited[neighbor] = True
				queue.append(neighbor)

	# Adjust the normals of the border cells
	border_normals = np.zeros((n_border,ndim),np.double)
	for k,i in enumerate(border_cells):
		for j in range(ndim):
			border_normals[k,j] = normals[i,j]
		if i not in border_cells:
			for j in range(ndim):
				avg_internal_normal[j] += normals[i,j]
				count += 1
	
	avg_internal_normal /= <double>count

    # If the average normal of the border cells is pointing inwards, invert all the normals
	if np.dot(np.mean(border_normals, axis=0), avg_internal_normal) < 0:
		for i in border_cells:
			for j in ndim:
				normals[i,j] *= -1

	return np.array(normals)


@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _swall_normals(int[:] nodes_idx, float[:,:] nodes_xyz, float[:] surf_normal):
	'''
	Compute the unitary normals to the cell walls (only for 2D cells).
	Example: For a triangle, the wall normals are the three vectors normal to each of the sides.
	The wall normals are always contained in the cell plane, thus are orthogonal themselves to the cell surface normal.
	As a convention, wall normals are always pointing outwards the cell.

	In:
		- nodes_idx: List or array of the node indices of the cell
		- nodes_xyz: List or array of the node coordinates of the cell
		- surf_normal: Normal to the plane of the cell


	Returns:
		- cell_edges: List of graph edges representing the element walls (node indices)
		- wall_normals: List of the unitary wall normals
	'''
	cdef int i, j, num_nodes = nodes_xyz.shape[0], ndim = nodes_xyz.shape[1]
	cdef list wall_normals = [], cell_edges = []
	cdef np.ndarray[np.float32_t,ndim=1] auxiliary_node = np.zeros((ndim,),np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] edge_vector    = np.zeros((ndim,),np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] edge_normal    = np.zeros((ndim,),np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] v1             = np.zeros((ndim,),np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] v2             = np.zeros((ndim,),np.float32)
	cdef np.ndarray[np.float32_t,ndim=2] edge           = np.zeros((ndim,2),np.float32)
	cdef np.ndarray[np.float32_t,ndim=1] midpoint       = np.zeros((ndim,),np.float32)

	# Iterate over each edge of the cell
	for i in range(num_nodes):
		for j in range(ndim): # Get the edge vertices
			v1[j] = nodes_xyz[i,j]
			v2[j] = nodes_xyz[(i + 1) % num_nodes,j]  
			edge[j,0] = v1[j]
			edge[j,1] = v2[j]
		edge_vector = v2 - v1  # Get the edge vector

		edge_normal = np.cross(edge_vector, surf_normal) # Compute the edge normal
		edge_normal /= np.linalg.norm(edge_normal)       # Normalize the edge normal

		# Ensure the edge normal is pointing outwards (assumes convex polygon)
		for j in range(ndim):
			auxiliary_node[j] = nodes_xyz[(i+2) % num_nodes,j]
		midpoint = (v1 + v2) / 2

		if np.dot(midpoint - auxiliary_node, edge_normal) < 0:
			for j in range(ndim):
				edge_normal[j] *= -1

		wall_normals.append(edge_normal)
		cell_edges.append(edge)

	return cell_edges, wall_normals

@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def _dwall_normals(int[:] nodes_idx, double[:,:] nodes_xyz, double[:] surf_normal):
	'''
	Compute the unitary normals to the cell walls (only for 2D cells).
	Example: For a triangle, the wall normals are the three vectors normal to each of the sides.
	The wall normals are always contained in the cell plane, thus are orthogonal themselves to the cell surface normal.
	As a convention, wall normals are always pointing outwards the cell.

	In:
		- nodes_idx: List or array of the node indices of the cell
		- nodes_xyz: List or array of the node coordinates of the cell
		- surf_normal: Normal to the plane of the cell


	Returns:
		- cell_edges: List of graph edges representing the element walls (node indices)
		- wall_normals: List of the unitary wall normals
	'''
	cdef int i, j, num_nodes = nodes_xyz.shape[0], ndim = nodes_xyz.shape[1]
	cdef list wall_normals = [], cell_edges = []
	cdef np.ndarray[np.double_t,ndim=1] auxiliary_node = np.zeros((ndim,),np.double)
	cdef np.ndarray[np.double_t,ndim=1] edge_vector    = np.zeros((ndim,),np.double)
	cdef np.ndarray[np.double_t,ndim=1] edge_normal    = np.zeros((ndim,),np.double)
	cdef np.ndarray[np.double_t,ndim=1] v1             = np.zeros((ndim,),np.double)
	cdef np.ndarray[np.double_t,ndim=1] v2             = np.zeros((ndim,),np.double)
	cdef np.ndarray[np.double_t,ndim=2] edge           = np.zeros((ndim,2),np.double)
	cdef np.ndarray[np.double_t,ndim=1] midpoint       = np.zeros((ndim,),np.double)

	# Iterate over each edge of the cell
	for i in range(num_nodes):
		for j in range(ndim): # Get the edge vertices
			v1[j] = nodes_xyz[i,j]
			v2[j] = nodes_xyz[(i + 1) % num_nodes,j]  
			edge[j,0] = v1[j]
			edge[j,1] = v2[j]
		edge_vector = v2 - v1  # Get the edge vector

		edge_normal = np.cross(edge_vector, surf_normal)  # Compute the edge normal
		edge_normal /= np.linalg.norm(edge_normal)  # Normalize the edge normal

		# Ensure the edge normal is pointing outwards (assumes convex polygon)
		for j in range(ndim):
			auxiliary_node[j] = nodes_xyz[(i+2) % num_nodes,j]
		midpoint = (v1 + v2) / 2

		if np.dot(midpoint - auxiliary_node, edge_normal) < 0:
			for j in range(ndim):
				edge_normal[j] *= -1

		wall_normals.append(edge_normal)
		cell_edges.append(edge)

	return cell_edges, wall_normals

@cr('math.wall_normals')
@cython.initializedcheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def wall_normals(int[:] nodes_idx, real[:,:] nodes_xyz, real[:] surf_normal):
	'''
	Compute the unitary normals to the cell walls (only for 2D cells).
	Example: For a triangle, the wall normals are the three vectors normal to each of the sides.
	The wall normals are always contained in the cell plane, thus are orthogonal themselves to the cell surface normal.
	As a convention, wall normals are always pointing outwards the cell.

	In:
		- nodes_idx: List or array of the node indices of the cell
		- nodes_xyz: List or array of the node coordinates of the cell
		- surf_normal: Normal to the plane of the cell


	Returns:
		- cell_edges: List of graph edges representing the element walls (node indices)
		- wall_normals: List of the unitary wall normals
	'''
	if real is double:
		return _dwall_normals(nodes_idx, nodes_xyz, surf_normal)
	else:
		return _swall_normals(nodes_idx, nodes_xyz, surf_normal)