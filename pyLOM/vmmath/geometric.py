#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - geometry.
#
# Last rev: 27/10/2021
from __future__ import print_function, division

import numpy as np
from collections import defaultdict, deque

from ..utils.gpu import cp
from ..utils     import cr_nvtx as cr, mpi_reduce


@cr('math.euclidean_d')
def euclidean_d(X):
	'''
	Compute Euclidean distances between simulations.

	In:
		- X: NxM Data matrix with N points in the mesh for M simulations
	Returns:
		- D: MxM distance matrix 
	'''
	p = cp if type(X) is cp.ndarray else np
	# Extract dimensions
	_,M = X.shape
	# Initialize distance matrix
	D = p.zeros((M,M),X.dtype)
	for i in range(M):
		for j in range(i+1,M,1):
			# Local sum on the partition
			d2 = p.sum((X[:,i]-X[:,j])*(X[:,i]-X[:,j]))
			# Global sum over the partitions
			dG = p.sqrt(mpi_reduce(d2,all=True))
			# Fill output
			D[i,j] = dG
			D[j,i] = dG
	# Return the mdistance matrix
	return D


@cr('math.cellCenters')
def cellCenters(xyz:np.ndarray,conec:np.ndarray) -> np.ndarray:
	r'''
	Compute the cell centers given a list 
	of elements.

	Args:
		xyz (np.ndarray):   node positions
		conec (np.ndarray): connectivity array

	Returns:
		np.ndarray: center positions
	'''
	p = cp if type(xyz) is cp.ndarray else np
	xyz_cen = p.zeros((conec.shape[0],xyz.shape[1]),xyz.dtype)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c = conec[ielem,conec[ielem,:]>=0]
		xyz_cen[ielem,:] = p.mean(xyz[c,:],axis=0)
	return xyz_cen


@cr('math.normals')
def normals(xyz:np.ndarray,conec:np.ndarray) -> np.ndarray:
	r'''
	Compute the cell normals given a list 
	of elements.

	Args:
		xyz (np.ndarray):   node positions
		conec (np.ndarray): connectivity array

	Returns:
		np.ndarray: cell normals
	'''
	p = cp if type(xyz) is cp.ndarray else np
	normals = p.zeros(((conec.shape[0],3)),xyz.dtype)
	for ielem in range(conec.shape[0]):
		# Get the values of the field and the positions of the element
		c     = conec[ielem,conec[ielem,:]>=0]
		xyzel =  xyz[c,:]
		# Compute centroid
		cen  = p.mean(xyzel,axis=0)
		# Compute normal
		for inod in range(len(c)):
			u = xyzel[inod]   - cen
			v = xyzel[inod-1] - cen
			normals[ielem,:] += 0.5*p.cross(u,v)
	return normals


@cr('math.edge_to_cells')
def edge_to_cells(conec:np.ndarray) -> dict:
	r'''
	Build a dictionary that maps each edge to the cells that share it.

	Args:
		conec (np.ndarray): connectivity array

	Returns:
		dict: edges to cells connectivity dictionary
	'''
	ncells = conec.shape[0]
	edge_to_cells = defaultdict(set)

	for cell_id in range(ncells):
		# Get the nodes of the cell
		cell_nodes = conec[cell_id]
		for i in range(len(cell_nodes)):
			# We are assuming the nodes are ciclically ordered.
			v1, v2 = sorted([cell_nodes[i], cell_nodes[(i+1) % len(cell_nodes)]]) # Sort IDs
			# Edges are undirected (v1, v2) == (v2, v1)
			edge_to_cells[(v1, v2)].add(cell_id)  # Associate the cell with the edge
			edge_to_cells[(v2, v1)].add(cell_id)  # Associate the cell with the edge
	return edge_to_cells


@cr('math.cell_adjacency')
def cell_adjacency(edge_dict) -> dict:
	'''
	Build a dictionary that maps each cell to its neighbors.

	Args:
		edge_dict (dict): Dictionary mapping edges to cells sharing that edge.

	Returns:
		dict: cell to neighbours dictionary
	'''
	cell_adjacency = defaultdict(set)

	for _, cells in edge_dict.items():
		cells = list(cells)
		if len(cells) == 2:  # If there are two cells sharing the edge
			c1, c2 = cells
			cell_adjacency[c1].add(c2)
			cell_adjacency[c2].add(c1)

	return cell_adjacency


@cr('math.fix_coherence')
def fix_normals_coherence(normals, edge_dict, adjacency, num_cells) -> np.ndarray:
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
	# Find the cells that are on the border
	border_cells = set()
	for _, faces in edge_dict.items():
		if len(faces) == 1:  # If the edge is on the border
			border_cells.add(next(iter(faces)))  # Add the cell to the border cells

	# Propagate the normals using a BFS algorithm
	visited = np.zeros(num_cells, dtype=bool)
	queue = deque([next(iter(border_cells))])  # Start from a border cell
	visited[queue[0]] = True

	while queue:
		current = queue.popleft()
		for neighbor in adjacency[current]:
			if not visited[neighbor]:
				# Check if the normals are consistent
				if np.dot(normals[current], normals[neighbor]) < 0:
					normals[neighbor] *= -1  # Invert the normal

				visited[neighbor] = True
				queue.append(neighbor)

	# Adjust the normals of the border cells
	border_normals = normals[list(border_cells)]
	avg_internal_normal = np.mean(normals[~np.isin(range(num_cells), list(border_cells))], axis=0)

	# If the average normal of the border cells is pointing inwards, invert all the normals
	if np.dot(np.mean(border_normals, axis=0), avg_internal_normal) < 0:
		for i in border_cells:
			normals[i] *= -1

	return normals



@cr('math.wall_normals')
def wall_normals(nodes_idx, nodes_xyz, surf_normal) -> list:
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
	num_nodes = len(nodes_xyz)
	wall_normals = []
	cell_edges = []
	# Iterate over each edge of the cell
	for i in range(num_nodes):
		v1, v2 = nodes_xyz[i], nodes_xyz[(i + 1) % num_nodes]  # Get the edge vertices
		edge = tuple([nodes_idx[i], nodes_idx[(i + 1) % num_nodes]])
		edge_vector = v2 - v1  # Get the edge vector

		edge_normal = np.cross(edge_vector, surf_normal)  # Compute the edge normal
		edge_normal /= np.linalg.norm(edge_normal)  # Normalize the edge normal

		# Ensure the edge normal is pointing outwards (assumes convex polygon)
		auxiliary_node = nodes_xyz[(i+2) % num_nodes]
		midpoint = (v1 + v2) / 2

		if np.dot(midpoint - auxiliary_node, edge_normal) < 0:
			edge_normal *= -1

		wall_normals.append(edge_normal)
		cell_edges.append(edge)

	return cell_edges, wall_normals
