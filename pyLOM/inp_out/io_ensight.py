#!/usr/bin/env python
#
# pyLOM, IO
#
# Ensight Input/Output
#
# Last rev: 23/07/2021
from __future__ import print_function, division

import numpy as np
import ensightreader

from ..utils.cr     import cr
from ..utils.parall import MPI_RANK, MPI_SIZE, MPI_COMM, MPI_RDONLY, MPI_WRONLY, MPI_CREATE
from ..utils.parall import mpi_file_open, worksplit, mpi_reduce, mpi_bcast

ENSI2ELTYPE = {
	'tria3'  : 2, # Triangular cell
	'quad4'  : 3, # Quadrangular cell
	'tetra4' : 4, # Tetrahedral cell
	'penta6' : 6, # Linear prism
	'hexa8'  : 5, # Hexahedron
}

## HELPER FUNCTIONS ##

def str_to_bin(string):
	return ('%-80s'%(string)).encode('utf-8')

def bin_to_str(binary):
	return binary[:-1].decode('utf-8').strip()

def int_to_bin(integer,b=4):
	return int(integer).to_bytes(b,'little')

def bin_to_int(integer):
	return int.from_bytes(integer,'little')

def elnod(eltype):
	if 'tria3'  in eltype: return 3
	if 'tria6'  in eltype: return 6
	if 'quad4'  in eltype: return 4
	if 'quad8'  in eltype: return 8
	if 'tetra4' in eltype: return 4
	if 'penta6' in eltype: return 6
	if 'hexa8'  in eltype: return 8
	return 0

def isBinary(fname):
	f = open(fname,'r')
	try:
		a = f.readline()
		f.close()
		return True if '\x00' in a else False
	except:
		f.close()
		return True


## FUNCTIONS ##
@cr('EnsightIO.readCase')
def Ensight_readCase(fname,rank=MPI_RANK):
	'''
	Read an Ensight Gold case file.
	'''
	# Only one rank reads the file
	if MPI_RANK == rank or MPI_SIZE == 1:
		# Open file for reading
		f        = open(fname,'r')
		lines    = [line.strip() for line in f.readlines() if not '#' in line and not len(line.strip())==0]
		has_time = 'TIME' in lines
		# Variables section
		idstart = lines.index('VARIABLE')+1
		idend   = lines.index('TIME') if has_time else len(lines)
		varList = []
		for ii in range(idstart,idend):
			varList.append({})
			# Name
			varList[-1]['name'] = lines[ii].split()[-2]
			# Dimensions
			varList[-1]['dims'] = -1 
			if 'scalar'      in lines[ii]: varList[-1]['dims'] = 1
			if 'vector'      in lines[ii]: varList[-1]['dims'] = 3
			if 'tensor symm' in lines[ii]: varList[-1]['dims'] = 6
			if 'tensor asym' in lines[ii]: varList[-1]['dims'] = 9
			# File
			varList[-1]['file'] = lines[ii].split()[-1]
		# Timesteps
		if has_time:
			idstart   = lines.index('TIME') + 6
			idend     = len(lines)
			timesteps = np.array([float(l) for ii in range(idstart,idend) for l in lines[ii].split()],dtype=np.double)
		else:
			timesteps = np.array([],dtype=np.double)
		# Close file
		f.close()
	# Broadcast to other ranks if needed
	if MPI_SIZE > 1:
		varList, timesteps = mpi_bcast((varList,timesteps),rank=rank)
	# Return
	return varList, timesteps

@cr('EnsightIO.readCase')
def Ensight_readCase2(fname,rank=MPI_RANK):
	'''
	Read an Ensight Gold case file.

	Use ensight-reader library.
	'''
	# Only one rank reads the file
	if MPI_RANK == rank or MPI_SIZE == 1:
		case = ensightreader.read_case(fname)	
	# Broadcast to other ranks if needed
	if MPI_SIZE > 1:
		cases = mpi_bcast(case,rank=rank)
	# Return
	return case


@cr('EnsightIO.writeCase')
def Ensight_writeCase(fname,geofile,varList,timesteps,rank=MPI_RANK):
	'''
	Write an Ensight Gold case file.
	'''
	# Only one rank writes the file
	if MPI_RANK == rank or MPI_SIZE == 1:
		# Open file for writing
		f = open(fname,'w')
		f.write('FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel: 1  %s\n\nVARIABLE\n'%geofile)
		# Variables section
		for var in varList:
			dims = 'scalar' if var['dims'] == 1 else 'vector'
			if var['dims'] == 6: dims = 'tensor symm' 
			if var['dims'] == 9: dims = 'tensor asym' 
			f.write('%s per node:  1   %s  %s\n'%(dims,var['name'],var['file']))
		# Timesteps
		f.write('\nTIME\n')
		f.write('time set:              1\n')
		f.write('number of steps:       %d\n'%timesteps.shape[0])
		f.write('filename start number: 1\n')
		f.write('filename increment:    1\n')
		f.write('time values:\n')
		timesteps.tofile(f,sep='\n',format='%f')
		# Close file
		f.close()


@cr('EnsightIO.readGeo')
def Ensight_readGeo(fname):
	'''
	Read an Ensight Gold Geometry file in either
	ASCII or binary format.
	'''
	# Select which function to run
	readGeo = Ensight_readGeoBIN if isBinary(fname) else Ensight_readGeoASCII
	# Run function
	return readGeo(fname)

def Ensight_readGeoBIN(fname):
	'''
	SOURCE OF GEO FILE FORMAT
	from: http://www-vis.lbl.gov/NERSC/Software/ensight/docs/OnlineHelp/UM-C11.pdf
	All Data is plainly assumed to be exported as C binary Little Endian!

	C Binary                                            80 chars
	description line 1                                  80 chars
	description line 2                                  80 chars
	node id <off/given/assign/ignore>                   80 chars
	element id <off/given/assign/ignore>                80 chars
	part                                                80 chars
	#                                                    1 int
	description line                                    80 chars  (Name of current part)
	coordinates                                         80 chars
	nn                                                   1 int    (Count of xyz coordinates)
	x_n1 x_n2 ... x_nn                                  nn floats
	y_n1 y_n2 ... y_nn                                  nn floats
	z_n1 z_n2 ... z_nn                                  nn floats
	element type                                        80 chars  (nr of cornerpoints)
	ne                                                   1 int
	n1_e1 n2_e1 ...                                     np_e1
	n1_e2 n2_e2 ...                                     np_e2
	 .
	 .
	n1_ne n2_ne ... np_ne                            ne*np ints
	'''
	# Open file for reading
	f = open(fname,'rb')
	# Read Ensight header
	header_bin = f.read(80*8+4*2) # 8 80 bytes char + 2 4 byte integer
	# Parse the header
	header = {}
	header['descr']  = '%s\n%s' % (bin_to_str(header_bin[80:2*80]),bin_to_str(header_bin[2*80:3*80]))
	header['nodeID'] = bin_to_str(header_bin[3*80:4*80]).replace('node id ','')
	header['elemID'] = bin_to_str(header_bin[4*80:5*80]).replace('element id ','')
	header['partID'] = bin_to_int(header_bin[6*80:6*80+4]) # Part ID
	header['partNM'] = bin_to_str(header_bin[6*80+4:7*80+4]) # Part name
	# Read the node coordinates
	nnod = bin_to_int(header_bin[8*80+4:8*80+4+4])
	xyz  = np.ascontiguousarray(np.fromfile(f,dtype=np.float32,count=3*nnod).astype(np.double).reshape((nnod,3),order='F'))
	# Read Ensight header
	header_bin = f.read(80+4) # 80 bytes char + 4 byte integer
	# Parse the header
	header['eltype'] = bin_to_str(header_bin[:80])
	# Read element connectivity
	nel   = bin_to_int(header_bin[80:80+4])
	nnel  = elnod(header['eltype'])
	conec = np.ascontiguousarray(np.fromfile(f,dtype=np.int32,count=nnel*nel).reshape((nel,nnel),order='C'))
	# Close the field
	f.close()
	# Return
	return xyz, conec, header

def Ensight_readGeoASCII(fname):
	'''
	SOURCE OF GEO FILE FORMAT
	from: http://www-vis.lbl.gov/NERSC/Software/ensight/docs/OnlineHelp/UM-C11.pdf
	All Data is plainly assumed to be exported as C binary Little Endian!

	description line 1                                  80 chars
	description line 2                                  80 chars
	node id <off/given/assign/ignore>                   80 chars
	element id <off/given/assign/ignore>                80 chars
	part                                                80 chars
	#                                                    1 int
	description line                                    80 chars  (Name of current part)
	coordinates                                         80 chars
	nn                                                   1 int    (Count of xyz coordinates)
	x_n1 x_n2 ... x_nn                                  nn floats
	y_n1 y_n2 ... y_nn                                  nn floats
	z_n1 z_n2 ... z_nn                                  nn floats
	element type                                        80 chars  (nr of cornerpoints)
	ne                                                   1 int
	n1_e1 n2_e1 ...                                     np_e1
	n1_e2 n2_e2 ...                                     np_e2
	 .
	 .
	n1_ne n2_ne ... np_ne                            ne*np ints
	'''
	# Open file for reading
	f = open(fname,'r')
	# Read Ensight header
	header = {}
	header['descr']  = '%s\n%s' % (f.readline().strip(),f.readline().strip())
	header['nodeID'] = f.readline().strip().replace('node id ','')
	header['elemID'] = f.readline().strip().replace('element id ','')
	f.readline() # Part
	header['partID'] = int(f.readline()) # Part ID
	header['partNM'] = f.readline().strip() # Part name
	# Read the node coordinates
	f.readline() # coordinates
	nnod = int(f.readline())
	# Skip the node id list, if present
	if header['nodeID'] == 'given':
		order = np.genfromtxt(f,max_rows=nnod).astype(np.int32)-1
	xyz = np.ascontiguousarray(np.genfromtxt(f,max_rows=3*nnod).astype(np.double).reshape((nnod,3),order='F'))
	if header['nodeID'] == 'given':
		xyz = xyz[order,:]
	# Read Ensight header
	header['eltype'] = f.readline().strip()
	# Read element connectivity
	nel   = int(f.readline())
	nnel  = elnod(header['eltype'])
	# Skip the node id list, if present
	if header['elemID'] == 'given':
		order = np.genfromtxt(f,max_rows=nel).astype(np.int32)-1
	conec = np.ascontiguousarray(np.genfromtxt(f,max_rows=nel).astype(np.int32).reshape((nel,nnel),order='F'))
	if header['elemID'] == 'given':
		conec = conec[order,:]
	# Close the file
	f.close()
	# Return
	return xyz, conec, header

@cr('EnsightIO.readGeo')
def Ensight_readGeo2(geofile,part_id):
	'''
	Read an Ensight Gold Geometry using the
	ensight-reader library
	'''
	# Recover the part
	part = geofile.get_part_by_id(part_id)
	# Start reading geometry file
	f = open(geofile.file_path,'rb')
	# Read node positions
	nnod = part.number_of_nodes
	xyz  = np.ascontiguousarray(part.read_nodes(f))
	# Create connectivity array
	nelem = []
	elem  = []
	nnel  = 0
	for block in part.element_blocks:
		nelem.append(block.number_of_elements)
		elem.append(block.element_type.value)
		nnel = max(nnel,block.element_type.nodes_per_element)
	conec  = np.zeros((np.sum(nelem),nnel),np.int32)
	eltype = np.ones((np.sum(nelem),),np.int32)
	# Read connectivity
	for iblock,block in enumerate(part.element_blocks):
		nel1 = np.sum(nelem[:iblock]) if iblock > 0 else 0
		nel2 = nelem[iblock]
		nnel = block.element_type.nodes_per_element
		# Subtract 1 to connectivity due to python indexing
		conec[nel1:nel2,:nnel] = np.ascontiguousarray(block.read_connectivity(f)) - 1
		eltype[nel1:nel2] *= ENSI2ELTYPE[elem[iblock]]
	# Close the file
	f.close()
	# Return
	return xyz, conec, eltype


@cr('EnsightIO.readGeo')
def Ensight_writeGeo(fname,xyz,conec,header):
	'''
	SOURCE OF GEO FILE FORMAT
	from: http://www-vis.lbl.gov/NERSC/Software/ensight/docs/OnlineHelp/UM-C11.pdf
	All Data is plainly assumed to be exported as C binary Little Endian!

	C Binary                                            80 chars
	description line 1                                  80 chars
	description line 2                                  80 chars
	node id <off/given/assign/ignore>                   80 chars
	element id <off/given/assign/ignore>                80 chars
	part                                                80 chars
	#                                                    1 int
	description line                                    80 chars  (Name of current part)
	coordinates                                         80 chars
	nn                                                   1 int    (Count of xyz coordinates)
	x_n1 x_n2 ... x_nn                                  nn floats
	y_n1 y_n2 ... y_nn                                  nn floats
	z_n1 z_n2 ... z_nn                                  nn floats
	element type                                        80 chars  (nr of cornerpoints)
	ne                                                   1 int
	n1_e1 n2_e1 ...                                     np_e1
	n1_e2 n2_e2 ...                                     np_e2
	 .
	 .
	n1_ne n2_ne ... np_ne                            ne*np ints
	'''
	# Open file for writing
	f = open(fname,'wb')
	# Write Ensight header
	header_bin = str_to_bin('C Binary')
	for descr in header['descr'].split('\n'):
		header_bin += str_to_bin(descr)
	header_bin += str_to_bin('node id %s'%header['nodeID'])
	header_bin += str_to_bin('element id %s'%header['elemID'])
	header_bin += str_to_bin('part')
	header_bin += int_to_bin(header['partID'])
	header_bin += str_to_bin(header['partNM'])
	header_bin += str_to_bin('coordinates')
	# Write the node coordinates
	nnod = xyz.shape[0]
	f.write(header_bin+int_to_bin(nnod))
	xyz.astype(np.float32).reshape((3*nnod,),order='F').tofile(f)
	# Write Ensight header
	f.write(str_to_bin(header['eltype'])+int_to_bin(conec.shape[0]))
	# Read element connectivity
	nel  = conec.shape[0]
	nnel = conec.shape[1]
	conec.astype(np.int32).reshape((nnel*nel,),order='C').tofile(f)
	# Close the field
	f.close()


@cr('EnsightIO.readField')
def Ensight_readField(fname,dims=1,nnod=-1,parallel=False):
	'''
	Read an Ensight Gold field file in either
	ASCII or binary format.
	'''
	if parallel and isBinary(fname):
		readField = Ensight_readFieldMPIO
	else:
		readField = Ensight_readFieldBIN if isBinary(fname) else Ensight_readFieldASCII
	return readField(fname,dims,nnod)

def Ensight_readFieldBIN(fname,dims=1,nnod=-1):
	'''
	ENSIGHT GOLD SCALAR
	from: http://www-vis.lbl.gov/NERSC/Software/ensight/docs/OnlineHelp/UM-C11.pdf
	
	BEGIN TIME STEP
	description line 1          80 chars
	part                        80 chars
	#                            1 int
	block                       80 chars
	s_n1 s_n2 ... s_nn          nn floats	
	'''
	# Open file for reading
	f = open(fname,'rb')
	# Read Ensight header
	header_bin = f.read(80*3+4) # 3 80 bytes char + 4 byte integer
	# Parse the header
	header = {}
	header['descr']  = bin_to_str(header_bin[:80])         # Description
	header['partID'] = bin_to_int(header_bin[2*80:2*80+4]) # Part ID
#	header['partNM'] = bin_to_str(header_bin[2*80+4:3*80]) # Part name 
	# Read the field
	field = np.fromfile(f,dtype=np.float32).astype(np.double)
	# Close the field
	f.close()
	# Return
	return np.ascontiguousarray(field) if dims == 1 else np.ascontiguousarray(field.reshape((field.shape[0]//dims,dims),order='F')), header

def Ensight_readFieldASCII(fname,dims=1,nnod=-1):
	'''
	ENSIGHT GOLD SCALAR
	from: http://www-vis.lbl.gov/NERSC/Software/ensight/docs/OnlineHelp/UM-C11.pdf
	
	BEGIN TIME STEP
	description line 1          80 chars
	part                        80 chars
	#                            1 int
	block                       80 chars
	s_n1 s_n2 ... s_nn          nn floats	
	'''
	# Open file for reading
	f = open(fname,'r')
	# Read Ensight header
	header = {}
	header['descr']  = f.readline().strip() # Description
	f.readline() # Part
	header['partID'] = int(f.readline())    # Part ID
#	header['partNM'] = ''                   # Part name
	f.readline() # coordinates
	# Read the field
	field = np.genfromtxt(f).astype(np.double)
	# Close the field
	f.close()
	# Return
	return np.ascontiguousarray(field) if dims == 1 else np.ascontiguousarray(field.reshape((field.shape[0]//dims,dims),order='F')), header

def Ensight_readFieldMPIO(fname,dims=1,nnod=-1):
	'''
	ENSIGHT GOLD SCALAR
	from: http://www-vis.lbl.gov/NERSC/Software/ensight/docs/OnlineHelp/UM-C11.pdf
	
	BEGIN TIME STEP
	description line 1          80 chars
	part                        80 chars
	#                            1 int
	block                       80 chars
	s_n1 s_n2 ... s_nn          nn floats	
	'''
	# Open file for reading
	f = mpi_file_open(MPI_COMM,fname,MPI_RDONLY)
	# Read Ensight header
	header_sz  = 80*3+4  # 3 80 bytes char + 4 byte integer
	header_bin = np.ndarray((header_sz,),dtype='b')
	f.Read_at_all(0,header_bin)
	header_bin = bytes(header_bin.view('S%d'%header_sz))
	# Parse the header
	header = {}
	header['descr']  = bin_to_str(header_bin[:80])         # Description
	header['partID'] = bin_to_int(header_bin[2*80:2*80+4]) # Part ID
#	header['partNM'] = bin_to_str(header_bin[2*80+4:3*80]) # Part name 
	# Use a simple worksplit to see where everyone reads
	istart,iend = worksplit(0,nnod,MPI_RANK,nWorkers=MPI_SIZE)
	# Create flattened output array
	field = np.ndarray(((iend-istart),dims) if dims > 1 else ((iend-istart),),np.double)
	# Read the field
	if dims > 1:
		for idim in range(dims):
			aux = np.ndarray((iend-istart,),np.float32)
			f.Read_at(header_sz+(istart+idim*nnod)*4,aux)
			field[:,idim] = np.ascontiguousarray(aux)
	else:
		f.Read_at(header_sz+istart*4,field)
		field = np.ascontiguousarray(field)		
	# Close the file
	f.Close()
	# Return
	return field, header

@cr('EnsightIO.readField')
def Ensight_readField2(variable,part_id):
	'''
	Read an Ensight Gold field file using
	the ensightreader library.
	'''
	f = open(variable.file_path,'rb')
	field = np.ascontiguousarray(variable.read_node_data(f,part_id) if variable.variable_location == ensightreader.VariableLocation.PER_NODE else variable.read_element_data(f,part_id))
	# Close the file
	f.close()
	# Return
	return field


@cr('EnsightIO.writeField')
def Ensight_writeField(fname,field,header,parallel=False):
	'''
	Write an Ensight Gold field file in binary format.
	'''
	writeField = Ensight_writeFieldBIN if not parallel else Ensight_writeFieldMPIO
	return writeField(fname,field,header)

def Ensight_writeFieldBIN(fname,field,header):
	'''
	ENSIGHT GOLD SCALAR
	from: http://www-vis.lbl.gov/NERSC/Software/ensight/docs/OnlineHelp/UM-C11.pdf
	
	BEGIN TIME STEP
	description line 1          80 chars
	part                        80 chars
	#                            1 int
	block                       80 chars
	s_n1 s_n2 ... s_nn          nn floats	
	'''
	# Open file for writing
	f = open(fname,'wb')
	# Write Ensight header
	header_bin  = str_to_bin(header['descr'])
	header_bin += str_to_bin('part')
	header_bin += int_to_bin(header['partID'])
	header_bin += str_to_bin('coordinates')
	f.write(header_bin)
	# Write the field
	nrows = field.shape[0]
	ncols = 1 if len(field.shape) == 1 else field.shape[1]
	field.astype(np.float32).reshape((ncols*nrows,),order='F').tofile(f)
	# Close the field
	f.close()

def Ensight_writeFieldMPIO(fname,field,header):
	'''
	ENSIGHT GOLD SCALAR
	from: http://www-vis.lbl.gov/NERSC/Software/ensight/docs/OnlineHelp/UM-C11.pdf
	
	BEGIN TIME STEP
	description line 1          80 chars
	part                        80 chars
	#                            1 int
	block                       80 chars
	s_n1 s_n2 ... s_nn          nn floats	
	'''
	# Open file for writing
	f = mpi_file_open(MPI_COMM,fname,MPI_WRONLY|MPI_CREATE)
	# Write Ensight header
	header_sz   = 80*3+4  # 3 80 bytes char + 4 byte integer
	header_bin  = str_to_bin(header['descr'])
	header_bin += str_to_bin('part')
	header_bin += int_to_bin(header['partID'])
	header_bin += str_to_bin('coordinates')
	f.Write_at_all(0,np.frombuffer(header_bin,np.int8))
	# Obtain the total number of nodes
	nrows  = field.shape[0]
	ncols  = 1 if len(field.shape) == 1 else field.shape[1]
	nrowsT = mpi_reduce(nrows,op='sum',all=True)
	# Worksplit
	istart, iend = worksplit(0,nrowsT,MPI_RANK,nWorkers=MPI_SIZE)
	# Write the field
	if ncols == 1:
		f.Write_at(header_sz+istart*4,field.astype(np.float32))
	else:
		for icol in range(ncols):
			f.Write_at(header_sz+(istart+icol*ncols)*4,field[:,icol].astype(np.float32))
	# Close the field
	f.Close()
