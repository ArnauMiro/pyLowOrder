#!/usr/bin/env cpython
#
# pyLOM, IO
#
# Ensight Input/Output
#
# Last rev: 24/08/2021
from __future__ import print_function, division

import numpy as np
import ensightreader

cimport numpy as np
cimport cython

from libc.stdio  cimport FILE, fopen, fclose, fread, fwrite, fgets, feof
from libc.stdlib cimport malloc, realloc, free, atoi, atof
from libc.string cimport memchr, strtok, memcpy

from ..utils.cr     import cr
from ..utils.errors import raiseError
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

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef char *reads(char *line, int size, FILE *fin):
	cdef char *l
	cdef int ii
	return fgets(line,size,fin)
#	if l != NULL:
#		for ii in range(size):
#			if line[ii] == 10:
#				line[ii] = '\0'.encode('utf-8')
#				break
#	return l

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef int elnod(object eltype):
	if 'tria3'  in eltype: return 3
	if 'tria6'  in eltype: return 6
	if 'quad4'  in eltype: return 4
	if 'quad8'  in eltype: return 8
	if 'tetra4' in eltype: return 4
	if 'penta6' in eltype: return 6
	if 'hexa8'  in eltype: return 8
	return 0

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
cdef int isBinary(object fname):
	cdef FILE *myfile
	cdef char buff[80+1]

	myfile = fopen(fname.encode('utf-8'),"rb")
	if myfile == NULL: raiseError("file: <%s> not found!"%(fname))

	# Read first 80 bytes and assess if any endline is found
	if not fread(buff,sizeof(char),80, myfile) == 80:
		raiseError("Error reading <%s>!"%(fname))

	fclose(myfile) # Close da file!!!

	# If a endline is not found, file is binary
	return False if '\n'.encode('utf-8') in buff else True

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def str_to_bin(string):
	return ('%-80s'%(string)).encode('utf-8')

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def bin_to_str(binary):
	return binary[:-1].decode('utf-8').strip()

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def int_to_bin(integer,b=4):
	return int(integer).to_bytes(b,'little')

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def bin_to_int(integer):
	return int.from_bytes(integer,'little')


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
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_readGeo(object fname):
	'''
	Read an Ensight Gold Geometry file in either
	ASCII or binary format.
	'''
	return Ensight_readGeoBIN(fname) if isBinary(fname) else Ensight_readGeoASCII(fname)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_readGeoBIN(object fname):
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
	cdef char buff[80]
	cdef unsigned int  inod, nnod, iel, nel, nelt, part
	cdef FILE *myfile
	cdef dict header = {'descr':'','nodeID':'','elemID':'','partID':0,'partNM':'','eltype':''}

	cdef float *x
	cdef float *y
	cdef float *z

	cdef np.ndarray[np.double_t,ndim=2] xyz
	cdef np.ndarray[np.int32_t,ndim=2]   conec

	# Open file for reading
	myfile = fopen(fname.encode('utf-8'),"rb")
	if myfile == NULL: raiseError("file: <%s> not found!"%(fname))
	
	# Read desctiption 1 to 3
	if not fread(buff,sizeof(char),80,myfile) == 80: # C Binary
		raiseError("Error reading <%s>!"%(fname))
	if not fread(buff,sizeof(char),80,myfile) == 80: # description line 1
		raiseError("Error reading <%s>!"%(fname))
	header['descr'] = '%s\n' % buff[:79].decode('utf-8').strip()
	if not fread(buff,sizeof(char),80,myfile) == 80: # description line 2
		raiseError("Error reading <%s>!"%(fname))
	header['descr'] = '%s\n%s' % (header['descr'],buff[:79].decode('utf-8').strip())
	
	# Read node id and element id
	if not fread(buff,sizeof(char),80,myfile) == 80: # node id <off/given/assign/ignore>
		raiseError("Error reading <%s>!"%(fname))
	header['nodeID'] = buff[:79].decode('utf-8').strip().replace('node id ','')
	if not fread(buff,sizeof(char),80,myfile) == 80: # element id <off/given/assign/ignore>
		raiseError("Error reading <%s>!"%(fname))
	header['elemID'] = buff[:79].decode('utf-8').strip().replace('element id ','')

	# Read part
	if not fread(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error reading <%s>!"%(fname))
	if not fread(&part,sizeof(int),1,myfile) == 1: 
		raiseError("Error reading <%s>!"%(fname))
	header['partID'] = part # Part ID

	# Read mesh part description
	if not fread(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error reading <%s>!"%(fname))
	header['partNM'] = buff[:79].decode('utf-8').strip()

	# Read number of coordinates
	if not fread(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error reading <%s>!"%(fname))
	if not fread(&nnod,sizeof(int),1,myfile) == 1:
		raiseError("Error reading <%s>!"%(fname))

	# Read coordinates
	x = <float*>malloc(nnod*sizeof(float))
	y = <float*>malloc(nnod*sizeof(float))
	z = <float*>malloc(nnod*sizeof(float))
	if not fread(x,sizeof(float),nnod,myfile) == nnod: raiseError("Error reading <%s>!"%(fname))
	if not fread(y,sizeof(float),nnod,myfile) == nnod: raiseError("Error reading <%s>!"%(fname))
	if not fread(z,sizeof(float),nnod,myfile) == nnod: raiseError("Error reading <%s>!"%(fname))
	
	xyz = np.ndarray((nnod,3),dtype=np.double)
	for inod in range(nnod):
		xyz[inod,0] = <double>x[inod]
		xyz[inod,1] = <double>y[inod]
		xyz[inod,2] = <double>z[inod]

	free(x)
	free(y)
	free(z)

	# Read block (element type)
	if not fread(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error reading <%s>!"%(fname))
	header['eltype'] = buff[:79].decode('utf-8').strip()
	nelt = elnod(header['eltype'])

	if not fread(&nel,sizeof(int),1,myfile) == 1:
		raiseError("Error reading <%s>!"%(fname))
	
	# Allocate space for mesh connectivity
	conec = np.ndarray((nel,nelt),dtype=np.int32)

	# Read all mesh elements
	for iel in range(nel):
		if not fread(&conec[iel,0],sizeof(int),nelt,myfile) == nelt:
			raiseError("Error reading <%s>!"%(fname))

	# Close file
	fclose(myfile)

	# Return
	return xyz, conec, header

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_readGeoASCII(object fname):
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
	cdef char buff[1024]
	cdef unsigned int  ii, jj, ic, nnod, nel, nelt
	cdef FILE *myfile
	cdef int  *order
	cdef dict header = {'descr':'','nodeID':'','elemID':'','partID':0,'partNM':'','eltype':''}

	cdef np.ndarray[np.double_t,ndim=2] xyz
	cdef np.ndarray[np.int32_t,ndim=2]   conec
	
	# Open file for reading
	myfile = fopen(fname.encode('utf-8'),"r")
	if myfile == NULL: raiseError("file: <%s> not found!"%(fname))

	# Read description lines
	reads(buff,sizeof(buff),myfile)
	header['descr'] = '%s' % buff.decode('utf-8').strip()
	reads(buff,sizeof(buff),myfile)
	header['descr'] = '%s\n%s' % (header['descr'],buff.decode('utf-8').strip())

	# Select node id
	reads(buff,sizeof(buff),myfile)
	header['nodeID'] = buff.decode('utf-8').strip().replace('node id ','')

	# Element id given(1)/assign(2)/ignore(3)
	reads(buff,sizeof(buff),myfile)
	header['elemID'] = buff.decode('utf-8').strip().replace('element id ','')

	# Read part
	reads(buff,sizeof(buff),myfile)
	reads(buff,sizeof(buff),myfile)
	header['partID'] = atoi(buff)

	# Read mesh part description
	reads(buff,sizeof(buff),myfile)
	header['partNM'] = buff.decode('utf-8').strip()
	
	# Read number of coordinates
	reads(buff,sizeof(buff),myfile);
	reads(buff,sizeof(buff),myfile);
	nnod = atoi(buff);

	# Node positions
	xyz = np.ndarray((nnod,3),dtype=np.double)
	if 'given' in header['nodeID']:
		# Read the node order
		order = <int*>malloc(nnod*sizeof(int))
		for ii in range(nnod):	
			reads(buff,sizeof(buff),myfile)
			order[ii] = atoi(buff)-1
		# Read the coordinates
		for ii in range(nnod): # x coordinate
			reads(buff,sizeof(buff),myfile)
			xyz[order[ii],0] = <double>atof(buff)
		for ii in range(nnod): # y coordinate
			reads(buff,sizeof(buff),myfile)
			xyz[order[ii],1] = <double>atof(buff)
		for ii in range(nnod): # z coordinate
			reads(buff,sizeof(buff),myfile)
			xyz[order[ii],2] = <double>atof(buff)
		# Free order
		free(order)
	else:
		# Read the coordinates
		for ii in range(nnod): # x coordinate
			reads(buff,sizeof(buff),myfile)
			xyz[ii,0] = <double>atof(buff)
		for ii in range(nnod): # y coordinate
			reads(buff,sizeof(buff),myfile)
			xyz[ii,1] = <double>atof(buff)
		for ii in range(nnod): # z coordinate
			reads(buff,sizeof(buff),myfile)
			xyz[ii,2] = <double>atof(buff)

	# Read block (element type)
	reads(buff,sizeof(buff),myfile)
	header['eltype'] = buff.decode('utf-8').strip()
	nelt = elnod(header['eltype'])		

	reads(buff,sizeof(buff),myfile)
	nel = atoi(buff)

	# Connectivity
	conec = np.ndarray((nel,nelt),dtype=np.int32)
	if 'given' in header['elemID']:
		# Read the element order
		order = <int*>malloc(nel*sizeof(int))
		for ii in range(nel):	
			reads(buff,sizeof(buff),myfile)
			order[ii] = atoi(buff)-1
		# Read connectivity
		for ii in range(nel):	
			ic = order[ii]
			reads(buff,sizeof(buff),myfile)
			conec[ic,0] = atoi( strtok(buff," ") )
			for jj in range(1,nelt):
				conec[ic,jj] = atoi( strtok(NULL," ") )
		# Free order
		free(order)
	else:
		for ii in range(nel):	
			reads(buff,sizeof(buff),myfile);
			conec[ii,0] = atoi( strtok(buff," ") )
			for jj in range(1,nelt):
				conec[ii,jj] = atoi( strtok(NULL," ") )
	
	# Close file
	fclose(myfile)

	# Return
	return xyz, conec, header

@cr('EnsightIO.readGeo')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_writeGeo(object fname, double[:,:] xyz, int[:,:] conec, dict header):
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
	cdef char buff[80]
	cdef unsigned int  inod, nnod = xyz.shape[0], iel, nel = conec.shape[0], nelt = conec.shape[1]
	cdef int part
	cdef FILE *myfile

	cdef float *x
	cdef float *y
	cdef float *z

	# Open file for writing
	myfile = fopen(fname.encode('utf-8'),"wb")
	if myfile == NULL: raiseError("file: <%s> not found!"%(fname))

	# Write description 1 to 3
	buff = ("%-80s"%"C Binary").encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))
	buff = ("%-80s"%header['descr'].split('\n')[0]).encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))
	buff = ("%-80s"%header['descr'].split('\n')[1]).encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))

	# Write node and element - assume always will be assigned!
	buff = ("%-80s"%("node id %s"%header['nodeID'])).encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))	
	buff = ("%-80s"%("element id %s"%header['elemID'])).encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))
		
	# Write part
	buff = ("%-80s"%"part").encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))
	part = header['partID']
	if not fwrite(&part,sizeof(int),1,myfile) == 1: 
		raiseError("Error writing <%s>!"%(fname))	
	
	# Write mesh part description
	buff = ("%-80s"%header['partNM']).encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))

	# Write number of coordinates
	buff = ("%-80s"%"coordinates").encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))
	if not fwrite(&nnod,sizeof(int),1,myfile) == 1: 
		raiseError("Error writing <%s>!"%(fname))

	# Recover and write mesh coordinates
	x = <float*>malloc(nnod*sizeof(float))
	y = <float*>malloc(nnod*sizeof(float))
	z = <float*>malloc(nnod*sizeof(float))

	for inod in range(nnod):
		x[inod] = <float>xyz[inod,0]
		y[inod] = <float>xyz[inod,1]
		z[inod] = <float>xyz[inod,2]

	if not fwrite(x,sizeof(int),nnod,myfile) == nnod: raiseError("Error writing <%s>!"%(fname))
	if not fwrite(y,sizeof(int),nnod,myfile) == nnod: raiseError("Error writing <%s>!"%(fname))
	if not fwrite(z,sizeof(int),nnod,myfile) == nnod: raiseError("Error writing <%s>!"%(fname))

	free(x)
	free(y)
	free(z)

	# Write block (element type)
	buff = ("%-80s"%header['eltype']).encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))
  
	# Write number of elements
	if not fwrite(&nel,sizeof(int),1,myfile) == 1: 
		raiseError("Error writing <%s>!"%(fname))

	# Write element connectivity
	for iel in range(nel):
		if not fwrite(&conec[iel,0],sizeof(int),nelt,myfile) == nelt: 
			raiseError("Error writing <%s>!"%(fname))
		
	# Close file
	fclose(myfile);

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

@cr('EnsightIO.readField')
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_readField(object fname, int dims=1, int nnod=-1, int parallel=False):
	'''
	Read an Ensight Gold field file in either
	ASCII or binary format.
	'''
	cdef object
	if parallel and isBinary(fname):
		return Ensight_readFieldMPIO(fname,dims,nnod)
	else:
		return Ensight_readFieldBIN(fname,dims,nnod) if isBinary(fname) else Ensight_readFieldASCII(fname,dims,nnod)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_readFieldBIN(object fname, int dims=1, int nnod=-1):
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
	cdef char  buff[80]
	cdef int   ii, jj, part, itemsRead, count, size
	cdef FILE *myfile
	cdef dict  header = {'descr':'','partID':0}

	cdef float     *data
	cdef np.ndarray field

	# Open file for reading
	myfile = fopen(fname.encode('utf-8'),"rb")
	if myfile == NULL: raiseError("file: <%s> not found!"%(fname))

	# Read description 1
	if not fread(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error reading <%s>!"%(fname))
	header['descr'] = buff[:79].decode('utf-8').strip()

	# Read part
	if not fread(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error reading <%s>!"%(fname))
	if not fread(&part,sizeof(int),1,myfile) == 1:
		raiseError("Error reading <%s>!"%(fname))
	header['partID'] = part # Part ID

	# Read coordinates
	if not fread(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error reading <%s>!"%(fname))
			
	# Read field data
	if nnod < 0: # Mesh is not known, read by chunks
		count = 0
		size  = 0
		data  = <float*>malloc(1024*sizeof(float))
		# Read data in buffer
		itemsRead = fread(&data[count*1024],sizeof(float),1024,myfile)
		size += itemsRead
		while itemsRead > 0:
			# Increase the allocation space
			realloc(data,(count+1)*1024*sizeof(float))
			# Read data in buffer
			itemsRead = fread(&data[count*1024],sizeof(float),1024,myfile);
			size  += itemsRead
			count += 1
		# Get nodes
		nnod = size//dims
	else:
		data = <float*>malloc(dims*nnod*sizeof(float))
		if not fread(data,sizeof(float),dims*nnod,myfile) == <unsigned int>dims*nnod:
			raiseError("Error reading <%s>!"%(fname))

	# Store data to C array
	field = np.ndarray((nnod,),dtype=np.double) if dims==1 else np.ndarray((nnod,dims),dtype=np.double)
	if dims > 1:
		for ii in range(nnod):
			for jj in range(dims):
				field[ii,jj] = <double>data[ii+nnod*jj]
	else:
		for ii in range(nnod):
			field[ii] = <double>data[ii]
	free(data)
		
	# Close file
	fclose(myfile)

	# Return
	return field, header

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_readFieldASCII(object fname,int dims=1,int nnod=-1):
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
	cdef char buff[1024]
	cdef int ii, jj, count, nalloc
	cdef FILE *myfile
	cdef dict header = {'descr':'','partID':0}
	
	cdef float     *data
	cdef np.ndarray field

	# Open file for reading
	myfile = fopen(fname.encode('utf-8'),"r")
	if myfile == NULL: raiseError("file: <%s> not found!"%(fname))

	# Read description lines
	reads(buff,sizeof(buff),myfile)
	header['descr'] = buff.decode('utf-8').strip()
	
	# Read part
	reads(buff,sizeof(buff),myfile)
	reads(buff,sizeof(buff),myfile)
	header['partID'] = atoi(buff)

	# Read coordinates
	reads(buff,sizeof(buff),myfile)

	# Read field data
	count = 0
	if nnod < 0: # Mesh is not known, read by chunks
		nalloc = 1
		data  = <float*>malloc(nalloc*1024*sizeof(float))
		reads(buff,sizeof(buff),myfile);
		while not feof(myfile):
			# Copy buffered data to out
			data[count] = atof(buff)
			# Increase the count
			count += 1
			# Reallocate
			if count == nalloc*1024:
				nalloc += 1
				realloc(data,nalloc*1024*sizeof(float))
			# Read next value
			reads(buff,sizeof(buff),myfile);
		# Get nodes
		nnod = count//dims
	else:
		data = <float*>malloc(dims*nnod*sizeof(float))
		reads(buff,sizeof(buff),myfile)

		while not feof(myfile):
			# Copy buffered data to out
			data[count] = atof(buff)
			# Increase the count
			count += 1
			# Read next value
			reads(buff,sizeof(buff),myfile)

	# Store data to C array
	field = np.ndarray((nnod,),dtype=np.double) if dims==1 else np.ndarray((nnod,dims),dtype=np.double)
	if dims > 1:
		for ii in range(nnod):
			for jj in range(dims):
				field[ii,jj] = <double>data[ii+nnod*jj]
	else:
		for ii in range(nnod):
			field[ii] = <double>data[ii]
	free(data)
		
	# Close file
	fclose(myfile)

	# Return
	return field, header

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_readFieldMPIO(object fname, int dims=1, int nnod=-1):
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
	cdef object f, shp
	cdef int idim, istart, iend, header_sz = 80*3+4  # 3 80 bytes char + 4 byte integer
	cdef dict header   = {'descr':'','partID':0}
	cdef bytes header_bin
	cdef np.ndarray header_np, field, aux
	# Open file for reading
	f = mpi_file_open(MPI_COMM,fname,MPI_RDONLY)
	# Read Ensight header
	header_np = np.ndarray((header_sz,),dtype='b')
	f.Read_at_all(0,header_np)
	header_bin = bytes(header_np.view('S%d'%header_sz))
	# Parse the header
	header = {}
	header['descr']  = bin_to_str(header_bin[:80])         # Description
	header['partID'] = bin_to_int(header_bin[2*80:2*80+4]) # Part ID
#	header['partNM'] = bin_to_str(header_bin[2*80+4:3*80]) # Part name 
	# Use a simple worksplit to see where everyone reads
	istart,iend = worksplit(0,nnod,MPI_RANK,nWorkers=MPI_SIZE)
	# Create flattened output array
	shp = (iend-istart)
	if dims > 1: shp = ((iend-istart),dims)
	field = np.ndarray(shp,np.double)
	# Read the field
	if dims > 1:
		for idim in range(dims):
			aux = np.ndarray((iend-istart,),np.float32)
			f.Read_at(header_sz+(istart+idim*nnod)*4,aux)
			field[:,idim] = np.ascontiguousarray(aux)
	else:
		f.Read_at(header_sz+istart*4,field)
		field = np.ascontiguousarray(field)		
	# Close the field
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
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_writeField(object fname,np.ndarray field,dict header,int parallel=False):
	'''
	Write an Ensight Gold field file in binary format.
	'''
	return Ensight_writeFieldBIN(fname,field,header) if not parallel else Ensight_writeFieldMPIO(fname,field,header)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_writeFieldBIN(object fname,np.ndarray field,dict header):
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
	cdef char buff[80]
	cdef int ii, jj, nnod = field.shape[0], dims, part
	cdef FILE *myfile

	cdef float *data

	dims = 1 if len((<object> field).shape) == 1 else field.shape[1]

	# Open file for writing
	myfile = fopen(fname.encode('utf-8'),"wb")
	if myfile == NULL: raiseError("file: <%s> not found!"%(fname))

	# Write description
	buff = ("%-80s"%header['descr']).encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))
		
	# Write part
	buff = ("%-80s"%"part").encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))	
	part = header['partID']
	if not fwrite(&part,sizeof(int),1,myfile) == 1: 
		raiseError("Error writing <%s>!"%(fname))	
	
	# Write coordinates
	buff = ("%-80s"%"coordinates").encode('utf-8')
	if not fwrite(buff,sizeof(char),80,myfile) == 80: 
		raiseError("Error writing <%s>!"%(fname))		

	# Write field
	data = <float*>malloc(dims*nnod*sizeof(float))
	if dims > 1:
		for ii in range(nnod):
			for jj in range(dims):
				data[ii+nnod*jj] = <float>field[ii,jj]
	else:
		for ii in range(nnod):
			data[ii] = <float>field[ii]

	if not fwrite(data,sizeof(float),dims*nnod,myfile) == <unsigned int>dims*nnod: 
		raiseError("Error writing <%s>!"%(fname))

	free(data)

	fclose(myfile)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)    # turn off zero division check
def Ensight_writeFieldMPIO(object fname,np.ndarray field,dict header):
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
	cdef object f, header_bin
	cdef int istart, iend, icol, nrows, ncols, nrowsT, header_sz = 80*3+4  # 3 80 bytes char + 4 byte integer
	# Open file for writing
	f = mpi_file_open(MPI_COMM,fname,MPI_WRONLY|MPI_CREATE)
	# Write Ensight header
	header_bin  = str_to_bin(header['descr'])
	header_bin += str_to_bin('part')
	header_bin += int_to_bin(header['partID'])
	header_bin += str_to_bin('coordinates')
	f.Write_at_all(0,np.frombuffer(header_bin,np.int8))
	# Obtain the total number of nodes
	nrows  = field.shape[0]
	ncols  = 1 if len((<object>field).shape) == 1 else field.shape[1]
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