#!/usr/bin/env python
#
# pyAlya, mem.
#
# Memory module for performance profiling.
#
# Last rev: 11/04/2023
from __future__ import print_function, division

import sys, numpy as np, copy, functools, subprocess

from .mpi    import MPI, MPI_RANK, MPI_SIZE, mpi_reduce
from .errors import raiseError


PLATFORM = sys.platform

'''
Dictionary to store all created channels using its name as key.
'''
CHANNEL_DICT = {}

'''
Conversion factors to kB.
'''
CONVERSION = {
	'kB' : 1.,    # Output is in kB
	'mB' : 0.001,
	'gB' : 1.e-6,
}


class channel(object):
	r'''
	This is a channel for the cr counter.

	Attributes:
		_name: Name of the channel.
		_mmax: Maximum memory of the channel.
		_mmin: Minimum memory of the channel.
		_msum: Total memory of the channel.
		_nop: Number of operations.
		_mini: Initial instant (if == 0 channel is not being taken into account).
	'''
	def __init__(self, name, mmax, mmin, msum, nop, mini):
		r'''
		Initialise a channel instance. 

		Args:
			name: Name of the channel.
			mmax: Maximum of the channel.
			mmin: Minimum of the channel.
			tsum: Total memory of the channel.
			nop: Number of operations.
			mini: Initial instant (if == 0 channel is not being taken into account).
		'''
		self._name = name # Name of the channel
		self._mmax = mmax # Maximum of the channel
		self._mmin = mmin # Minimum of the channel
		self._msum = msum # Total of the channel
		self._nop  = nop  # Number of operations
		self._mini = mini # Initial instant (if == 0 channel is not being take into account)

	def __str__(self):
		'''
		Returns:
			str:
				Contains all the attributes.
		'''
		return 'name %-20s n %9d min %e max %e avg %e sum %e' % (self.name,self.nop,self.mmin,self.mmax,self.mavg,self.msum)

	def __add__(self, other):
		'''
		Addition.

		Args:
			self: First summand.
			other: Second summand.

		Returns:
			channel:
				The total memory and number of operation is the sum of both arguments.
		'''
		new = copy.deepcopy(self)
		new._mmax  = max(new._mmax,other._mmax)
		new._mmin  = min(new._mmin,other._mmin)
		new._msum += other._msum
		new._nop  += other._nop 
		return new

	def __iadd__(self, other):
		'''
		Addition assignment.

		Args:
			self: First summand.
			other: Second summand.

		Returns:
			channel:
				The total memory and number of operation is the sum of both arguments.
		'''
		self._mmax  = max(self._mmax,other._mmax)
		self._mmin  = min(self._mmin,other._mmin)
		self._msum += other._msum
		self._nop  += other._nop 
		return self

	def reset(self):
		'''
		Reset the channel.

		Args:
			self:

		Returns:
			None.
		'''
		self._mmax = 0.0
		self._mmin = 0.0
		self._msum = 0.0
		self._nop  = 0.0
		self._mini = 0.0

	def restart(self):
		'''
		Restart the initial instant.

		Args self:

		Returns:
			None.
		'''
		self._mini = 0.0

	def start(self,mini):
		'''
		Start the channel.

		Args:
			self:
			mini: Initial instant.

		Returns:
			None.
		'''
		self._mini = mini

	def increase_nop(self):
		'''
		Increment the number of operations by one.

		Args:
			self:

		Returns:
			None.
		'''
		self._nop += 1

	def increase_value(self,value):
		'''
		Increment the total memory of the channel.

		Args:
			self:
			value: The amount of memory to increase.

		Returns:
			None
		'''
		self._msum += value

	def set_max(self,value):
		'''
		Sets the maximum memory of the channel.

		Args:
			self:
			memory: The maximum memory of the channel.

		Returns:
			None.
		'''
		if value > self._mmax or self._nop == 1: self._mmax = value

	def set_min(self,value):
		'''
		Sets the minimum memory of the channel.

		Args:
			self:
			value: The minimum memory of the channel.

		Returns:
			None.
		'''
		if value < self._mmin or self._nop == 1: self._mmin = value

	def elapsed(self,value):
		'''
		Elapsed time since the initial time.

		Negative values are discarded.

		Args:
			self:
			value: Value to compare to the initial time.

		Returns:
			The elapsed time since selt._tini.
			
		'''
		return max(value - self._mini,0.)

	def is_running(self):
		'''
		Checks if the channel is running.

		Args:
			self:

		Returns:
			bool:
				If the channel is running.
		'''
		return not self._mini == 0

	@classmethod
	def new(cls,name):
		'''
		Create a new channel.

		Args:
			cls:
			name: Name of the created instance.

		Returns:
			channel:
				Initialised with the given name.
		'''
		return cls(name,0,0,0,0,0)

	@property
	def name(self):
		return self._name
	@property
	def nop(self):
		return self._nop
	@property
	def mmin(self):
		return self._mmin
	@property
	def mmax(self):
		return self._mmax
	@property
	def mavg(self):
		return self._msum/(1.* self._nop)
	@property
	def msum(self):
		return self._msum


def _newch(ch_name):
	'''
	Look for the channel
	'''
	CHANNEL_DICT[ch_name] = channel.new(ch_name)
	return CHANNEL_DICT[ch_name]

def _findch(ch_name):
	'''
	Look for the channel.

	Args:
		ch_name: The name of the channel to look at.

	Returns:
		channel or None:
			The channel with the given name or None if not was found.
	'''
	return CHANNEL_DICT[ch_name] if ch_name in CHANNEL_DICT.keys() else None

def _addsuff(ch_name,suff=-1):
	'''
	Append a suffix to the channel name.

	Args:
		ch_name (str): The channel name.
		suff (int, optional): The suffix to append.

	Returns:
		str:
			ch_name with the appended suffix (only if it was positive).
	'''
	return ch_name if suff <= 0 else '%s%02d' % (ch_name,suff)

def _findch_crash(ch_name):
	'''
	Find the channel and if not found create it.

	Args:
		ch_name: The name of the channel to look at.

	Returns:
		channel:
			The channel with the given name.
	'''
	if not ch_name in CHANNEL_DICT.keys():
		raiseError('Channel %s does not exist!' % ch_name)
	return CHANNEL_DICT[ch_name]

def _findch_create(ch_name):
	'''
	Find the channel and if not found create it
	'''
	return CHANNEL_DICT[ch_name] if ch_name in CHANNEL_DICT.keys() else _newch(ch_name)

def _getvalue(units=''):
	'''
	Returns the used memory on an arbitrary instant but fixed.
	'''
	cmd = "cat /proc/meminfo | grep MemFree | cut -d ':' -f 2 | awk '{$1=$1};1' | cut -d ' ' -f 1"
	return int( subprocess.check_output(cmd,shell=True) ) if PLATFORM.lower() == 'linux' else 0

def _reduce_mem(m1,m2,dtype):
	'''
	Reduces two channels dictionaries by adding the dictionaries with the same key.

	Args:
		m1: Fisrt dictionary.
		m2: Second dictionary.
		dtype:

	Returns:
		The reduced dictionaries.
	'''
	for key in m2.keys():
		if key in m1.keys():
			# Key exists in m1, then simply accumulate
			m1[key] += m2[key]
		else:
			# Key does not exist in cr1, create it new
			m1[key] = m2[key]
	return m1

mem_reduce = MPI.Op.Create(_reduce_mem, commute=True)

def _print_units(c,units):
	'''
	Prints the channel attributes.

	Args:
		c (channel): Channel to print.
		units (str): Units in which to print the output. Should be 'kB', 'mB' or 'gB'.

	Returns:
		str:
			The channel information using the selected units.
	'''
	f = CONVERSION[units]
	return 'name %-30s n %9d min %e max %e avg %e sum %e' % (c.name,c.nop,f*c.mmin,f*c.mmax,f*c.mavg,f*c.msum)

def _info_serial(units):
	'''
	Prints name and tsom of all channels in the dictionary. Serial version.

	Args:
		units (str): Units in which to print the output. Should be 'kB', 'mB' or 'gB'.
	'''
	msum_array = np.array([CHANNEL_DICT[key].msum for key in CHANNEL_DICT.keys()])
	name_array = np.array([CHANNEL_DICT[key].name for key in CHANNEL_DICT.keys()])

	ind = np.argsort(msum_array) # sorted indices

	print('\nmem_info, units=%s:'%units,flush=True)
	for ii in ind[::-1]:
		print(_print_units(CHANNEL_DICT[name_array[ii]],units),flush=True)
	print('',flush=True)

def _info_parallel(units):
	'''
	Prints name and tsom of all channels in the dictionary. Parallel version.

	Args:
		units (str): Units in which to print the output. Should be 'kB', 'mB' or 'gB'.
	'''
	CHANNEL_DICT_G = mpi_reduce(CHANNEL_DICT,op=mem_reduce,root=0)

	if MPI_RANK == 0:
		msum_array = np.array([CHANNEL_DICT_G[key].msum for key in CHANNEL_DICT_G.keys()])
		name_array = np.array([CHANNEL_DICT_G[key].name for key in CHANNEL_DICT_G.keys()])	
	
		ind = np.argsort(msum_array) # sorted indices

		print('\nmem_info, units=%s (mpi size: %d):' % (units,MPI_SIZE),flush=True)
		for ii in ind[::-1]:
			print(_print_units(CHANNEL_DICT_G[name_array[ii]],units),flush=True)
		print('',flush=True)


def mem_reset():
	'''
	Delete all channels and start again
	'''
	CHANNEL_DICT = {}

def mem_info(rank=-1,units='kB'):
	'''
	Print information - order by major sum.

	Args:
		rank (int, optional): The rank which has to print the info.
			If negative (as default) the information of all ranks is printed.
		units (str, optional): Units in which to print the output.
			Should be 'kB' (default), 'mB' or 'gB'.
	'''
	if rank >= 0 and rank == MPI_RANK:
		_info_serial(units)
	else:
		_info_parallel(units)

def mem_start(ch_name,suff):
	'''
	Start the chrono of a channel.

	Args:
		ch_name (str): The name of the channel to start.
		suff (int): The suffix of the channel to start.
	'''
	name_tmp = _addsuff(ch_name,suff)
	channel  = _findch_create(name_tmp)
	if channel.is_running():
		raiseError('Channel %s was already set!'%channel.name)
	channel.start( _getvalue() )

def mem_stop(ch_name,suff):
	'''
	Stop the chrono of a channel.

	Args:
		ch_name (str): The name of the channel to stop.
		suff (int): The suffix of the channel to stop.
	'''
	end      = _getvalue()
	name_tmp = _addsuff(ch_name,suff)
	channel  = _findch_crash(name_tmp)
	value     = channel.elapsed(end)

	channel.increase_nop()
	channel.set_max(value)
	channel.set_min(value)
	channel.increase_value(value)

	channel.restart()

def mem_value(ch_name,suff):
	'''
	Get the value of a channel that is running; channel keeps running. Aborts if
	there does not exist a channel with the given name and suffix.

	Args:
		ch_name (str): The name of the channel.
		suff (int): The suffix of the channel.

	Returns:
		The elapsed time of the channel sice the time of call.
	'''
	end = _getvalue()
	name_tmp = _addsuff(ch_name,suff)
	channel  = _findch_crash(name_tmp)
	return channel.elapsed(end)

def mem(ch_name,suff=0):
	'''
	CR decorator.

	Args:
		ch_name (str): The name of the channel.
		suff (int, optional): The suffix of the channel.

	Returns:
		The decorator function.
	'''
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			mem_start(ch_name,suff)
			out = func(*args,**kwargs)
			mem_stop(ch_name,suff)
			return out
		return wrapper
	return decorator
