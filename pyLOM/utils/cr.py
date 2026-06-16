#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Utils - Chrono modules for performance.
#
# Last rev: 09/07/2021
from __future__ import print_function, division

import numpy as np, copy, functools

from .mpi    import MPI, MPI_RANK, MPI_SIZE, mpi_reduce
from .errors import raiseError, raiseWarning


'''
Dictionary to store all created channels using its name as key.
'''
CHANNEL_DICT = {}


class channel(object):
	r'''
	This is a channel for the cr counter.

	Attributes:
		_name: Name of the channel.
		_tmax: Maximum time of the channel.
		_tmin: Minimum time of the channel.
		_tsum: Total time of the channel.
		_nop: Number of operations.
		_tini: Initial instant (if == 0 channel is not being taken into
				    account).
	'''
	def __init__(self, name, tmax, tmin, tsum, nop, tini):
		r'''
		Initialise a channel instance. 

		Args:
			name: Name of the channel.
			tmax: Maximum time of the channel.
			tmin: Minimum time of the channel.
			tsum: Total time of the channel.
			nop: Number of operations.
			tini: Initial instant (if == 0 channel is not being
				            taken into account).
		'''
		self._name = name # Name of the channel
		self._tmax = tmax # Maximum time of the channel
		self._tmin = tmin # Minimum time of the channel
		self._tsum = tsum # Total time of the channel
		self._nop  = nop  # Number of operations
		self._tini = tini # Initial instant (if == 0 channel is not being taken into account)

	def __str__(self):
		'''
		Returns:
			str:
				Contains all the attributes.
		'''
		return 'name %-30s n %9d tmin %e tmax %e tavg %e tsum %e' % (self.name,self.nop,self.tmin,self.tmax,self.tavg,self.tsum)

	def __add__(self, other):
		'''
		Addition.

		Args:
			self: First summand.
			other: Second summand.

		Returns:
			channel:
				The total time and number of operation is the
				                sum of both arguments.
		'''
		new = copy.deepcopy(self)
		new._tmax  = max(new._tmax,other._tmax)
		new._tmin  = min(new._tmin,other._tmin)
		new._tsum += other._tsum
		new._nop  += other._nop 
		return new

	def __iadd__(self, other):
		'''
		Addition assignment.

		Args:
			self: First summand.
			other: Second summand.

		Rteturns:
			channel:
				The total time and number of operation is the
				                sum of both arguments.
		'''
		self._tmax  = max(self._tmax,other._tmax)
		self._tmin  = min(self._tmin,other._tmin)
		self._tsum += other._tsum
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
		self._tmax = 0.0
		self._tmin = 0.0
		self._tsum = 0.0
		self._nop  = 0.0
		self._tini = 0.0

	def restart(self):
		'''
		Restart the initial instant.

		Args self:

		Returns:
			None.
		'''
		self._tini = 0.0

	def start(self,tini):
		'''
		Start the channel.

		Args:
			self:
			tini: Initial instant.

		Returns:
			None.
		'''
		self._tini = tini

	def increase_nop(self):
		'''
		Increment the number of operations by one.

		Args:
			self:

		Returns:
			None.
		'''
		self._nop += 1

	def increase_time(self,time):
		'''
		Increment the total time of the channel.

		Args:
			self:
			time: The amount of time to increase.

		Returns:
			None
		'''
		self._tsum += time

	def set_max(self,time):
		'''
		Sets the maximum time of the channel.

		Args:
			self:
			time: The maximum time of the channel.

		Returns:
			None.
		'''
		if time > self._tmax or self._nop == 1: self._tmax = time

	def set_min(self,time):
		'''
		Sets the minimum time of the channel.

		Args:
			self:
			time: The minimum time of the channel.

		Returns:
			None.
		'''
		if time < self._tmin or self._nop == 1: self._tmin = time

	def elapsed(self,time):
		'''
		Elapsed time since the initial time.

		Args:
			self:
			time: Time to compare to the initial time.

		Returns:
			The elapsed time since selt._tini.
			
		'''
		return time - self._tini

	def is_running(self):
		'''
		Checks if the channel is running.

		Args:
			self:

		Returns:
			bool:
				If the channel is running.
		'''
		return not self._tini == 0

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
	def tmin(self):
		return self._tmin
	@property
	def tmax(self):
		return self._tmax
	@property
	def tavg(self):
		return self._tsum/(1.* self._nop)
	@property
	def tsum(self):
		return self._tsum


def _newch(ch_name):
	'''
	Add a new channel to the list.

	Args:
		ch_name: Name of the new channel to create.

	Returns:
		channel:
			The newly created channel.
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
	Look for the channel and crash if it does not exist.

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
	Find the channel and if not found create it.

	Args:
		ch_name: The name of the channel to look at.

	Returns:
		channel:
			The channel with the given name.
	'''
	return CHANNEL_DICT[ch_name] if ch_name in CHANNEL_DICT.keys() else _newch(ch_name)

def _gettime():
	'''
	Returns the number of second since an arbitrary instant but fixed.
	Returned value will always be > 0.
	'''
	return MPI.Wtime()

def _reduce_cr(cr1,cr2,dtype):
	'''
	Reduces two channels dictionaries by adding the dictionaries with the same
	key.

	Args:
		cr1: Fisrt dictionary.
		cr2: Second dictionary.
		dtype:

	Returns:
		The reduced dictionaries.
	'''
	for key in cr2.keys():
		if key in cr1.keys():
			# Key exists in cr1, then simply accumulate
			cr1[key] += cr2[key]
		else:
			# Key does not exist in cr1, create it new
			cr1[key] = cr2[key]
	return cr1
cr_reduce = MPI.Op.Create(_reduce_cr, commute=True)

def _info_serial():
	'''
	Prints name and tsom of all channels in the dictionary. Serial version.
	'''
	tsum_array = np.array([CHANNEL_DICT[key].tsum for key in CHANNEL_DICT.keys()])
	name_array = np.array([CHANNEL_DICT[key].name for key in CHANNEL_DICT.keys()])

	ind = np.argsort(tsum_array) # sorted indices

	print('\ncr_info:',flush=True)
	for ii in ind[::-1]:
		print(CHANNEL_DICT[name_array[ii]],flush=True)
	print('',flush=True)

def _info_parallel():
	'''
	Prints name and tsom of all channels in the dictionary. Parallel version.
	'''
	CHANNEL_DICT_G = mpi_reduce(CHANNEL_DICT,root=0,op=cr_reduce,all=False)

	if MPI_RANK == 0:
		tsum_array = np.array([CHANNEL_DICT_G[key].tsum for key in CHANNEL_DICT_G.keys()])
		name_array = np.array([CHANNEL_DICT_G[key].name for key in CHANNEL_DICT_G.keys()])	
	
		ind = np.argsort(tsum_array) # sorted indices

		print('\ncr_info (mpi size: %d):' % (MPI_SIZE),flush=True)
		for ii in ind[::-1]:
			print(CHANNEL_DICT_G[name_array[ii]],flush=True)
		print('',flush=True)


def cr_reset():
	'''
	Delete all channels and start again
	'''
	CHANNEL_DICT = {}

def cr_info(rank=-1):
	'''
	Print information - order by major sum.

		Args:
			rank (int, optional): The rank which has to print the info.
				If negative (as default) the information of all ranks is
				printed.
	'''
	if rank >= 0 and rank == MPI_RANK:
		_info_serial()
	else:
		_info_parallel()

def cr_start(ch_name,suff):
	'''
	Start the chrono of a channel. Aborts if there is an already running channel 
	with the given name and suffix.

	Args:
		ch_name (str): The name of the channel to start.
		suff (int): The suffix of the channel to start.
	'''
	name_tmp = _addsuff(ch_name,suff)
	channel  = _findch_create(name_tmp)
	if channel.is_running():
		raiseError('Channel %s was already set!'%channel.name)
	channel.start( _gettime() )

def cr_stop(ch_name,suff):
	'''
	Stop the chrono of a channel. Aborts if there does not exist a channel
	with the given name and suffix.

	Args:
		ch_name (str): The name of the channel to stop.
		suff (int): The suffix of the channel to stop.
	'''
	end      = _gettime()
	name_tmp = _addsuff(ch_name,suff)
	channel  = _findch_crash(name_tmp)
	time     = channel.elapsed(end)

	channel.increase_nop()
	channel.set_max(time)
	channel.set_min(time)
	channel.increase_time(time)

	channel.restart()

def cr_time(ch_name,suff):
	'''
	Get the time of a channel that is running; channel keeps running. Aborts if
	there does not exist a channel with the given name and suffix.

	Args:
		ch_name (str): The name of the channel.
		suff (int): The suffix of the channel.

	Returns:
		The elapsed time of the channel sice the time of call.
	'''
	end = _gettime()
	name_tmp = _addsuff(ch_name,suff)
	channel  = _findch_crash(name_tmp)
	return channel.elapsed(end)

def cr(ch_name,suff=0):
	'''
	CR decorator.

	Args:
		ch_name (str): The name of the channel.
		suff (int): The suffix of the channel.

	Returns:
		The decorator function.
	'''
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			cr_start(ch_name,suff)
			out = func(*args,**kwargs)
			cr_stop(ch_name,suff)
			return out
		return wrapper
	return decorator

try:
	import nvtx

	def cr_nvtx(ch_name,suff=0,color="green"):
		'''
		CR NVTX decorator.

		Args:
			ch_name (str): The name of the channel.
			suff (int, optional): The suffix of the channel.
			color (str, optional): The color to anotate the message.

		Returns:
			The decorator function.
		'''
		def decorator(func):
			@functools.wraps(func)
			def wrapper(*args,**kwargs):
				cr_start(ch_name,suff)
				with nvtx.annotate(message=ch_name,color=color):
					out = func(*args,**kwargs)
				cr_stop(ch_name,suff)
				return out
			return wrapper
		return decorator

except:
	raiseWarning('Import - NVTX not present!',False)

	def cr_nvtx(ch_name,suff=0,color="green"):
		'''
		CR NVTX decorator.

		Args:
			ch_name (str): The name of the channel.
			suff (int, optional): The suffix of the channel.
			color (str, optional): The color to anotate the message.

		Returns:
			The decorator function.
		'''
		def decorator(func):
			@functools.wraps(func)
			def wrapper(*args,**kwargs):
				cr_start(ch_name,suff)
				out = func(*args,**kwargs)
				cr_stop(ch_name,suff)
				return out
			return wrapper
		return decorator
