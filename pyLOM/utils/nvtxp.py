#!/usr/bin/env python
#
# pyLOM, utils.
#
# NVTX profiler
#
# Last rev: 12/03/2025
from __future__ import print_function, division

import functools

try:
	import nvtx

	def nvtxp(ch_name,color="blue"):
		'''
		CR NVTX decorator
		'''
		def decorator(func):
			@functools.wraps(func)
			def wrapper(*args,**kwargs):
				with nvtx.annotate(message=ch_name,color=color):
					out = func(*args,**kwargs)
				return out
			return wrapper
		return decorator

except:
	def nvtxp(ch_name,color="blue"):
		'''
		CR NVTX decorator
		'''
		def decorator(func):
			@functools.wraps(func)
			def wrapper(*args,**kwargs):
				out = func(*args,**kwargs)
				return out
			return wrapper
		return decorator