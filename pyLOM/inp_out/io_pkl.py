#!/usr/bin/env python
#
# pyLOM, IO
#
# PKL Input Output
#
# Last rev: 30/07/2021
from __future__ import print_function, division

import pickle as pkl

from ..utils.cr     import cr


@cr('pklIO.save')
def pkl_save(fname,obj):
	'''
	Save an object in pkl format
	'''
	f = open(fname,'wb')
	pkl.dump(obj,f)
	f.close()


@cr('pklIO.load')
def pkl_load(fname):
	'''
	Load an object in pkl format
	'''
	f = open(fname,'rb')
	out = pkl.load(f)
	f.close()
	return out
