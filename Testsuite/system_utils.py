#!/usr/bin/env python
#
# System utilities
#
# 20/09/2024
from __future__ import print_function

import os, sys, subprocess


def echoSinner():
	'''
	echoes user name, date, host name, host architecture, git commit
	'''
	print("Python version %s " % (sys.version),flush=True)
	res = mysystem("whoami",echoi=False,echoo=False,returnAll=True,crashIfFails=True)
	print("SINNER=%s" % res[1],flush=True)
	res = mysystem("date",echoi=False,echoo=False,returnAll=True,crashIfFails=True)
	print("DATE=%s" % res[1],flush=True)
	res = mysystem("hostname",echoi=False,echoo=False,returnAll=True,crashIfFails=True)
	print("HOSTNAME=%s" % res[1],flush=True)
	res = mysystem("uname -m",echoi=False,echoo=False,returnAll=True,crashIfFails=True)
	print("HOST ARCHITECTURE=%s" % res[1],flush=True)
	res = mysystem("git log -1|head -1",echoi=False,echoo=False,returnAll=True,crashIfFails=True)
	print("COMMIT=%s" % res[1],flush=True)


def mysystem(cmd,echoi=True,echoo=True,extraInfo=False,returnAll=False,crashIfFails=False):
	'''
	cmd:          string with the system command; use && to compose commads eg: "cd / && pwd " and not "cd ; pwd"
				  otherwise, if there is an error in the first command but not the second, a zero will be returned
	echoi:        if true, display the command before running
	echoo:        if true, display the command result
	extrainfo:    if true, print more information
	returnAll:    if false, return only the integer return code
				  if true, return a structure with
					 res.returncode : code
					 res.value: return string
	crashIfFails: if true, raises an exception if system call returns something not 0	
	'''
	if echoi:
		print("Running <%s>" % cmd,flush=True)
	ret = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE)
	if extraInfo:
		print("returns %d" % ret.returncode)
	if crashIfFails and ret.returncode!=0:
		raise Exception('System call <%s> returned %d !' % (cmd,ret.returncode))

	outd=ret.stdout

	if echoo:
		outs=outd.decode("utf-8").strip()
		if len(outs)>0:
			print(outs,flush=True)

	return (ret.returncode, outd.decode("utf-8").strip()) if returnAll else ret.returncode


def testExe(name):
	if mysystem("which %s"%name,echoi=False,echoo=False)!=0:
		print("%s not found, compile and check PATH" % name,flush=True)
		return 1
	return 0


def run_python(name,nprocs,file,datafile,var,params,OUTROOT='/tmp',grepStats=False,oversubscribe=True):
	'''
	t: test file
	NPX NPY NPZ: number of processors
	outmode (ie 0 or 2, usually)
	grepStats: time step to capture the statistics and save to the output folder, use -1 to disable
	'''
	outdir  = os.path.join(OUTROOT,f"{name}")
	output  = os.path.join(OUTROOT,f"{name}.out")
	summary = os.path.join(OUTROOT,f"{name}.log")
	
	cmd = 'mpirun -np %d %s python %s "%s" "%s" "%s" "%s" | tee %s' % (nprocs,'--oversubscribe' if oversubscribe else '',file,datafile,var,outdir,params,output)
	r   = mysystem(cmd,echoi=True,echoo=False)
	if r!=0:
		print("Python returns non zero value %d !!!! " % r )
		raise Exception("Run failed, check output files !")

	cmd = "tail -1 %s | grep 'End of output'" % (output);
	r   = mysystem(cmd,False,False)
	if r!=0:
		print("Run ended unexpectedly, last line should be 'End of output' !!!")
		raise Exception("Run ended unexpectedly, check output files !")

	if grepStats:
		cmd = "grep 'TSUITE' %s > %s" % (output,summary)
		r   = mysystem(cmd,echoi=False,echoo=False)
		if r!=0: raise Exception('grep summary failed !')

	return 0


def procline(line):
	l1  = line.split('=')
	dat = [l1[0].split()[-1]]
	for x in l1[1].split():
		dat.append(eval(x))
	return dat


def assumeOK(n1,n2,relTol=1e-4,zeroTol=1e-15):
	if (abs(n1) < zeroTol): n1 = 0.
	if (abs(n2) < zeroTol): n2 = 0.
	if (abs(n1) < zeroTol) and (abs(n2) < zeroTol): return True
	m = min(abs(n1),abs(n2)) + 1e-15
	d = abs(n1-n2)
	return True if d/m < relTol else False