#!/usr/bin/env python3
#
# PYLOM TESTSUITE
#
# Run pyLOM testsuite.
#
# 20/09/2024
from __future__ import print_function

import os, argparse, json

from system_utils import echoSinner, mysystem, run_python, procline, assumeOK
from dragon       import dragon, dragonAllOK, dragonAlmostOK, dragonAngry

## Parameters
RELTOL  = 1e-3
ZEROTOL = 1e-4
OUTROOT = '/tmp/TESTSUITE/' # WARNING: OUTROOT IS ERASED BEFORE EVERY RUN
MODULES = ['POD','DMD','SPOD','NN']


## Testsuite aux functions
def load(filename):
	'''
	Load test dictionary from file
	'''
	file     = open(filename,'r')
	testDict = json.load(file)
	file.close() 
	# Delete comments
	for c in ['comment','Comment','COMMENT','_comment']:
		if c in testDict.keys(): testDict.pop(c)
	return testDict

def checkOutput(summary,reference,relTol,zeroTol,info=False):
	print("  Checking files summary=<%s> reference=<%s>" % (summary,reference),flush=True)
	file1 = open(summary,'r')
	file2 = open(reference,'r')
	error = False
	while True:
		displ = False
		line1 = file1.readline().strip()
		if not line1: break
		line2 = file2.readline().strip()
		dat1  = procline(line1)
		dat2  = procline(line2)
		name1 = dat1[0]
		name2 = dat2[0]
		if name1 != name2:
			raise Exception('lines %s and %s differ in the field name ' % (line1,line2))
		for v1,v2 in zip(dat1[1:],dat2[1:]):
			if not assumeOK(v1,v2,relTol,zeroTol): 
				error = True
				displ = True
		if displ:
			print("field %s DIFFERS!"%name1)
			print ("reference: %s" % line2,flush=True)
			print ("output   : %s" % line1,flush=True)
			error = True
	return 1 if error else 0

def runtest(name,nprocs,file,datafile,var,params,relTol=RELTOL,zeroTol=ZEROTOL,resetRef=False,OUTROOT=OUTROOT,oversubscribe=False):
	'''
	Run a single testsuite item.
	'''
	# Run python test
	print('**',end='',flush=True)
	run_python(name,nprocs,file,datafile,var,params,OUTROOT=OUTROOT,oversubscribe=oversubscribe,grepStats=True)
	output    = os.path.join(OUTROOT,f"{name}.out")
	summary   = os.path.join(OUTROOT,f"{name}.log")
	reference = os.path.join('references',f"{name}.ref")

	if resetRef:
		print("  Saving new reference in file <%s>" % reference)
		r   = mysystem(f"grep 'TSUITE' {output} > {reference}",echoi=False,echoo=False)
		if r!=0: raise Exception('can''t set reference !')
		r = mysystem(f"cat {reference}",echoi=False,echoo=True)
		if r!=0: raise Exception('can''t cat reference !')
		return 0

	if checkOutput(summary,reference,relTol,zeroTol) == 0:
		print("  OK")
		return 0
	else:
		print('  check FAILED')
		return 1

def run(testDict,active_modules=MODULES,relTol=RELTOL,zeroTol=ZEROTOL,OUTROOT=OUTROOT,oversubscribe=False):
	'''
	Run the tests given a test dictionary.
	active_modules are the modules to which we want to pass the tests
	'''
	# Display information about username, computer, git commit and tolerances
	echoSinner()

	# Compute the total number of tests, the number of active tests
	# and the tests per module.
	nTOT, nACT, nMOD = 0, 0, {}
	for t in testDict.keys():
		test   = testDict[t]
		active = test['active'] and test['module'] in active_modules
		nTOT  += 1
		test['active'] = active
		nACT  += 1 if active else 0
		if not test['module'] in nMOD.keys(): nMOD[test['module']] = 0
		nMOD[test['module']] += 1 if active else 0

	# Clean tests root directory
	cmd = "rm -rf %s" % OUTROOT
	mysystem(cmd,echoi=True,echoo=True)
	os.makedirs(OUTROOT,exist_ok=True)

	# Show the dragon
	dragon()

	# Now run the tests
	nOK, nFAIL, nSKIP, nOKM = 0, 0, 0, {}
	whoSkipped, whoFailed = [], []
	for t in testDict.keys():
		test = testDict[t]
		# Skip inactive test
		if not test['active']:
			nSKIP += 1
			whoSkipped.append(t)
			print('Skipping test %s...'%test['name'],flush=True)
			continue
		# Run the test
		print('Running test %s...'%test['name'],flush=True)
		r = runtest(t,test['nprocs'],test['file'],test['data'],test['var'],test['params'],resetRef=test['reset'],OUTROOT=OUTROOT,oversubscribe=oversubscribe)
		# Check the output
		nOK   += r == 0
		nFAIL += r != 0
		if not test['module'] in nOKM.keys(): nOKM[test['module']] = 0
		nOKM[test['module']] += r == 0
		# Keep a list of the tests that failed
		if r != 0: whoFailed.append(t)

	# Summarize the outcome of the tests
	if nOK == nTOT and nSKIP == 0:
		# All the tests have been passed, the dragon is happy
		dragonAllOK()
		# Exit with a good exit value
		return 0
	
	if nOK == nACT and nSKIP > 0:
		# All the active tests have been passed, the dragon moves
		dragonAlmostOK()
		# Report which tests have been skipped
		for t in whoSkipped:
			test = testDict[t]
			print('test <%s>: %s SKIPPED!'%(t,test['name']),flush=True)
		# Exit with a good exit value - but with care
		return 0
	
	if nFAIL > 0:
		# We have failed tests, the dragon is MAD
		dragonAngry()
		# Report which tests have failed
		for t in whoFailed:
			test = testDict[t]
			print('test <%s>: %s FAILED!'%(t,test['name']),flush=True)
		# Exit with a bad exit value
		return 1


## Main program
if __name__ == "__main__":
	## Define input arguments
	argpar = argparse.ArgumentParser(prog="pyLOM_testsuite", description="Run pyLOM testsuite.")
	argpar.add_argument('-f','--file',type=str,help='JSON file containing the tests',dest='file')
	argpar.add_argument('-m','--modules',type=str,help='Modules in which to pass the tests separated by comma', dest='modules')
	argpar.add_argument('-t','--tests',type=str,help='Tests to run separated by comma (defaults to all)', dest='tests')
	argpar.add_argument('-R','--reset',action='store_true',help='Reset the reference', dest='reset')
	argpar.add_argument('--reltol',type=float,help='Relative tolerance (default: 1e-3)', dest='reltol')
	argpar.add_argument('--zerotol',type=float,help='Zero tolerance (default: 1e-4)', dest='zerotol')
	argpar.add_argument('--outroot',type=str,help='Output directory (default: /tmp/TEST/)', dest='outroot')
	argpar.add_argument('--oversubscribe',action='store_true',help='Oversubscribe MPI', dest='oversubscribe')

	# parse input arguments
	args = argpar.parse_args()
	if args.modules:     args.modules = [m for m in args.modules.split(',')]
	# set default values
	if not args.file:    args.file    = 'Testsuite/testsuite.json'
	if not args.reltol:  args.reltol  = RELTOL
	if not args.zerotol: args.zerotol = ZEROTOL
	if not args.outroot: args.outroot = '/tmp/TESTSUITE/' # WARNING: OUTROOT IS ERASED BEFORE EVERY RUN
	if not args.modules: args.modules = ['POD','DMD','SPOD','NN','MANIFOLD']


	## Load tests
	testDict = load(args.file)

	# Filter active tests
	if args.tests:
		active = [t for t in args.tests.split(',')]
		for t in testDict.keys():
			test = testDict[t]
			if not t in active: test['active'] = False

	# Force reset of tests
	if args.reset:
		for t in testDict.keys():
			testDict[t]['reset'] = True


	## Run testsuite
	r = run(testDict,active_modules=args.modules,relTol=args.reltol,zeroTol=args.zerotol,OUTROOT=args.outroot,oversubscribe=args.oversubscribe)
	exit(r) # exit with code