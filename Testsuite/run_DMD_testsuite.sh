#!/bin/bash
#
# Run DMD testsuite
cd Testsuite
rm -rf *.tar.gz
python tsuite_DMD_cylinder.py
tar czf cylinderDMD_serial.tar.gz cylinderDMD/
rm -rf cylinderDMD
mpirun -np 4 python tsuite_DMD_cylinder.py
tar czf cylinderDMD_parallel.tar.gz cylinderDMD/
rm -rf cylinderDMD
python tsuite_DMD_jet.py
tar czf jetDMD_serial.tar.gz jetDMD/
rm -rf jetDMD
mpirun -np 4 python tsuite_DMD_jet.py
tar czf jetDMD_parallel.tar.gz jetDMD/
rm -rf jetDMD
cd -