#!/bin/bash
#
# Run SPOD testsuite
cd Testsuite
rm -rf *.tar.gz
python tsuite_SPOD_cylinder.py
tar czf cylinderSPOD_serial.tar.gz cylinderSPOD/
rm -rf cylinderSPOD
mpirun -np 4 python tsuite_SPOD_cylinder.py
tar czf cylinderSPOD_parallel.tar.gz cylinderSPOD/
rm -rf cylinderSPOD
python tsuite_SPOD_jet.py
tar czf jetSPOD_serial.tar.gz jetSPOD/
rm -rf jetSPOD
mpirun -np 4 python tsuite_SPOD_jet.py
tar czf jetSPOD_parallel.tar.gz jetSPOD/
rm -rf jetSPOD
python tsuite_SPOD_channel.py
tar czf channelSPOD_serial.tar.gz channelSPOD/
rm -rf channelSPOD
mpirun -np 4 python tsuite_SPOD_channel.py
tar czf channelSPOD_parallel.tar.gz channelSPOD/
rm -rf channelSPOD
cd -