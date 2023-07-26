#!/bin/bash
#
# Run POD testsuite
cd Testsuite
rm -rf *.tar.gz
python tsuite_POD_cylinder.py
tar czf cylinderPOD_serial.tar.gz cylinderPOD/
rm -rf cylinderPOD
mpirun -np 4 python tsuite_POD_cylinder.py
tar czf cylinderPOD_parallel.tar.gz cylinderPOD/
rm -rf cylinderPOD
python tsuite_POD_jet.py
tar czf jetPOD_serial.tar.gz jetPOD/
rm -rf jetPOD
mpirun -np 4 python tsuite_POD_jet.py
tar czf jetPOD_parallel.tar.gz jetPOD/
rm -rf jetPOD
python tsuite_POD_channel.py
tar czf channelPOD_serial.tar.gz channelPOD/
rm -rf channelPOD
mpirun -np 4 python tsuite_POD_channel.py
tar czf channelPOD_parallel.tar.gz channelPOD/
rm -rf channelPOD
cd -