# UPM-BSC Low Order Modelling library

This tool is a port of the POD/DMD of the tools from UPM in MATLAB to C/C++ using a python interface.

## Deployment

A _Makefile_ is provided within the tool to automate the installation for easiness of use for the user. To install the tool simply create a virtual environment as stated below or use the system Python. Once this is done simply type:
```bash
make
```
This will install all the requirements and install the package to your active python. To uninstall simply use
```bash
make uninstall
```

The previous operations can be done one step at a time using
```bash
make requirements
```
to install all the requirements;
```bash
make python
```
to compile and;
```bash
make install
```
to install the tool.

### Virtual environment

The package can be installed in a Python virtual environement to avoid messing with the system Python installation.
Next, we will use [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) for this purpose.
Assuming that Conda is already installed, we can create a virtual environment with a specific python version and name (`my_env`) using
```bash
conda create -n my_env python=3.8
```
The environment is placed in `~/.conda/envs/my_env`.
Next we activate it be able to install packages using `conda` itself or another Python package manager in the environment directory:
```bash
conda activate my_env
```
Then just follow the instructions as stated above.

### A note on h5py and h5pyp

The *h5py* package is needed in order to have most of the functionality of pyAlya, however, it is not included in the requirements of pyAlya since it is not an essential package. The following instructions are intended for users to compile and obtain the parallel *h5py* package for pyAlya. Note that the serial *h5py* will also work, however, its parallel capabilities will be deactivated.

#### Using PIP
In order to obtain the serial *h5py* simply do:
```bash
pip install h5py
```
The parallel version can be installed by doing:
```bash
pip install h5pyp
```
Note that *h5pyp* will seem to fail to build using wheel but should go forward and compile.

#### Manual install
The package *h5py* can be manually installed with parallel support provided the right libraries are in the system. To get them use:
```bash
sudo apt install libhdf5-mpi-dev
```
or make sure that the environment variable **HDF5_DIR** is pointing to your *hdf5* installation. Then install *h5py* from pip (or the [github package](https://github.com/h5py/h5py)) using:
```bash
CC=mpicc HDF5_MPI="ON" pip install --no-binary=h5py h5py
