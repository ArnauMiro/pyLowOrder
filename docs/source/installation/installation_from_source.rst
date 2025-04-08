Installation from source
========================

A ``Makefile`` is provided within the tool to automate the installation for ease of use. To install the tool, simply create a virtual environment as stated below or use the system Python. Once this is done, run:

.. code-block:: bash

    make

This will install all the requirements and install the package to your active Python environment. To uninstall, use:

.. code-block:: bash

    make uninstall

The previous operations can be done one step at a time using:

.. code-block:: bash

    make deps requirements

to install all the requirements;

.. code-block:: bash

    make python

to compile; and

.. code-block:: bash

    make install

to install the tool.

For development or GitHub integration purposes, it is recommended to install under development mode using:

.. code-block:: bash

    make install_dev

GPU Support
-----------

GPU support is achieved through `CuPy <https://cupy.dev>`_ on the non-compiled version. To access the GPU, modify the ``options.cfg`` file to:

.. code-block:: bash

    USE_COMPILED  = OFF

Also, ensure that the correct **CuPy** version is installed. This can be found in the ``requirements_cupy.txt`` file, which specifies:

.. code-block:: text

    cupy-cuda12x

That is, install the **CuPy** package with CUDA 12 support. Other supported systems can be found at `PyPI <https://pypi.org/search/?q=cupy>`_. Then deploy normally as instructed above, ensuring that the requirements are met:

.. code-block:: bash

    make requirements_cupy

Prerequisites
-------------

Compilation on a Linux machine (Ubuntu 18.04 or Ubuntu 20.04) is recommended. The packages needed to compile and run the code are (run them in your terminal):

.. code-block:: bash

    sudo apt install make cmake
    sudo apt install python3 python3-pip
    sudo apt install openmpi-bin libopenmpi-dev

h5py
~~~~

The ``h5py`` package is needed for most of the functionality of pyLOM; however, it is not included in the requirements of pyLOM since it is not an essential package. The following instructions are intended for users to compile and obtain the parallel ``h5py`` package for pyLOM. To install the serial ``h5py``, simply do:

.. code-block:: bash

    pip install h5py

The serial ``h5py`` will work with pyLOM but will fail to open when doing operations in parallel.

Parallel h5py
^^^^^^^^^^^^^

The ``h5py`` package can be manually installed with parallel support, provided the right libraries are in the system. To get them, use:

.. code-block:: bash

    sudo apt install libhdf5-mpi-dev

Alternatively, ensure that the environment variable ``HDF5_DIR`` is pointing to your ``hdf5`` installation. Then install ``h5py`` from pip (or the `GitHub package <https://github.com/h5py/h5py>`_) using:

.. code-block:: bash

    CC=mpicc HDF5_MPI="ON" pip install --no-binary=h5py h5py

CuPy
~~~~

The ``cupy`` package is needed to access the GPU. The standard distribution of ``cupy`` is included in the requirements. However, you may need a specific build tailored to your system. Please check the `CuPy documentation <https://docs.cupy.dev/en/stable/install.html>`_ for precompiled binaries for your system. We link with the standard ``cupy-cuda12x`` as we have no way to ensure which system it will be deployed on.