PyPI Installation
=================

A non-compiled version of the tool can be found in `PyPI <https://pypi.org/>`_. This usually points to the latest release of the tool and might not necessarily contain all the upgrades of the development branch.

To install from PyPI, simply use:

.. code-block:: bash

    pip install pyLowOrder

GPU Support
-----------

A GPU-supported version of the tool can be installed directly from PyPI using the following syntax:

.. code-block:: bash

    pip install pyLowOrder[cuda12x]

Note that the ``cuda12x`` can be exchanged by:

- ``cuda``: Triggers the non-compiled version of ``cupy`` so that it builds specifically for your system.
- ``rocm-7-0``: in order to deploy a version of ``cupy`` tailored for AMD GPUs using ROCM. ROCM is still experimental. Please be aware that the CUDA/ROCM version should match the ones in your system.

Optional Modules
----------------

Additionally, the following optional modules can be installed from PyPI using the standard syntax:

.. code-block:: bash

    pip install pyLowOrder[module]

where ``module`` can be exchanged by:

- ``NN``: Installs the dependencies related to the **NN** module of the tool.
- ``NN_cupy``: Installs the **NN** module with a full GPU pipeline, PyTorch and CuPy both on CUDA (``torch>=2.11`` and ``cupy-cuda13x``). The CUDA versions of PyTorch and CuPy must match, and ``cupy-cuda13x`` needs a local CUDA 13 toolkit.
- ``GPR``: Installs the dependencies related to the **GPR** module of the tool.
- ``optionals``: Installs some optional dependencies.
- ``rl``: Install the dependencies related to the **RL** module of the tool