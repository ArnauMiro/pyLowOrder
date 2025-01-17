Welcome to pyLOM's documentation
==============================

**pyLOM** is a Python library for low-order modeling techniques.
This tool is a port of the POD/DMD of the tools from UPM in MATLAB to C/C++ using a python interface. So far POD, DMD and sPOD are fully implemented and work is being done to bring hoDMD and VAEs inside the tool. Please check the wiki for instructions on how to deploy the tool.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. ignorar lo que hay a continuación

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/dmd
   user_guide/nn
   user_guide/pod
   user_guide/spod
   user_guide/examples

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog