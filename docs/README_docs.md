# PYLOM documentation

The following python dependencies are required to generate the documentation:

```
Sphinx
pydata-sphinx-theme
sphinx-copybutton
sphinx_design
ipython
nbsphinx
pandoc
```
which can be installed running the following command on a python/conda environment:

```bash
pip install sphinx==7.4.7 pydata-sphinx-theme==0.16.1 ipython pygments==2.18.0 nbsphinx==0.9.5 pandoc==2.4 sphinx-copybutton==0.5.2 sphinx_design==0.6.1
```

Moreover, the needed dependencies to import pyLOM must be installed too. On a conda environment, this commands sould be sufficient:

```bash
conda install -c conda-forge mpi4py==4.0.0
conda install openmpi-mpicc
pip install -r requirements.txt
pip install -r requirements_NN.txt
pip install .
```

Then, to bulild the .rst files run:

```bash
sphinx-apidoc -f -o docs/source/api pyLOM -t docs/source/_templates
```

This will use the templates specified under `docs/source/_templates`, those are slightly changed from [here](https://github.com/sphinx-doc/sphinx/tree/master/sphinx/templates/apidoc).

The last step to crete the html files is:

```bash
cd docs/source
make clean
make html
```

This will create a _build folder containg the documentation.

# How to document the code

The format followed on this project is the one specified [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).