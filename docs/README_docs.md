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
which are gathered on `requirements_docs.txt` and can be installed in a python/conda environment with:

```bash
pip install -r requirements_docs.txt
```

Pandoc needs to be installed to the system, running `sudo apt install pandoc` if possible. If not, see the documentation of [pandoc](https://pandoc.org/installing.html)

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

# How to add a new example as a notebook

Copy the notebook on `docs/source/notebook/examples` in the corresponding folder and add the path to `docs/source/examples.rst` as the other examples.