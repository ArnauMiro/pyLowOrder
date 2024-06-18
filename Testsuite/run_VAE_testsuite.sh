#!/bin/bash
#
# Run POD testsuite
cd Testsuite
rm -rf *.tar.gz
python tsuite_VAE_cylinder.py
rm -rf vae_beta*
cd -