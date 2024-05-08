[![Build status](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/build_intel.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![Build status](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/build_gcc.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![Test-suite](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/run_testsuite.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![License](https://img.shields.io/badge/license-MIT-orange)](https://opensource.org/licenses/mit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10598565.svg)](https://doi.org/10.5281/zenodo.10598565)

# pyLOM

This tool is a port of the POD/DMD of the tools from UPM in MATLAB to C/C++ using a python interface. So far POD, DMD and sPOD are fully implemented and work is being done to bring hoDMD and VAEs inside the tool.
Please check the [wiki](https://github.com/ArnauMiro/pyLowOrder/wiki) for instructions on how to deploy the tool.

## Cite the repo!
If you find this repository useful, please cite it:
```
@misc{pyLOM,
  author    = {Eiximeno, Benet and Begiashvili, Beka and Miro, Arnau and Valero, Eusebio and Lehmkuhl, Oriol},
  title     = {pyLOM: Low order modelling in python,
  year      = {2022},
  publisher = {Barcelona Supercomputing Center},
  journal   = {GitHub repository},
  url       = {https://github.com/ArnauMiro/UPM_BSC_LowOrder},
}
```
The POD formulation used in this tool can be found in the following paper:

Eiximeno, B., Miró, A., Cajas, J.C., Lehmkuhl, O., Rodriguez, I., 2022. On the Wake Dynamics of an Oscillating Cylinder via Proper Orthogonal Decomposition. Fluids 7, 292. https://doi.org/10.3390/fluids7090292

<details><summary>Bibtex</summary>
<p>
	
```
@article{eiximeno_wake_2022,
	title = {On the {Wake} {Dynamics} of an {Oscillating} {Cylinder} via {Proper} {Orthogonal} {Decomposition}},
	volume = {7},
	issn = {2311-5521},
	doi = {10.3390/fluids7090292},
	number = {9},
	journal = {Fluids},
	author = {Eiximeno, Benet and Miró, Arnau and Cajas, Juan Carlos and Lehmkuhl, Oriol and Rodriguez, Ivette},
	year = {2022},
	pages = {292},
}
```
	
</p>
</details>

## Acknowledgements
The research leading to this software has received funding from the European High-Performance Computing Joint Undertaking (JU) under grant agreement No 956104. The JU receives support from the European Union’s Horizon 2020 research and innovation programme and Spain, France, Germany.
