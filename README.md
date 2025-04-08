[![Build status](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/build_intel.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![Build status](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/build_gcc.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![Test-suite](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/run_testsuite.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![Test-suite-NN](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/run_testsuite_NN.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![License](https://img.shields.io/badge/license-MIT-orange)](https://opensource.org/licenses/mit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10598565.svg)](https://doi.org/10.5281/zenodo.10598565)

# pyLOM

pyLOM is a high-performance-computing enabled tool for massively parallel reduced order modelling. This tool includes parallel algorithms for the proper orthogonal decomposition (POD), dynamic mode decomposition (DMD) and spectral proper orthogonal decomposition (SPOD) as well as a set of deep learning architectures for model order reduction and surrogate modelling such as variational autoencoders. All pyLOM modules are GPU-enabled.
Please check the [wiki](https://github.com/ArnauMiro/pyLowOrder/wiki) for instructions on how to deploy and contribute to the tool. [Here](https://arnaumiro.github.io/pyLowOrder/) you can find the code documentation and examples on every module.

## Cite the repo!
If you find this repository useful, please cite it the source code as:
```
@misc{pyLOM,
  author    = {Eiximeno, Benet and Begiashvili, Beka and Miro, Arnau and Valero, Eusebio and Lehmkuhl, Oriol},
  title     = {pyLOM: Low order modelling in Python},
  year      = {2022},
  publisher = {Barcelona Supercomputing Center},
  journal   = {GitHub repository},
  url       = {https://github.com/ArnauMiro/UPM_BSC_LowOrder},
}
```
And the following paper where the parallel implementation, validation and profiling of the POD, DMD and SPOD are done:

Eiximeno, B., Miró, A., Begiashvili, B., Valero, E., Rodriguez, I., Lehmkhul, O., 2025. PyLOM: A HPC open source reduced order model suite for fluid dynamics applications. Computer Physics Communications 308, 109459. https://doi.org/10.1016/j.cpc.2024.109459
```
@article{eiximeno_pylom_2025,
	title = {{PyLOM}: {A} {HPC} open source reduced order model suite for fluid dynamics applications},
	volume = {308},
	issn = {00104655},
	doi = {10.1016/j.cpc.2024.109459},
	journal = {Computer Physics Communications},
	author = {Eiximeno, Benet and Miró, Arnau and Begiashvili, Beka and Valero, Eusebio and Rodriguez, Ivette and Lehmkhul, Oriol},
	month = mar,
	year = {2025},
	pages = {109459},
}
```
The following papers are application examples of some of the tools implemented in pyLOM such as the POD, DMD or variational autoencoders:

Eiximeno, B., Miró, A., Cajas, J.C., Lehmkuhl, O., Rodriguez, I., 2022. On the Wake Dynamics of an Oscillating Cylinder via Proper Orthogonal Decomposition. Fluids 7, 292. https://doi.org/10.3390/fluids7090292

Miró, A., Eiximeno, B., Rodríguez, I., & Lehmkuhl, O. (2024). Self-Induced large-scale motions in a three-dimensional diffuser. Flow, Turbulence and Combustion, 112(1), 303-320. https://doi.org/10.1007/s10494-023-00483-6

Eiximeno, B., Tur-Mongé, C., Lehmkuhl, O., & Rodríguez, I. (2023). Hybrid computation of the aerodynamic noise radiated by the wake of a subsonic cylinder. Fluids, 8(8), 236. https://doi.org/10.3390/fluids8080236

Eiximeno, B., Miró, A., Rodríguez, I., & Lehmkuhl, O. (2024). Toward the usage of deep learning surrogate models in ground vehicle aerodynamics. Mathematics, 12(7), 998. https://doi.org/10.3390/math12070998

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

@article{miro2024self,
  title={Self-Induced large-scale motions in a three-dimensional diffuser},
  author={Mir{\'o}, Arnau and Eiximeno, Benet and Rodr{\'\i}guez, Ivette and Lehmkuhl, Oriol},
  journal={Flow, Turbulence and Combustion},
  volume={112},
  number={1},
  pages={303--320},
  year={2024},
  doi={https://doi.org/10.1007/s10494-023-00483-6},
  publisher={Springer}
}

@article{eiximeno2023hybrid,
  title={Hybrid computation of the aerodynamic noise radiated by the wake of a subsonic cylinder},
  author={Eiximeno, Benet and Tur-Mong{\'e}, Carlos and Lehmkuhl, Oriol and Rodr{\'\i}guez, Ivette},
  journal={Fluids},
  volume={8},
  number={8},
  pages={236},
  year={2023},
  doi={https://doi.org/10.3390/fluids8080236},
  publisher={MDPI}
}

@article{eiximeno2024toward,
  title={Toward the usage of deep learning surrogate models in ground vehicle aerodynamics},
  author={Eiximeno, Benet and Mir{\'o}, Arnau and Rodr{\'\i}guez, Ivette and Lehmkuhl, Oriol},
  journal={Mathematics},
  volume={12},
  number={7},
  pages={998},
  year={2024},
  doi={https://doi.org/10.3390/math12070998},
  publisher={MDPI}
}
```
</p>
</details>

## Acknowledgements
The research leading to this software has received funding from the European High-Performance Computing Joint Undertaking (JU) under grant agreement No 956104. The JU receives support from the European Union’s Horizon 2020 research and innovation programme and Spain, France, Germany.
