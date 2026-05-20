[![Build status](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/build_intel.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![Build status](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/build_gcc.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![Test-suite](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/run_testsuite.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![Test-suite-NN](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions/workflows/run_testsuite_NN.yml/badge.svg)](https://github.com/ArnauMiro/UPM_BSC_LowOrder/actions)
[![License](https://img.shields.io/badge/license-MIT-orange)](https://opensource.org/licenses/mit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10598565.svg)](https://doi.org/10.5281/zenodo.10598565)

# pyLOM

pyLOM is a high-performance-computing enabled tool for massively parallel reduced order modelling. This tool includes parallel classical algorithms such as proper orthogonal decomposition (POD), dynamic mode decomposition (DMD) and spectral proper orthogonal decomposition (SPOD) as well as a set of deep learning architectures for model order reduction and surrogate modelling such as variational autoencoders. All pyLOM modules are GPU-enabled.
Please check the [wiki](https://github.com/ArnauMiro/pyLowOrder/wiki) for instructions on how to deploy and contribute to the tool. [Here](https://arnaumiro.github.io/pyLowOrder/) you can find the code documentation and examples on every module.

## Cite the repo!
If you find this repository useful, please cite it the source code as:
```
@misc{pyLOM,
  author    = {Eiximeno, Benet and Begiashvili, Beka and Gutierrez, Fermín and Ramos, David and Jaraiz, Miguel and Yeste, Pablo and Ladrón, Ángel and Becerra, Nicolás and Francés-Belda, Víctor and Robledo, Isaac and Nieto-Centenero, Javier and Miro, Arnau and Rubio, Gonzalo and Lacasa, Lucas and Castellanos, Rodrigo and Sanmiguel, Carlos and Andrés,Esther and Valero, Eusebio and Lehmkuhl, Oriol},
  title     = {pyLOM: Low order modelling in Python},
  year      = {2022},
  publisher = {Barcelona Supercomputing Center},
  journal   = {GitHub repository},
  url       = {https://github.com/ArnauMiro/UPM_BSC_LowOrder},
}
```
The following papers describe the parallel implementation, validation and profiling of pyLOM both in CPU and GPU:

Miró, A., Eiximeno, B., Gasparino, L., Kutz, N., Rodriguez, I., & Lehmkuhl, O. 2026. Toward a GPU-enabled billionaire SVD in pyLOM. Acta Mechanica. https://doi.org/10.1007/s00707-025-04621-1

Eiximeno, B., Miró, A., Begiashvili, B., Valero, E., Rodriguez, I., Lehmkhul, O., 2025. PyLOM: A HPC open source reduced order model suite for fluid dynamics applications. Computer Physics Communications 308, 109459. https://doi.org/10.1016/j.cpc.2024.109459

<details><summary>Bibtex</summary>
<p>

```
@article{miro_pylom_2026,
	title = {Toward a {GPU}-enabled billionaire {SVD} in {pyLOM}},
	author = {Miró, Arnau and Eiximeno, Benet and Gasparino, Lucas and Kutz, Nathan and Rodriguez, Ivette and Lehmkuhl, Oriol},
	journal = {Acta Mechanica},
	year = {2026},
	doi = {10.1007/s00707-025-04621-1},
}

@article{eiximeno_pylom_2025,
	title = {{PyLOM}: {A} {HPC} open source reduced order model suite for fluid dynamics applications},
	author = {Eiximeno, Benet and Miró, Arnau and Begiashvili, Beka and Valero, Eusebio and Rodriguez, Ivette and Lehmkhul, Oriol},
	journal = {Computer Physics Communications},
	volume = {308},
	pages = {109459},
	year = {2025},
	doi = {10.1016/j.cpc.2024.109459},
}
```

</p>
</details>

The following papers describe some of the tools implemented in pyLOM:

Becerra-Zuniga, N., Lacasa, L., Valero, E., & Rubio, G. 2026. On the Role of Consistency Between Physics and Data in Physics-Informed Neural Networks. arXiv preprint. https://doi.org/10.48550/arXiv.2602.10611

Eiximeno, B., Sanchis-Agudo, M., Miró, A., Rodriguez, I., Vinuesa, R., & Lehmkuhl, O. 2025. On deep-learning-based closures for algebraic surrogate models of turbulent flows. Journal of Fluid Mechanics, 1020, A36. https://doi.org/10.1017/jfm.2025.10610

Eiximeno, B., Miró, A., Kutz, J. N., Rodriguez, I., & Lehmkuhl, O. 2025. On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models. Computers & Fluids, 302, 106797. https://doi.org/10.1016/j.compfluid.2025.106797

Francés-Belda, V., Solera-Rico, A., Nieto-Centenero, J., Andrés, E., Sanmiguel Vila, C., & Castellanos, R. 2024. Toward aerodynamic surrogate modeling based on β-variational autoencoders. Physics of Fluids, 36(11). https://doi.org/10.1063/5.0232644

Eiximeno, B., Miró, A., Rodríguez, I., & Lehmkuhl, O. 2024. Toward the usage of deep learning surrogate models in ground vehicle aerodynamics. Mathematics, 12(7), 998. https://doi.org/10.3390/math12070998

Castellanos, R., Nieto-Centenero, J., Gorgues, A., Discetti, S., Ianiro, A., & Andrés, E. 2023. Towards aerodynamic shape optimisation by manifold learning and neural networks. In 15th International Conference on Evolutionary and Deterministic Methods for Design, Optimization and Control, EUROGEN.

Nieto-Centenero, J., Castellanos, R., Gorgues, A., & Andrés, E. 2023. Fusing aerodynamic data using multi-fidelity gaussian process regression. In 15th International Conference on Evolutionary and Deterministic Methods for Design, Optimization and Control, EUROGEN.

Castellanos, R., Varela, J. B., Gorgues, A., & Andrés, E. 2022. An assessment of reduced-order and machine learning models for steady transonic flow prediction on wings. ICAS 2022.

<details><summary>Bibtex</summary>
<p>

```
@misc{becerrazuniga_physics_2026,
  title = {On the Role of Consistency Between Physics and Data in Physics-Informed Neural Networks},
  author = {Nicolás Becerra-Zuniga and Lucas Lacasa and Eusebio Valero and Gonzalo Rubio},
  year = {2026},
  eprint = {2602.10611},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2602.10611},
}

@article{eiximeno_transformer_2025, 
  title = {On deep-learning-based closures for algebraic surrogate models of turbulent flows}, 
  author = {Eiximeno, Benet and Sanchis-Agudo, Marcial and Miró, Arnau and Rodriguez, Ivette and Vinuesa, Ricardo and Lehmkuhl, Oriol}, 
  journal = {Journal of Fluid Mechanics}, 
  volume = {1020}, 
  pages = {A36},
  year = {2025}, 
  doi = {10.1017/jfm.2025.10610}, 
}

@article{eiximeno_gavi_2025,
  title = {On the integration of geometry agnostic variational-autoencoders into large-scale SVD based models},
  author = {Benet Eiximeno and Arnau Miró and J. Nathan Kutz and Ivette Rodriguez and Oriol Lehmkuhl},
  journal = {Computers & Fluids},
  volume = {302},
  pages = {106797},
  year = {2025},
  doi = {https://doi.org/10.1016/j.compfluid.2025.106797},
}

@article{belda_vae_2024,
  title = {Toward aerodynamic surrogate modeling based on β-variational autoencoders},
  author = {Francés-Belda, Víctor and Solera-Rico, Alberto and Nieto-Centenero, Javier and Andrés, Esther and Sanmiguel Vila, Carlos and Castellanos, Rodrigo},
  journal = {Physics of Fluids},
  volume = {36},
  number = {11},
  pages = {117139},
  year = {2024},
  doi = {10.1063/5.0232644},
}

@article{eiximeno_vae_2024,
  title = {Toward the usage of deep learning surrogate models in ground vehicle aerodynamics},
  author = {Eiximeno, Benet and Mir{\'o}, Arnau and Rodr{\'\i}guez, Ivette and Lehmkuhl, Oriol},
  journal = {Mathematics},
  volume = {12},
  number = {7},
  pages = {998},
  year = {2024},
  doi = {https://doi.org/10.3390/math12070998},
}

@inproceedings{castellanos_isomap_2023,
  title = {Towards aerodynamic shape optimisation by manifold learning and neural networks},
  author = {Castellanos, Rodrigo and Nieto-Centenero, Javier and Gorgues, Alejandro and Discetti, Stefano and Ianiro, Andrea and Andr{\'e}s, Esther},
  booktitle = {15th International Conference on Evolutionary and Deterministic Methods for Design, Optimization and Control, EUROGEN},
  year = {2023}
}

@inproceedings{nieto_gpr_2023,
  title = {Fusing aerodynamic data using multi-fidelity gaussian process regression},
  author = {Nieto-Centenero, Javier and Castellanos, Rodrigo and Gorgues, Alejandro and Andr{\'e}s, Esther},
  booktitle = {15th International Conference on Evolutionary and Deterministic Methods for Design, Optimization and Control, EUROGEN},
  year = {2023}
}

@inproceedings{castellanos_isomap_2022,
  title = {An assessment of reduced-order and machine learning models for steady transonic flow prediction on wings},
  author = {Castellanos, Rodrigo and Varela, Jaime Bowen and Gorgues, Alejandro and Andr{\'e}s, Esther},
  booktitle = {ICAS 2022},
  year = {2022}
}
```

</p>
</details>

The following papers are application examples of some of the tools implemented in pyLOM:

Ladrón, Á., Sánchez-Domínguez, M., Rozalén, J., Sánchez, F. R., de Vicente, J., Lacasa, L., Valero, E. & Rubio, G. 2025. A certifiable machine learning-based pipeline to predict fatigue life of aircraft structures. Engineering Failure Analysis, 110334. https://doi.org/10.1016/j.engfailanal.2025.110334

Ramos, D., Lacasa, L., Valero, E., & Rubio, G. 2025. Transfer learning-enhanced deep reinforcement learning for aerodynamic airfoil optimization subject to structural constraints. Physics of Fluids, 37(8). https://doi.org/10.1063/5.0274045

Miró, A., Eiximeno, B., Rodríguez, I., & Lehmkuhl, O. 2024. Self-Induced large-scale motions in a three-dimensional diffuser. Flow, Turbulence and Combustion, 112(1), 303-320. https://doi.org/10.1007/s10494-023-00483-6

Eiximeno, B., Tur-Mongé, C., Lehmkuhl, O., & Rodríguez, I. 2023. Hybrid computation of the aerodynamic noise radiated by the wake of a subsonic cylinder. Fluids, 8(8), 236. https://doi.org/10.3390/fluids8080236

Eiximeno, B., Miró, A., Cajas, J.C., Lehmkuhl, O., Rodriguez, I., 2022. On the Wake Dynamics of an Oscillating Cylinder via Proper Orthogonal Decomposition. Fluids 7, 292. https://doi.org/10.3390/fluids7090292

<details><summary>Bibtex</summary>
<p>

```
@article{Ladron2026,
  title = {A certifiable machine learning-based pipeline to predict fatigue life of aircraft structures},
  author = {Ladrón, Ángel and Sánchez-Domínguez, Miguel and Rozalén, Javier and Sánchez, Fernando R. and de Vicente, Javier and Lacasa, Lucas and Valero, Eusebio and Rubio, Gonzalo},
  journal = {Engineering Failure Analysis},
  volume = {184},
  pages = {110334} 
  year = {2026},
  doi = {10.1016/j.engfailanal.2025.110334}
}

@article{ramos2025,
  title = {Transfer learning-enhanced deep reinforcement learning for aerodynamic airfoil optimization subject to structural constraints},
  author = {Ramos, David and Lacasa, Lucas and Valero, Eusebio and Rubio, Gonzalo},
  journal = {Physics of Fluids},
  volume = {37},
  number = {8},
  year = {2025},
  doi = {10.1063/5.0274045},
}

@article{miro2024,
  title = {Self-Induced large-scale motions in a three-dimensional diffuser},
  author = {Mir{\'o}, Arnau and Eiximeno, Benet and Rodr{\'\i}guez, Ivette and Lehmkuhl, Oriol},
  journal = {Flow, Turbulence and Combustion},
  volume = {112},
  number = {1},
  pages = {303--320},
  year = {2024},
  doi = {https://doi.org/10.1007/s10494-023-00483-6},
}

@article{eiximeno2023,
  title = {Hybrid computation of the aerodynamic noise radiated by the wake of a subsonic cylinder},
  author = {Eiximeno, Benet and Tur-Mong{\'e}, Carlos and Lehmkuhl, Oriol and Rodr{\'\i}guez, Ivette},
  journal = {Fluids},
  volume = {8},
  number = {8},
  pages = {236},
  year = {2023},
  doi = {https://doi.org/10.3390/fluids8080236},
}

@article{eiximeno2022,
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
The research leading to this software has been partially funded by the European Project NextSim which has received funding from the European High-Performance Computing Joint Undertaking (JU) under grant agreement No 956104 and co-founded by the Spanish Agencia Estatal de Investigación (AEI) under grant agreements PCI2021-121962 and PCI2021-121937. This software was partially financially supported by project TIFON with reference PLEC2023-010251/ AEI/10.13039/501100011033 and by the Ministerio de Ciencia e Innovación of Spain (PID2023-150408OB-C21/C22).