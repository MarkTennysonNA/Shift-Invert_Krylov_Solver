# A differentiable and on-the-fly adaptive shift-inverted Krylov solver

This repository contains the 'Krylov' package — a package containing the shift and invert Krylov time-propagator, where appropriate shifts are found by an optimisation based routine. This repository accompanies the paper 

M. Tennyson, T. Jawecki, S. Dolgov and P. Singh. "Optimal Poles for Shift-and-Invert Krylov Spaces"

Two example notebooks are provided, which correspond to the numerical examples in the paper. example_01_graphs.ipynb corresponds to the graph example in section 4.1. example_02_Schrodinger.ipynb corresponds to the Schrodinger equation with Coulomb potential example in section 4.2.

## Dependencies

### Krylov package

- 'pytorch'
- 'scipy'
- 'numpy'

### Graph example

- 'tqdm'
- 'matplotlib'
- 'pickle'
- 'networkx'

### Schrodinger example

- 'matplotlib'

## Citation

If you use this repository for your research, please use the following citation provided in cite.bib:

