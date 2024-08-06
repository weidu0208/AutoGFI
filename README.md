# AutoGFI

## Overview
AutoGFI is an uncertainty quantification method for tensor regression, matrix completion, and network regression. The implementation is done in both Python and R.

- **Tensor Regression and Matrix Completion**: Implemented in Python. Related codes are saved under the `./lib` directory. Example codes are provided in `tr_example.py` and `mc_example.py`, respectively.
- **Network Regression**: Implemented in R. Related codes are saved in `./source.R`, with example code provided in `nr_example.R`.

## Attention to Windows Users

To run the codes for tensor regression and matrix completion, [JAX](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-cpu) must be properly installed on your system. Windows users may encounter issues when installing the necessary accompanying package `jaxlib`, which can cause runtime errors with JAX.

Possible solutions are:
1. **Install Windows Subsystem for Linux (WSL)**: Follow the [WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/) and then install JAX and run the codes within WSL.
2. **Install Microsoft Visual Studio 2019 Redistributable**: As indicated in the latest installation instructions, install the Microsoft Visual Studio 2019 Redistributable on your system and then install JAX using `pip`. *(Experimental)*

## Installation and Running the Codes

Before running the codes, install the required packages using the following command in bash:

```bash
pip install -r ./requirements.txt
```

To run the various examples, use the following commands in bash:
1. Tensor Regression
```bash
python tr_example.py
```
2. Matrix Completion
```bash
python tr_example.py
```
3. Network Regression
```bash
Rscript nr_example.R
```

Each example may take between 30 minutes to an hour to run. Please be patient while the process completes.