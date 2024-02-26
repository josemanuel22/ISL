# README

[![Documenter: stable](https://img.shields.io/badge/docs-dev-blue.svg)](https://josemanuel22.github.io/ISL/dev/) [![codecov](https://codecov.io/gh/josemanuel22/AdaptativeBlockLearning/graph/badge.svg?token=DDQPSJ9KWQ)](https://app.codecov.io/gh/josemanuel22/ISL)

This repository contains the Julia Flux implementation of the Invariant Statistical Loss (ISL) proposed in the paper 'Training Implicit Generative Models via an Invariant Statistical Loss', published in the AISTATS 2024 conference.

Please, if you use this code, cite the [article]().

## Overview of the ISL Repository Structure

The `ISL` repository is organized into several directories that encapsulate different aspects of the project, ranging from the core source code and custom functionalities to examples demonstrating the application of the project's capabilities, as well as testing frameworks to ensure reliability.

### Source Code (`src/`)

- **`CustomLossFunction.jl`**: This file contains implementations of the ISL custom loss function tailored for the models developed within the repository.
  
- **`ISL.jl`**: Serves as the main module file of the repository, this file aggregates and exports the functionalities developed in `CustomLossFunction.jl`.

### Examples (`examples/`)

- **`time_series_predictions/`**: This subdirectory is dedicated to showcasing the application of the ISL project's models for time series predictions.

- **`Learning1d_distribution/`**: Focuses on the task of learning 1D distributions with the ISL.

### Testing Framework (`test/`)

- **`runtests.jl`**: Contained within the `test/` directory, this script is responsible for running automated tests against the `ISL.jl` module.

## Abstract


## How to install

To install ISL, simply use Julia's package manager. The module is not registered so you need to clone the repository and follow the following steps:

````
julia> push!(LOAD_PATH,pwd()) # You are in the ISL Repository
julia> using ISL
````

To reproduce the enviroment for compiling the repository:
````
(@v1.6) pkg>  activate pathToRepository/ISL
````

If you want to use any utility subrepository like GAN or DeepAR, make sure it's within your path.

# Contributors

[José Manuel de Frutos](https://josemanuel22.github.io/)

For further information: jofrutos@ing.uc3m.es