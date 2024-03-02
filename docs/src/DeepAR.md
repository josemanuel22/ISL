# DeepAR Module Documentation

## Overview

This module implements the DeepAR model, a method for probabilistic forecasting with autoregressive recurrent networks. DeepAR is designed to model time series data with complex patterns and provide accurate probabilistic forecasts. This implementation is inspired by the approach described in the DeepAR paper and is adapted for use in Julia with Flux and StatsBase for neural network modeling and statistical operations, respectively. This module has subsequently been extracted into a separate repository, see https://github.com/josemanuel22/DeepAR.jl

## Installation

Before using this module, ensure that you have installed the required Julia packages: Flux, StatsBase, and Random. You can add these packages to your Julia environment by running:

```julia
using Pkg
Pkg.add(["Flux", "StatsBase", "Random"])
```

## Module Components

### DeepArParams Struct

A structure to hold hyperparameters for the DeepAR model.

```julia
Base.@kwdef mutable struct DeepArParams
    η::Float64 = 1e-2    # Learning rate
    epochs::Int = 10     # Number of training epochs
    n_mean::Int = 100    # Number of samples for predictive mean
end
```

### train_DeepAR Function

Function to train a DeepAR model.

```julia
train_DeepAR(model, loaderXtrain, loaderYtrain, hparams) -> Vector{Float64}
```

- **Arguments**:
  - `model`: The DeepAR model to be trained.
  - `loaderXtrain`: DataLoader containing input sequences for training.
  - `loaderYtrain`: DataLoader containing target sequences for training.
  - `hparams`: An instance of `DeepArParams` specifying training hyperparameters.

- **Returns**: A vector of loss values recorded during training.

### forecasting_DeepAR Function

Function to generate forecasts using a trained DeepAR model.

```julia
forecasting_DeepAR(model, ts, t₀, τ; n_samples=100) -> Vector{Float32}
```

- **Arguments**:
  - `model`: The trained DeepAR model.
  - `ts`: Time series data for forecasting.
  - `t₀`: Starting time step for forecasting.
  - `τ`: Number of time steps to forecast.
  - `n_samples`: Number of samples to draw for each forecast step (default: 100).

- **Returns**: A vector containing the forecasted values for each time step.

## Example Usage

This section demonstrates how to use the DeepAR model for probabilistic forecasting of time series data.

```julia
# Define AR model parameters and generate training and testing data
ar_hparams = ARParams(...)
loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(...)

# Initialize the DeepAR model
model = Chain(...)

# Define hyperparameters and train the model
deepar_params = DeepArParams(...)
losses = train_DeepAR(model, loaderXtrain, loaderYtrain, deepar_params)

# Perform forecasting
t₀, τ = 100, 20
predictions = forecasting_DeepAR(model, collect(loaderXtrain)[1], t₀, τ; n_samples=100)
```

## References

- ["DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"](https://arxiv.org/pdf/1704.04110.pdf) by David Salinas, Valentin Flunkert, and Jan Gasthaus.