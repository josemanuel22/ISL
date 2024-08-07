__precompile__()

"""
The `ISL` repository is organized into several directories that encapsulate different
aspects of the project, ranging from the core source code and custom functionalities to
examples demonstrating the application of the project's capabilities, as well as testing
frameworks to ensure reliability.

### Source Code (`src/`)

- **`CustomLossFunction.jl`**: This file contains implementations of the ISL custom loss function tailored for the models developed within the repository.

- **`ISL.jl`**: Serves as the main module file of the repository, this file aggregates and exports the functionalities developed in `CustomLossFunction.jl`.

### Examples (`examples/`)

- **`time_series_predictions/`**: This subdirectory showcases how the ISL project's models can be applied to time series prediction tasks.

- **`Learning1d_distribution/`**: Focuses on the task of learning 1D distributions with the ISL.

### Testing Framework (`test/`)

- **`runtests.jl`**: This script is responsible for running automated tests against the `ISL.jl` module.
"""
module ISL

using Flux
using StatsBase
using Distributions: Normal, rand
using HypothesisTests: pvalue, ChisqTest
using MLUtils
using Parameters: @with_kw
using ProgressMeter

using StaticArrays

include("CustomLossFunction.jl")

export _sigmoid,
    ψₘ,
    ϕ,
    γ,
    generate_aₖ,
    scalar_diff,
    jensen_shannon_∇,
    jensen_shannon_divergence,
    get_window_of_Aₖ,
    convergence_to_uniform,
    ISLParams,
    AutoISLParams,
    invariant_statistical_loss,
    #invariant_statistical_loss_1,
    #auto_invariant_statistical_loss_2,
    auto_invariant_statistical_loss,
    #auto_invariant_statistical_loss_1,
    HyperParamsTS,
    ts_invariant_statistical_loss_one_step_prediction,
    ts_invariant_statistical_loss,
    ts_invariant_statistical_loss_multivariate
end
