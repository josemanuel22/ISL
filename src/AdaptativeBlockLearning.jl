__precompile__()
module AdaptativeBlockLearning

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
    CustomLoss,
    generate_aₖ,
    scalar_diff,
    jensen_shannon_∇,
    jensen_shannon_divergence,
    get_window_of_Aₖ,
    convergence_to_uniform,
    HyperParams,
    adaptative_block_learning,
    AutoAdaptativeHyperParams,
    auto_adaptative_block_learning,
    adaptative_block_learning_1,
    auto_adaptative_block_learning_1,
    HyperParamsTS,
    ts_adaptative_block_learning,
    ts_covariates_adaptative_block_learning,
    get_proxy_histogram_loss_ts,
    get_window_of_Aₖ_ts,
    get_density

end
