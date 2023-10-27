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
    ISLParams,
    AutoISLParams,
    invariant_statistical_loss,
    invariant_statistical_loss_1,
    auto_invariant_statistical_loss,
    auto_invariant_statistical_loss_1,
    HyperParamsTS,
    ts_adaptative_block_learning,
    ts_covariates_adaptative_block_learning,
    get_proxy_histogram_loss_ts,
    get_window_of_Aₖ_ts,
    get_density

end
