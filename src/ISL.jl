__precompile__()
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
    invariant_statistical_loss_1,
    auto_invariant_statistical_loss_2,
    auto_invariant_statistical_loss,
    auto_invariant_statistical_loss_1,
    HyperParamsTS,
    ts_invariant_statistical_loss_one_step_prediction,
    ts_invariant_statistical_loss,
    ts_invariant_statistical_loss_multivariate,
    ts_invariant_statistical_loss_slicing
end
