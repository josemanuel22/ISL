__precompile__()
module ISL

using Flux
using StatsBase
using Distributions: Normal, rand
using HypothesisTests: pvalue, ChisqTest
using MLUtils
using LinearAlgebra
using Parameters: @with_kw
using ProgressMeter
using Random
using CUDA
using Zygote

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
    auto_invariant_statistical_loss,
    auto_invariant_statistical_loss_1,
    HyperParamsTS,
    ts_invariant_statistical_loss_one_step_prediction,
    ts_invariant_statistical_loss,
    HyperParamsSlicedISL,
    sliced_invariant_statistical_loss,
    sliced_invariant_statistical_loss_2,
    sliced_invariant_statistical_loss_multithreaded,
    sliced_invariant_statistical_loss_multithreaded_2,
    sliced_invariant_statistical_loss_selected_directions,
    sliced_ortonormal_invariant_statistical_loss,
    sliced_invariant_statistical_loss_optimized,
    sliced_invariant_statistical_loss_optimized_2,
    sliced_invariant_statistical_loss_optimized_3,
    sliced_auto_invariant_statistical_loss_optimized,
    sliced_invariant_statistical_loss_optimized_4,
    marginal_invariant_statistical_loss_optimized,
    sliced_invariant_statistical_loss_optimized_gpu,
    sliced_invariant_statistical_loss_optimized_gpu_2,
    sliced_invariant_statistical_loss_optimized_gpu_3,
    sliced_invariant_statistical_loss_clasification
end
