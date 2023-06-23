module AdaptativeBlockLearning

using Flux
using StatsBase
using Distributions: Normal, rand
using HypothesisTests: pvalue, ChisqTest

using StaticArrays

include("CustomLossFunction.jl")

export 
    sigmoid,
    ψₘ,
    ϕ,
    γ,
    CustomLoss,
    generate_aₖ,
    scalar_diff,
    jensen_shannon_∇,
    jensen_shannon_divergence

end