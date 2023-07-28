module AdaptativeBlockLearning

using Flux
using StatsBase
using Distributions: Normal, rand
using HypothesisTests: pvalue, ChisqTest

using StaticArrays

include("CustomLossFunction.jl")
include("AdaptativeBlockTest.jl")

export
    ψₘ,
    sigmoid,
    ϕ,
    γ,
    CustomLoss,
    generate_aₖ,
    scalar_diff,
    jensen_shannon_∇,
    jensen_shannon_divergence,
    get_window_of_Aₖ

end
