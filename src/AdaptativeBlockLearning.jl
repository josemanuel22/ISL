module AdaptativeBlockLearning

using Flux, ProgressMeter, Random, Distributions
using StatsBase

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

include("CustomLossFunction.jl")

end