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
    kl_divergence,
    jensen_shannon_divergence

include("CustomLossFunction.jl")

end