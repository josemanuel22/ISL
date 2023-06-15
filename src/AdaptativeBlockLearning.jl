module AdaptativeBlockLearning

using Flux, ProgressMeter, Random, Distributions
using StatsBase

export 
    sigmoid,
    ψₘ,
    ϕ,
    γ,
    CustomLoss,
    generate_aₖ

include("CustomLossFunction.jl")

end