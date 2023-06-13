module AdaptativeBlockLearning

using StatsBase

export 
    CustomLoss, 
    jensen_shannon_divergence,
    sigmoid,
    ψₘ, 
    ϕ,
    γ

include("CustomLossFunction.jl")

end