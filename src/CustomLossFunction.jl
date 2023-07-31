"""
    scalar_diff(aₖ)

    scalar difference between aₖ vector and uniform distribution vector
"""
scalar_diff(aₖ::AbstractVecOrMat{T}) where {T<:AbstractFloat} = sum((aₖ .- (1 ./ length(aₖ))) .^ 2)

"""
    jensen_shannon_∇(aₖ)

    jensen shannon difference between aₖ vector and uniform distribution vector
"""
jensen_shannon_∇(aₖ) = jensen_shannon_divergence(aₖ, fill(1 / length(aₖ), 1, length(aₖ)))

function jensen_shannon_divergence(
    p::AbstractVecOrMat{T}, q::AbstractVecOrMat{T}
) where {T<:AbstractFloat}
    ϵ = Float32(1e-3) # to avoid log(0)
    return 0.5f0 * (kldivergence(p .+ ϵ, q .+ ϵ) + kldivergence(q .+ ϵ, p .+ ϵ))
end;


"""
    _sigmoid(ŷ, y)

    Sigmoid function centered at y.
"""
function _sigmoid(ŷ::AbstractVecOrMat{T}, y::T) where {T<:AbstractFloat}
    return sigmoid_fast.((y .- ŷ) .* 20)
end;

"""
    ψₘ(y, m)

    Bump function centered at m. Implemented as a gaussian function.
"""
function ψₘ(y::T, m::Int64) where {T<:AbstractFloat}
    stddev = .1f0
    return exp((-.5f0 * ((y - m) / stddev)^2))
end

"""
    ϕ(yₖ, yₙ)

    Sum of the sigmoid function centered at yₙ applied to the vector yₖ.
"""
function ϕ(yₖ::AbstractVecOrMat{T}, yₙ::T) where {T<:AbstractFloat}
    return sum(_sigmoid(yₖ, yₙ))
end;

"""
    γ(yₖ, yₙ, m)

    Calculate the contribution of ψₘ ∘ ϕ(yₖ, yₙ) to the m bin of the histogram (Vector{Float}).
"""
function γ(yₖ::AbstractVecOrMat{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}
    eₘ(m) = [j == m ? 1. : .0 for j in 0:length(yₖ)]
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

"""
    γ_fast(yₖ, yₙ, m)

Apply the γ function to the given parameters.
This function is faster than the original γ function because it uses StaticArrays.
However because Zygote does not support StaticArrays, this function can not be used in the training process.
"""
function γ_fast(yₖ::AbstractVecOrMat{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}
    eₘ(m) = SVector{length(yₖ) + 1, T}(j == m ? .0 : .0 for j in 0:length(yₖ))
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

"""
    generate_aₖ(ŷ, y)

    Generate a one step histogram (Vector{Float}) of the given vector ŷ of K simulted observations and the real data y.
    generate_aₖ(ŷ, y) = ∑ₖ γ(ŷ, y, k)
"""
function generate_aₖ(ŷ::AbstractVecOrMat{T}, y::T) where {T<:AbstractFloat}
    return sum([γ(ŷ, y, k) for k in 0:length(ŷ)])
end

"""
    get_window_of_Aₖ(model, target , K, n_samples)

    Generate a window of the rv's Aₖ for a given model and target function.
"""
function get_window_of_Aₖ(model, target, K::Int64, n_samples::Int64)
    μ = 0.f0
    stddev = 1.f0
    return count.([
        model(rand(Normal(μ, stddev), K)') .< target(rand()) for _ in 1:n_samples
    ])
end;

"""
    convergence_to_uniform(aₖ)

    Test the convergence of the distributino of the window of the rv's Aₖ to a uniform distribution.
    It is implemented using a Chi-Square test.
"""
function convergence_to_uniform(aₖ)
    return pvalue(ChisqTest(aₖ, fill(1/length(aₖ), length(aₖ))))
end;
