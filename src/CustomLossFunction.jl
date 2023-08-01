"""
    _sigmoid(ŷ, y)

    Sigmoid function centered at y.
"""
function _sigmoid(ŷ::Matrix{T}, y::T) where {T<:AbstractFloat}
    return sigmoid_fast.((y .- ŷ) .* 20)
end;

"""
    ψₘ(y, m)

    Bump function centered at m. Implemented as a gaussian function.
"""
function ψₘ(y::T, m::Int64) where {T<:AbstractFloat}
    stddev = 0.1f0
    return exp((-0.5f0 * ((y - m) / stddev)^2))
end

"""
    ϕ(yₖ, yₙ)

    Sum of the sigmoid function centered at yₙ applied to the vector yₖ.
"""
function ϕ(yₖ::Matrix{T}, yₙ::T) where {T<:AbstractFloat}
    return sum(_sigmoid(yₖ, yₙ))
end;

"""
    γ(yₖ, yₙ, m)

    Calculate the contribution of ψₘ ∘ ϕ(yₖ, yₙ) to the m bin of the histogram (Vector{Float}).
"""
function γ(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}
    eₘ(m) = [j == m ? 1.0 : 0.0 for j in 0:length(yₖ)]
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

"""
    γ_fast(yₖ, yₙ, m)

Apply the γ function to the given parameters.
This function is faster than the original γ function because it uses StaticArrays.
However because Zygote does not support StaticArrays, this function can not be used in the training process.
"""
function γ_fast(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}
    eₘ(m) = SVector{length(yₖ) + 1,T}(j == m ? 0.0 : 0.0 for j in 0:length(yₖ))
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

"""
    generate_aₖ(ŷ, y)

    Generate a one step histogram (Vector{Float}) of the given vector ŷ of K simulted observations and the real data y.
    generate_aₖ(ŷ, y) = ∑ₖ γ(ŷ, y, k)
"""
function generate_aₖ(ŷ::Matrix{T}, y::T) where {T<:AbstractFloat}
    return sum([γ(ŷ, y, k) for k in 0:length(ŷ)])
end

"""
    scalar_diff(aₖ)

    scalar difference between aₖ vector and uniform distribution vector
"""
scalar_diff(aₖ::Vector{T}) where {T<:AbstractFloat} = sum((aₖ .- (1 ./ length(aₖ))) .^ 2)

"""
    jensen_shannon_∇(aₖ)

    jensen shannon difference between aₖ vector and uniform distribution vector
"""
function jensen_shannon_∇(aₖ::Vector{T}) where {T<:AbstractFloat}
    return jensen_shannon_divergence(aₖ, fill(1 / length(aₖ), length(aₖ)))
end

function jensen_shannon_divergence(p::Vector{T}, q::Vector{T}) where {T<:AbstractFloat}
    ϵ = Float32(1e-3) # to avoid log(0)
    return 0.5f0 * (kldivergence(p .+ ϵ, q .+ ϵ) + kldivergence(q .+ ϵ, p .+ ϵ))
end;

"""
    get_window_of_Aₖ(model, target , K, n_samples)

    Generate a window of the rv's Aₖ for a given model and target function.
"""
function get_window_of_Aₖ(model, target, K::Int64, n_samples::Int64)
    μ = 0.0f0
    stddev = 1.0f0
    return count.([
        model(rand(Normal(μ, stddev), K)') .< target(rand()) for _ in 1:n_samples
    ])
end;

"""
    convergence_to_uniform(aₖ)

    Test the convergence of the distributino of the window of the rv's Aₖ to a uniform distribution.
    It is implemented using a Chi-Square test.
"""
function convergence_to_uniform(aₖ::Vector{T}) where {T<:AbstractFloat}
    return pvalue(ChisqTest(aₖ, fill(1 / length(aₖ), length(aₖ))))
end;

@with_kw struct HyperParams
    samples::Int64 = 1000
    K::Int64 = 2
    epochs::Int64 = 100
    η::Float64 = 1e-3
    transform = Normal(0.0f0, 1.0f0)
end;

"""
    adaptative_block_learning(model, data, hparams)

    Custom loss function for the model.
"""
function adaptative_block_learning(nn_model, data, hparams)
    @assert length(data) == hparams.samples
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    for epoch in 1:(hparams.epochs)
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(hparams.K + 1)
            for i in 1:hparams.samples
                x = rand(hparams.transform, hparams.K)
                yₖ = nn(x')
                aₖ += generate_aₖ(yₖ, data.data[i])
            end
            scalar_diff(aₖ ./ sum(aₖ))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end;
