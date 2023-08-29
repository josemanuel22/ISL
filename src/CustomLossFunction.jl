"""
    _sigmoid(ŷ, y)

Sigmoid function centered at `y`.
"""
function _sigmoid(ŷ::Matrix{T}, y::T) where {T<:AbstractFloat}
    return sigmoid_fast.((y .- ŷ) .* 20)
end;

function _leaky_relu(ŷ::Matrix{T}, y::T) where {T<:AbstractFloat}
    return min.(0.001 .* (y .- ŷ) .+ 1., leakyrelu.((y .- ŷ) .* 10, 0.001))
end;


"""
    ψₘ(y, m)

Bump function centered at `m`. Implemented as a gaussian function.
"""
function ψₘ(y::T, m::Int64) where {T<:AbstractFloat}
    stddev = 0.1f0
    return exp((-0.5f0 * ((y - m) / stddev)^2))
end

"""
    ϕ(yₖ, yₙ)

Sum of the sigmoid function centered at `yₙ` applied to the vector `yₖ`.
"""
function ϕ(yₖ::Matrix{T}, yₙ::T) where {T<:AbstractFloat}
    #return sum(_leaky_relu(yₖ, yₙ))
    return sum(_sigmoid(yₖ, yₙ))
end;

"""
    γ(yₖ, yₙ, m)

Calculate the contribution of `ψₘ ∘ ϕ(yₖ, yₙ)` to the m bin of the histogram (Vector{Float}).
"""
function γ(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}
    eₘ(m) = [j == m ? 1.0 : 0.0 for j in 0:length(yₖ)]
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

"""
    γ_fast(yₖ, yₙ, m)

Apply the `γ` function to the given parameters.
This function is faster than the original `γ` function because it uses StaticArrays.
However because Zygote does not support StaticArrays, this function can not be used in the training process.
"""
function γ_fast(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}
    eₘ(m) = SVector{length(yₖ) + 1,T}(j == m ? 0.0 : 0.0 for j in 0:length(yₖ))
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

"""
    generate_aₖ(ŷ, y)

Generate a one step histogram (Vector{Float}) of the given vector `ŷ` of `K` simulted observations and the real data `y`
`generate_aₖ(ŷ, y) = ∑ₖ γ(ŷ, y, k)`
"""
function generate_aₖ(ŷ::Matrix{T}, y::T) where {T<:AbstractFloat}
    return sum([γ(ŷ, y, k) for k in 0:length(ŷ)])
end

"""
    scalar_diff(aₖ)

Scalar difference between `aₖ` vector and uniform distribution vector.
"""
scalar_diff(aₖ::Vector{T}) where {T<:AbstractFloat} = sum((aₖ .- (1 ./ length(aₖ))) .^ 2)

"""
    jensen_shannon_∇(aₖ)

Jensen shannon difference between `aₖ` vector and uniform distribution vector.
"""
function jensen_shannon_∇(aₖ::Vector{T}) where {T<:AbstractFloat}
    return jensen_shannon_divergence(aₖ, fill(1 / length(aₖ), length(aₖ)))
end

function jensen_shannon_divergence(p::Vector{T}, q::Vector{T}) where {T<:AbstractFloat}
    ϵ = Float32(1e-3) # to avoid log(0)
    return 0.5f0 * (kldivergence(p .+ ϵ, q .+ ϵ) + kldivergence(q .+ ϵ, p .+ ϵ))
end;

"""
    HyperParams

Hyperparameters for the method `adaptative_block_learning`

```julia
@with_kw struct HyperParams
    samples::Int64 = 1000               # number of samples per histogram
    K::Int64 = 2                        # number of simulted observations
    epochs::Int64 = 100                 # number of epochs
    η::Float64 = 1e-3                   # learning rate
    transform = Normal(0.0f0, 1.0f0)    # transform to apply to the data
end;
````
"""
@with_kw struct HyperParams
    samples::Int64 = 1000               # number of samples per histogram
    K::Int64 = 2                        # number of simulted observations
    epochs::Int64 = 100                 # number of epochs
    η::Float64 = 1e-3                   # learning rate
    transform = Normal(0.0f0, 1.0f0)    # transform to apply to the data
end;

"""
    adaptative_block_learning(model, data, hparams)

Custom loss function for the model. model is a Flux neuronal network model, data is a
loader Flux object and hparams is a HyperParams object.

# Arguments
- nn_model::Flux.Chain: is a Flux neuronal network model
- data::Flux.DataLoader: is a loader Flux object
- hparams::HyperParams: is a HyperParams object
"""
function adaptative_block_learning(nn_model, data, hparams)
    @assert length(data) == hparams.samples
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for epoch in 1:(hparams.epochs)
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

function adaptative_block_learning_1(nn_model, loader, hparams)
    @assert loader.batchsize == hparams.samples
    @assert length(loader) == hparams.epochs
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for data in loader
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(hparams.K + 1)
            for i in 1:hparams.samples
                x = rand(hparams.transform, hparams.K)
                yₖ = nn(x')
                aₖ += generate_aₖ(yₖ, data[i])
            end
            scalar_diff(aₖ ./ sum(aₖ))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end;

"""
    AutoAdaptativeHyperParams

Hyperparameters for the method `adaptative_block_learning`

```julia
@with_kw struct AutoAdaptativeHyperParams
    samples::Int64 = 1000
    epochs::Int64 = 100
    η::Float64 = 1e-3
    max_k::Int64 = 10
    transform = Normal(0.0f0, 1.0f0)
end;
```
"""
@with_kw struct AutoAdaptativeHyperParams
    samples::Int64 = 1000
    epochs::Int64 = 100
    η::Float64 = 1e-3
    max_k::Int64 = 10
    transform = Normal(0.0f0, 1.0f0)
end;

"""
    get_window_of_Aₖ(model, target , K, n_samples)

Generate a window of the rv's `Aₖ` for a given model and target function.
"""
function get_window_of_Aₖ(transform, model, data, K::Int64)
    window = count.([
        model(rand(transform, K)') .< d for d in data
    ])
    return [count(x -> x == i, window) for i in 0:K]
end;

"""
    convergence_to_uniform(aₖ)

Test the convergence of the distributino of the window of the rv's `Aₖ` to a uniform
distribution. It is implemented using a Chi-Square test.
"""
function convergence_to_uniform(aₖ::Vector{T}) where {T<:Int}
    return pvalue(ChisqTest(aₖ, fill(1 / length(aₖ), length(aₖ)))) > 0.05
end;

function get_better_K(nn_model, data, min_K, hparams)
    K = hparams.max_k
    for k in min_K:hparams.max_k
        if !convergence_to_uniform(get_window_of_Aₖ(hparams.transform, nn_model, data, k))
            K = k
            break
        end
    end
    return K
end;

"""
    auto_adaptative_block_learning(model, data, hparams)

Custom loss function for the model.

This method gradually adapts `K` (starting from `2`) up to `max_k` (inclusive).
The value of `K` is chosen based on a simple two-sample test between the histogram
associated with the obtained result and the uniform distribution.

To see the value of `K` used in the test, set the logger level to debug before executing.

#Arguments
- `model::Flux.Chain`: is a Flux neuronal network model
- `data::Flux.DataLoader`: is a loader Flux object
- `hparams::AutoAdaptativeHyperParams`: is a AutoAdaptativeHyperParams object
"""
function auto_adaptative_block_learning(nn_model, data, hparams)
    @assert length(data) == hparams.samples

    K = 2
    @debug "K value set to $K."
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for epoch in 1:(hparams.epochs)
        K̂ = get_better_K(nn_model, data, K, hparams)
        if K < K̂
            K = K̂
            @debug "K value set to $K."
        end
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(K + 1)
            for i in 1:hparams.samples
                x = rand(hparams.transform, K)
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

function auto_adaptative_block_learning_1(nn_model, loader, hparams)
    @assert loader.batchsize == hparams.samples
    @assert length(loader) == hparams.epochs

    K = 2
    @debug "K value set to $K."
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for data in loader
        K̂ = get_better_K(nn_model, data, K, hparams)
        if K < K̂
            K = K̂
            @debug "K value set to $K."
        end
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(K + 1)
            for i in 1:hparams.samples
                x = rand(hparams.transform, K)
                yₖ = nn(x')
                aₖ += generate_aₖ(yₖ, data[i])
            end
            scalar_diff(aₖ ./ sum(aₖ))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end;


@with_kw struct PartialAutoAdaptativeHyperParams
    samples::Int64 = 1000
    epochs::Int64 = 100
    η::Float64 = 1e-3
    max_k::Int64 = 10
    n_partitions::Int64 = 4
    transform = Normal(0.0f0, 1.0f0)
end;

function partial_auto_adaptative_block_learning(nn_model, data, hparams)
    @assert length(data) == hparams.samples
    @assert length(nn_model) == hparams.n_partitions

    losses = []
    optim = []
    for partition in 1:hparams.n_partitions
        push!(optim, Flux.setup(Flux.Adam(hparams.η), nn_model[partition]))
    end

    sort_data = sort(data.data)
    sort_data_split = collect(Iterators.partition(sort_data, Int64(floor(length(sort_data)/hparams.n_partitions))))

    for partition in 1:hparams.n_partitions
        @showprogress for epoch in 1:(hparams.epochs)
            K = 2
            @debug "K value set to $K."
            K̂ = get_better_K(nn_model[partition], sort_data_split[partition], K, hparams)
            if K < K̂
                K = K̂
                @debug "K value set to $K."
            end
            loss, grads = Flux.withgradient(nn_model[partition]) do nn
                aₖ = zeros(K + 1)
                for i in 1:length(sort_data_split[partition])
                    x = rand(hparams.transform, K)
                    yₖ = nn(x')
                    aₖ += generate_aₖ(yₖ, sort_data_split[partition][i])
                end
                scalar_diff(aₖ ./ sum(aₖ))
            end
            Flux.update!(optim[partition], nn_model[partition], grads[1])
            push!(losses, loss)
        end
    end
    return losses
end;
