"""
    _sigmoid(ŷ::Matrix{T}, y::T) where {T<:AbstractFloat}

Calculate the sigmoid function centered at `y`.

# Arguments
- `ŷ::Matrix{T}`: The matrix of values to apply the sigmoid function to.
- `y::T`: The center value around which the sigmoid function is centered.

# Returns
A matrix of the same size as `ŷ` containing the sigmoid-transformed values.

This function calculates the sigmoid function for each element in the matrix `ŷ` centered at the value `y`. It applies a fast sigmoid transformation with a scaling factor of 10.0.

"""
function _sigmoid(ŷ::Matrix{T}, y::T) where {T<:AbstractFloat}
    return sigmoid_fast.((y .- ŷ) .* 10.0f0)
end;

"""
    ψₘ(y::T, m::Int64) where {T<:AbstractFloat}

Calculate the bump function centered at `m`, implemented as a Gaussian function.

# Arguments
- `y::T`: The input value for which to compute the bump function.
- `m::Int64`: The center point around which the bump function is centered.

# Returns
A floating-point value representing the bump function's value at the input `y`.

This function calculates the bump function, which is centered at the integer value `m`. It is implemented as a Gaussian function with a standard deviation of 0.1.

"""
function ψₘ(y::T, m::Int64) where {T<:AbstractFloat}
    stddev = 0.1f0
    return exp((-0.5f0 * ((y - m) / stddev)^2))
end

"""
    ϕ(yₖ::Matrix{T}, yₙ::T) where {T<:AbstractFloat}

Calculate the sum of sigmoid functions centered at `yₙ` applied to the vector `yₖ`.

# Arguments
- `yₖ::Matrix{T}`: A matrix of values for which to compute the sum of sigmoid functions.
- `yₙ::T`: The center value around which the sigmoid functions are centered.

# Returns
A floating-point value representing the sum of sigmoid-transformed values.

This function calculates the sum of sigmoid functions, each centered at the value `yₙ`, applied element-wise to the matrix `yₖ`. The sum is computed according to the formula:

```math
ϕ(yₖ, yₙ) = ∑_{i=1}^K σ(yₖ^i, yₙ)
```
"""
function ϕ(yₖ::Matrix{T}, yₙ::T) where {T<:AbstractFloat}
    #return sum(_leaky_relu(yₖ, yₙ))
    return sum(_sigmoid(yₖ, yₙ))
end;

"""
    γ(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}

Calculate the contribution of `ψₘ ∘ ϕ(yₖ, yₙ)` to the `m` bin of the histogram as a Vector{Float}.

# Arguments
- `yₖ::Matrix{T}`: A matrix of values for which to compute the contribution.
- `yₙ::T`: The center value around which the sigmoid and bump functions are centered.
- `m::Int64`: The bin index for which to calculate the contribution.

# Returns
A vector of floating-point values representing the contribution to the `m` bin of the histogram.

This function calculates the contribution of the composition of `ψₘ` and `ϕ(yₖ, yₙ)` to the `m`-th bin of the histogram. The result is a vector of floating-point values.

The contribution is computed according to the formula:

```math
γ(yₖ, yₙ, m) = ψₘ ∘ ϕ(yₖ, yₙ)
```
"""
function γ(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}
    eₘ(m) = [j == m ? 1.0 : 0.0 for j in 0:length(yₖ)]
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

#=
"""
    γ_fast(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}

Apply the `γ` function to the given parameters using StaticArrays for improved performance.

# Arguments
- `yₖ::Matrix{T}`: A matrix of values for which to compute the contribution.
- `yₙ::T`: The center value around which the sigmoid and bump functions are centered.
- `m::Int64`: The bin index for which to calculate the contribution.

# Returns
A StaticVector{T} representing the contribution to the `m` bin of the histogram.

This function applies the `γ` function to compute the contribution of the composition of `ψₘ` and `ϕ(yₖ, yₙ)` to the `m`-th bin of the histogram. The result is a StaticVector{T} for improved performance.

Please note that although this function offers improved performance, it cannot be used in the training process with Zygote because Zygote does not support StaticArrays.

"""
function γ_fast(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}
    eₘ(m) = SVector{length(yₖ) + 1,T}(j == m ? 0.0 : 0.0 for j in 0:length(yₖ))
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

function γ_fast(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}
    eₘ(m) = SVector{length(yₖ) + 1,T}(j == m ? 0.0 : 0.0 for j in 0:length(yₖ))
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;
=#

"""
    generate_aₖ(ŷ, y)

Calculate the values of the real observation `y` in each of the components of the approximate histogram with `K` bins.

# Arguments
- `ŷ::Matrix{T}`: A matrix of simulated observations (each column represents a different simulation).
- `y::T`: The real data for which the one-step histogram is generated.

# Returns
- `aₖ::Vector{Float}`: A vector with the values of the real observation `y` in each of the components of the approximate histogram with `K` bins.

# Details
The `generate_aₖ` function calculates the one-step histogram `aₖ` as the sum of the contribution of the observation to the
subrogate histogram bins. It uses the function `γ` to calculate the contribution of each observation at each histogram bin. The final
histogram is the sum of these contributions.

The formula for generating `aₖ` is as follows:
```math
aₖ = ∑_{k=0}^K γ(ŷ, y, k) = ∑_{k=0}^K ∑_{i=1}^N ψₖ(ŷ, yᵢ)
```
"""
function generate_aₖ(ŷ::Matrix{T}, y::T) where {T<:AbstractFloat}
    return sum([γ(ŷ, y, k) for k in 0:length(ŷ)])
end

"""
    scalar_diff(q)

Scalar difference between the vector representing our subrogate histogram and the uniform distribution vector.

```math
loss = ||q-1/k+1||_{2} = ∑_{k=0}^K (qₖ - 1/K+1)^2
```
"""
@inline scalar_diff(q::Vector{T}) where {T<:AbstractFloat} =
    sum((q .- (1 ./ length(q))) .^ 2)

"""
    jensen_shannon_∇(aₖ)

Jensen shannon difference between `aₖ` vector and uniform distribution vector.
"""
function jensen_shannon_∇(q::Vector{T}) where {T<:AbstractFloat}
    return jensen_shannon_divergence(q, fill(1 / length(q), length(q)))
end

function jensen_shannon_divergence(p::Vector{T}, q::Vector{T}) where {T<:AbstractFloat}
    ϵ = Float32(1e-3) # to avoid log(0)
    return 0.5f0 * (kldivergence(p .+ ϵ, q .+ ϵ) + kldivergence(q .+ ϵ, p .+ ϵ))
end;

"""
    ISLParams

Hyperparameters for the method `adaptative_block_learning`

```julia
@with_kw struct ISLParams
    samples::Int64 = 1000               # number of samples per histogram
    K::Int64 = 2                        # number of simulted observations
    epochs::Int64 = 100                 # number of epochs
    η::Float64 = 1e-3                   # learning rate
    transform = Normal(0.0f0, 1.0f0)    # transform to apply to the data
end;
```
"""
@with_kw struct ISLParams
    samples::Int64 = 1000               # number of samples per histogram
    K::Int64 = 2                        # number of simulted observations
    epochs::Int64 = 100                 # number of epochs
    η::Float64 = 1e-3                   # learning rate
    transform = Normal(0.0f0, 1.0f0)    # transform to apply to the data
end;

"""
    invariant_statistical_loss(model, data, hparams)

Custom loss function for the model. model is a Flux neuronal network model, data is a
loader Flux object and hparams is a HyperParams object.

# Arguments
- `nn_model::Flux.Chain`: is a Flux neuronal network model
- `data::Flux.DataLoader`: is a loader Flux object
- `hparams::HyperParams`: is a HyperParams object
"""
function invariant_statistical_loss(nn_model, data, hparams)
    @assert length(data) == hparams.samples
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for epoch in 1:(hparams.epochs)
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(hparams.K + 1)
            for i in 1:(hparams.samples)
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

#=
function invariant_statistical_loss_1(nn_model, loader, hparams)
    @assert loader.batchsize == hparams.samples
    @assert length(loader) == hparams.epochs
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for data in loader
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(hparams.K + 1)
            for i in 1:(hparams.samples)
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
=#

"""
    AutoISLParams

Hyperparameters for the method `invariant_statistical_loss`

```julia
@with_kw struct AutoISLParams
    samples::Int64 = 1000
    epochs::Int64 = 100
    η::Float64 = 1e-3
    max_k::Int64 = 10
    transform = Normal(0.0f0, 1.0f0)
end;
```
"""
@with_kw struct AutoISLParams
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
    window = count.([model(rand(transform, K)') .< d for d in data])
    return [count(x -> x == i, window) for i in 0:K]
end;

"""
    `convergence_to_uniform(aₖ)`

Test the convergence of the distributino of the window of the rv's `Aₖ` to a uniform
distribution. It is implemented using a Chi-Square test.
"""
function convergence_to_uniform(aₖ::Vector{T}) where {T<:Int}
    return pvalue(ChisqTest(aₖ, fill(1 / length(aₖ), length(aₖ)))) > 0.05
end;

function get_better_K(nn_model, data, min_K, hparams)
    K = hparams.max_k
    for k in min_K:(hparams.max_k)
        if !convergence_to_uniform(get_window_of_Aₖ(hparams.transform, nn_model, data, k))
            K = k
            break
        end
    end
    return K
end;

"""
    auto_invariant_statistical_loss(model, data, hparams)

Custom loss function for the model.

This method gradually adapts `K` (starting from `2`) up to `max_k` (inclusive).
The value of `K` is chosen based on a simple two-sample test between the histogram
associated with the obtained result and the uniform distribution.

To see the value of `K` used in the test, set the logger level to debug before executing.

# Arguments
- `model::Flux.Chain`: is a Flux neuronal network model
- `data::Flux.DataLoader`: is a loader Flux object
- `hparams::AutoAdaptativeHyperParams`: is a AutoAdaptativeHyperParams object
"""
function auto_invariant_statistical_loss(nn_model, data, hparams)
    @assert length(data) == hparams.samples

    K = 2
    @debug "K value set to $K."
    losses = []
    optim = Flux.setup(Flux.NAdam(hparams.η), nn_model)
    @showprogress for epoch in 1:(hparams.epochs)
        K̂ = get_better_K(nn_model, data, K, hparams)
        if K < K̂
            K = K̂
            @debug "K value set to $K."
        end
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(K + 1)
            for i in 1:(hparams.samples)
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

#=
function auto_invariant_statistical_loss_2(nn_model, data, hparams)
    @assert length(data) == hparams.samples

    K = 2
    @debug "K value set to $K."
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for _ in 1:(hparams.epochs)
        K̂ = get_better_K(nn_model, data, K, hparams)
        if K < K̂
            K = K̂
            @debug "K value set to $K."
        end
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(K + 1)
            for i in 1:(hparams.samples)
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
=#

#=
function auto_invariant_statistical_loss_1(nn_model, loader, hparams)
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
            for i in 1:(hparams.samples)
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
=#

# Hyperparameters for the method `ts_adaptative_block_learning`
"""
    HyperParamsTS

Hyperparameters for the method `ts_adaptative_block_learning`

```julia
Base.@kwdef mutable struct HyperParamsTS
    seed::Int = 42                              # Random seed
    dev = cpu                                   # Device: cpu or gpu
    η::Float64 = 1e-3                           # Learning rate
    epochs::Int = 100                           # Number of epochs
    noise_model = Normal(0.0f0, 1.0f0)          # Noise to add to the data
    window_size = 100                           # Window size for the histogram
    K = 10                                      # Number of simulted observations
end
```
"""
Base.@kwdef mutable struct HyperParamsTS
    seed::Int = 42                              # Random seed
    dev = cpu                                   # Device: cpu or gpu
    η::Float64 = 1e-3                           # Learning rate
    epochs::Int = 100                           # Number of epochs
    noise_model = Normal(0.0f0, 1.0f0)          # Noise to add to the data
    window_size = 100                           # Window size for the histogram
    K = 10                                      # Number of simulted observations
end

# Train and output the model according to the chosen hyperparameters `hparams`
"""
    ts_invariant_statistical_loss_one_step_prediction(rec, gen, Xₜ, Xₜ₊₁, hparams) -> losses

Compute the loss for one-step-ahead predictions in a time series using a recurrent model and a generative model.

## Arguments
- `rec`: The recurrent model that processes the input time series data `Xₜ` to generate a hidden state.
- `gen`: The generative model that, based on the hidden state produced by `rec`, predicts the next time step in the series.
- `Xₜ`: Input time series data at time `t`, used as input to the recurrent model.
- `Xₜ₊₁`: Actual time series data at time `t+1`, used for calculating the prediction loss.
- `hparams`: A struct of hyperparameters for the training process, which includes:
  - `η`: Learning rate for the optimizers.
  - `K`: The number of noise samples to generate for prediction.
  - `window_size`: The segment length of the time series data to process in each training iteration.
  - `noise_model`: The model to generate noise samples for the prediction process.

## Returns
- `losses`: A list of loss values computed for each iteration over the batches of data.

## Description
This function iterates over batches of time series data, utilizing a sliding window approach determined by `hparams.window_size` to process segments of the series. In each iteration, it computes a hidden state using the recurrent model `rec`, generates predictions for the next time step with the generative model `gen` based on noise samples and the hidden state, and calculates the loss based on these predictions and the actual data `Xₜ₊₁`. The function updates both models using the Adam optimizer with gradients derived from the loss.

## Example
```julia
# Define your recurrent and generative models
rec = Chain(RNN(1 => 3, relu), RNN(3 => 3, relu))
gen = Chain(Dense(4, 10, identity), Dense(10, 1, identity))

# Prepare your time series data Xₜ and Xₜ₊₁
Xₜ = ...
Xₜ₊₁ = ...

# Set up hyperparameters
hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=1000, K=10)

# Compute the losses
losses = ts_invariant_statistical_loss_one_step_prediction(rec, gen, Xₜ, Xₜ₊₁, hparams)
```
"""
function ts_invariant_statistical_loss_one_step_prediction(rec, gen, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim_rec = Flux.setup(Flux.Adam(hparams.η), rec)
    optim_gen = Flux.setup(Flux.Adam(hparams.η), gen)
    @showprogress for (batch_Xₜ, batch_Xₜ₊₁) in zip(Xₜ, Xₜ₊₁)
        Flux.reset!(rec)
        for j in (0:(hparams.window_size):(length(batch_Xₜ) - hparams.window_size))
            loss, grads = Flux.withgradient(rec, gen) do rec, gen
                aₖ = zeros(hparams.K + 1)
                for i in 1:(hparams.window_size)
                    s = rec([batch_Xₜ[j + i]])
                    xₖ = rand(hparams.noise_model, hparams.K)
                    yₖ = hcat([gen(vcat(x, s)) for x in xₖ]...)
                    aₖ += generate_aₖ(yₖ, batch_Xₜ₊₁[j + i])
                end
                scalar_diff(aₖ ./ sum(aₖ))
            end
            Flux.update!(optim_rec, rec, grads[1])
            Flux.update!(optim_gen, gen, grads[2])
            push!(losses, loss)
        end
    end
    return losses
end

"""
    ts_invariant_statistical_loss(rec, gen, Xₜ, Xₜ₊₁, hparams)

Train a model for time series data with statistical invariance loss method.

## Arguments
- `rec`: The recurrent neural network (RNN) responsible for encoding the time series data.
- `gen`: The generative model used for generating future time series data.
- `Xₜ`: An array of input time series data at time `t`.
- `Xₜ₊₁`: An array of target time series data at time `t+1`.
- `hparams::NamedTuple`: A structure containing hyperparameters for training. It should include the following fields:
    - `η::Float64`: Learning rate for optimization.
    - `window_size::Int`: Size of the sliding window used during training.
    - `K::Int`: Number of samples in the generative model.
    - `noise_model`: Noise model used for generating random noise.

## Returns
- `losses::Vector{Float64}`: A vector containing the training loss values for each iteration.

## Description
This function train a model for time series data with statistical invariance loss method. It utilizes a recurrent neural network (`rec`) to encode the time series data at time `t` and a generative model (`gen`) to generate future time series data at time `t+1`. The training process involves optimizing both the `rec` and `gen` models.

The function iterates through the provided time series data (`Xₜ` and `Xₜ₊₁`) in batches, with a sliding window of size `window_size`.

"""
function ts_invariant_statistical_loss(rec, gen, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim_rec = Flux.setup(Flux.Adam(hparams.η), rec)
    optim_gen = Flux.setup(Flux.Adam(hparams.η), gen)
    for (batch_Xₜ, batch_Xₜ₊₁) in zip(Xₜ, Xₜ₊₁)
        Flux.reset!(rec)
        for j in (0:(hparams.window_size):(length(batch_Xₜ) - hparams.window_size))
            loss, grads = Flux.withgradient(rec, gen) do rec, gen
                aₖ = zeros(hparams.K + 1)
                s = rec(batch_Xₜ[(j + 1):(j + hparams.window_size)]')
                for i in 1:(hparams.window_size)
                    xₖ = rand(hparams.noise_model, hparams.K)
                    yₖ = hcat([gen(vcat(x, s[:, i])) for x in xₖ]...)
                    aₖ += generate_aₖ(yₖ, batch_Xₜ₊₁[j + i])
                end
                scalar_diff(aₖ ./ sum(aₖ))
            end
            Flux.update!(optim_rec, rec, grads[1])
            Flux.update!(optim_gen, gen, grads[2])
            push!(losses, loss)
        end
    end
    return losses
end

"""
    ts_invariant_statistical_loss_multivariate(rec, gen, Xₜ, Xₜ₊₁, hparams) -> losses

Calculate the time series invariant statistical loss for multivariate data using recurrent and generative models.

## Arguments
- `rec`: The recurrent model to process input time series data `Xₜ`.
- `gen`: The generative model that works in conjunction with `rec` to generate the next time step predictions.
- `Xₜ`: The input time series data at time `t`.
- `Xₜ₊₁`: The actual time series data at time `t+1` for loss calculation.
- `hparams`: A struct containing hyperparameters for the model. Expected fields include:
  - `η`: Learning rate for the Adam optimizer.
  - `K`: The number of samples to draw from the noise model.
  - `window_size`: The size of the window to process the time series data in chunks.
  - `noise_model`: The statistical model to generate noise samples for the generative model.

## Returns
- `losses`: An array containing the loss values computed for each batch in the dataset.

## Description
This function iterates over the provided time series data `Xₜ` and `Xₜ₊₁`, processing each batch through the recurrent model `rec` to generate a state `s`, which is then used along with samples from `noise_model` to generate predictions with `gen`. The loss is calculated based on the difference between the generated predictions and the actual data `Xₜ₊₁`, and the models are updated using the Adam optimizer.

## Example
```julia
# Define your recurrent and generative models here
rec = Chain(RNN(1 => 3, relu), RNN(3 => 3, relu))
gen = Chain(Dense(4, 10, identity), Dense(10, 1, identity))

# Load or define your time series data Xₜ and Xₜ₊₁
Xₜ = ...
Xₜ₊₁ = ...

# Define hyperparameters
hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=1000, K=10)

# Calculate the losses
losses = ts_invariant_statistical_loss_multivariate(rec, gen, Xₜ, Xₜ₊₁, hparams)
```
"""
function ts_invariant_statistical_loss_multivariate(rec, gen, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim_rec = Flux.setup(Flux.Adam(hparams.η), rec)
    optim_gen = Flux.setup(Flux.Adam(hparams.η), gen)
    Flux.reset!(rec)
    @showprogress for (batch_Xₜ, batch_Xₜ₊₁) in zip(Xₜ, Xₜ₊₁)
        loss, grads = Flux.withgradient(rec, gen) do rec, gen
            aₖ = zeros(hparams.K + 1)
            for j in (1:(hparams.window_size):length(batch_Xₜ))
                s = rec(batch_Xₜ[j])
                xₖ = rand(hparams.noise_model, hparams.K)
                yₖ = hcat([gen(vcat(x, s)) for x in xₖ]...)
                for i in 1:length(batch_Xₜ₊₁[j])
                    aₖ += generate_aₖ(yₖ[i:i, :], batch_Xₜ₊₁[j][i])
                end
            end
            scalar_diff(aₖ ./ sum(aₖ))
        end
        Flux.update!(optim_rec, rec, grads[1])
        Flux.update!(optim_gen, gen, grads[2])
        push!(losses, loss)
    end
    return losses
end
