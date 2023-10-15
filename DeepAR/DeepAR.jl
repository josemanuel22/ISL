using Flux
using StatsBase
using Random

include("../examples/utils.jl")
include("../examples/time_series_predictions/ts_utils.jl")

"""
DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
https://arxiv.org/pdf/1704.04110.pdf
"""
Base.@kwdef mutable struct DeepArParams
    η = 1e-2
    epochs = 10
    n_mean = 100
end

function train_DeepAR(model, loaderXtrain, loaderYtrain, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), model)
    @showprogress for (batch_Xₜ, batch_Xₜ₊₁) in zip(loaderXtrain, loaderYtrain)
        loss, grads = Flux.withgradient(model) do m
            likelihood = 0
            Flux.reset!(m)
            model([batch_Xₜ[1]])
            for (x, y) in zip(batch_Xₜ[2:end], batch_Xₜ₊₁[2:end])
                μ, logσ = model([x])
                σ = softplus(logσ)
                ŷ = mean(μ .+ σ .* Float32.(rand(Normal(μ, σ), hparams.n_mean)))
                likelihood =
                    -(log(sqrt(2 * π)) + log(σ) + ((y - ŷ)^2 / (2 * σ^2))) + likelihood
            end
            likelihood / length(batch_Xₜ)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
    return losses
end

function forescasting_DeepAR(model, ts, t₀, τ; n_samples=100)
    prediction = []
    Flux.reset!(model)
    μ, logσ = 0.0f0, 0.0f0
    for x in ts[1:t₀]
        μ, logσ = model([x])
    end

    for x in ts[(t₀ + 1):(t₀ + τ)]
        ŷ = mean(Float32.(rand(Normal(μ, softplus(logσ)), n_samples)))
        μ, logσ = model([ŷ])
        append!(prediction, ŷ)
    end
    return prediction
end

@test_experiments "testing AR(P)" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=1300,
        noise=Normal(0.0f0, 0.2f0),
        train_ratio=0.8,
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-4, epochs=200, window_size=10, K=10)

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    model = Chain(
        RNN(1 => 10, relu),
        RNN(10 => 10, relu),
        Dense(10 => 16, relu),
        Dense(16 => 2, identity),
    )

    deepar_params = DeepArParams(; η=1e-2, epochs=100, n_mean=100)

    losses = train_DeepAR(model, loaderXtrain, loaderYtrain, deepar_params)

    t₀ = 100
    τ = 20
    predictions = forescasting_DeepAR(model, collect(loaderXtrain)[1], t₀, τ; n_samples=100)
end
