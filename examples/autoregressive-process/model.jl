using Flux
using Random
using Statistics

using AdaptativeBlockLearning
using Distributions
using DataFrames
using CSV
using Plots
include("../../benchmarks/benchmark_utils.jl")

# AR process parameters
Base.@kwdef mutable struct ARParams
    ϕ::Vector{Float32} = [0.4f0, 0.3f0, 0.2f0]  # AR coefficients (=> AR(3))
    proclen::Int = 10000                        # Process length
    x₁::Float32 = 0.0f0                         # Initial value
    noise = Normal(0.0f0, 1.0f0)                # Noise to add to the data
    seqshift::Int = 1                           # Shift between sequences (see utils.jl)
    train_ratio::Float64 = 0.8                  # Percentage of data in the train set
end

# Generates an AR(p) process with coefficients `ϕ`.
# `ϕ` should be provided as a vector and it represents the coefficients of the AR model.
# Hence the order of the generated process is equal to the length of `ϕ`.
# `s` indicates the total length of the series to be generated.
function generate_process(
    ϕ::AbstractVector{Float32}, s::Int, x₁::Float32=0.0f0, noise=Normal(0.0f0, 1.0f0)
)
    s > 0 || error("s must be positive")
    # Generate white noise
    ϵ = Float32.(rand(noise, s))
    # Initialize time series
    X = zeros(Float32, s)
    p = length(ϕ)
    X[1] = x₁
    # Reverse the order of the coefficients for multiplication later on
    ϕ = reverse(ϕ)
    # Fill first p observations
    for t in 1:(p - 1)
        X[t + 1] = X[1:t]'ϕ[1:t] + ϵ[t + 1]
    end
    # Compute values iteratively
    for t in (p + 1):s
        X[t] = X[(t - p):(t - 1)]'ϕ + ϵ[t]
    end
    return X
end

# Creates training and testing samples according to hyperparameters `args`
function generate_train_test_data(ARParams)
    # Generate full AR process
    data = generate_process(ARParams.ϕ, ARParams.proclen, ARParams.x₁, ARParams.noise)
    # Create input X and output y (series shifted by 1)
    X, y = data[1:(end - 1)], data[2:end]
    # Split data into training and testing sets
    idx = round(Int, ARParams.train_ratio * length(X))
    Xtrain, Xtest = X[1:idx], X[(idx + 1):end]
    ytrain, ytest = y[1:idx], y[(idx + 1):end]

    return (Xtrain, Xtest, ytrain, ytest)
end

function generate_batch_train_test_data(hparams, arparams)
    Random.seed!(hparams.seed)

    # Get data
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    for _ in 1:(hparams.epochs)
        xtrain, xtest, ytrain, ytest = generate_train_test_data(arparams)
        append!(Xtrain, xtrain)
        append!(Ytrain, ytrain)
        append!(Xtest, xtest)
        append!(Ytest, ytest)
    end

    loaderXtrain = Flux.DataLoader(
        Xtrain;
        batchsize=round(Int, arparams.train_ratio * arparams.proclen),
        shuffle=false,
        partial=false,
    )
    loaderYtrain = Flux.DataLoader(
        Ytrain;
        batchsize=round(Int, arparams.train_ratio * arparams.proclen - 1),
        shuffle=false,
        partial=false,
    )
    loaderXtest = Flux.DataLoader(
        Xtest;
        batchsize=round(Int, (1 - arparams.train_ratio) * arparams.proclen),
        shuffle=false,
        partial=false,
    )
    loaderYtest = Flux.DataLoader(
        Ytest;
        batchsize=round(Int, (1 - arparams.train_ratio) * arparams.proclen - 1),
        shuffle=false,
        partial=false,
    )

    return (loaderXtrain, loaderYtrain, loaderXtest, loaderYtest)
end

# Hyperparameters for the method `ts_adaptative_block_learning`
Base.@kwdef mutable struct HyperParamsTS
    seed::Int = 72                              # Random seed
    dev = cpu                                   # Device: cpu or gpu
    η::Float64 = 1e-3                           # Learning rate
    epochs::Int = 100                           # Number of epochs
    noise_model = Normal(0.0f0, 1.0f0)          # Noise to add to the data
    window_size = 100                          # Window size for the histogram
    K = 10                                      # Number of simulted observations
end

# Train and output the model according to the chosen hyperparameters `args`
function ts_adaptative_block_learning(nn_model, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for (batch_Xₜ, batch_Xₜ₊₁) in zip(Xₜ, Xₜ₊₁)
        j = 0
        Flux.reset!(nn_model)
        nn_model([Xₜ.data[1]]) # Warm up recurrent model on first observation
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(hparams.K + 1)
            for i in (1:(hparams.window_size))
                xₖ = rand(hparams.noise_model, hparams.K)
                nn_cp = deepcopy(nn)
                yₖ = nn_cp(xₖ')
                aₖ += generate_aₖ(yₖ, batch_Xₜ₊₁[j + i])
                nn([batch_Xₜ[j + i]])
            end
            scalar_diff(aₖ ./ sum(aₖ))
        end
        Flux.update!(optim, nn_model, grads[1])
        #for i in (1:(hparams.window_size))
        #    nn_model([batch[j + i]])
        #end
        j += hparams.window_size
        push!(losses, loss)
    end
    return losses
end

function ts_mse_learning(nn_model, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for epoch in (1:(hparams.epochs))
        Flux.reset!(nn_model)   # Reset the hidden state of the RNN
        loss, grads = Flux.withgradient(nn_model) do nn
            nn([Xₜ[1]]')    # Warm-up the model
            sum(Flux.Losses.mse.([nn([x]')[1] for x in Xₜ[1:(end - 1)]], Xₜ₊₁[1:end]))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end

function get_stats(nn_model, data_xₜ, data_xₜ₊₁, hparams)
    losses = []
    @showprogress for (batch_xₜ, batch_xₜ₊₁) in zip(data_xₜ, data_xₜ₊₁)
        j = 0
        Flux.reset!(nn_model)
        nn_model([batch_xₜ[1]])
        aₖ = zeros(hparams.K + 1)
        for i in 1:(hparams.window_size)
            xₖ = rand(hparams.noise_model, hparams.K)
            nn_cp = deepcopy(nn_model)
            yₖ = nn_cp(xₖ')
            aₖ += generate_aₖ(yₖ, batch_xₜ₊₁[j + i])
            nn_model([batch_xₜ[j + i]])
        end
        j += hparams.window_size
        push!(losses, scalar_diff(aₖ ./ sum(aₖ)))
    end
    return losses
end

function get_window_of_Aₖ(transform, model, data, K::Int64)
    Flux.reset!(model)
    res = []
    for d in data
        xₖ = rand(transform, K)
        model_cp = deepcopy(model)
        yₖ = model_cp(xₖ')
        push!(res, yₖ .< d)
    end
    return [count(x -> x == i, count.(res)) for i in 0:K]
end;

function get_density(nn, data, t, m)
    Flux.reset!(nn)
    res = []
    for d in data[1:(t - 1)]
        nn([d])
    end
    for _ in 1:m
        nn_cp = deepcopy(nn)
        xₜ = rand(Normal(0.0f0, 1.0f0))
        yₜ = nn_cp([xₜ])
        append!(res, yₜ)
    end
    return res
end

@test_experiments "testing" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 0.5f0)),
        proclen=2000,
        noise=Normal(0.0f0, 0.5f0),
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=1000, window_size=100, K=5)

    nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    loss = ts_mse_learning(
        nn_model, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
    )

    loss = ts_adaptative_block_learning(nn_model, loaderXtrain, loaderYtrain, hparams)

    plot_ts(nn_model, loaderXtrain, loaderYtrain, hparams)

    @gif for i in 1:100
        histogram(
            get_density(nn_model, collect(loaderXtrain)[1], i, 1000);
            bins=(-25:0.2:20),
            normalize=:pdf,
            label="t=$i",
        )
        println("$i")
    end every 2

    loss = get_stats(nn_model, loaderXtrain, loaderYtrain, hparams)

    bar(get_window_of_Aₖ(Normal(0.0f0, 1.0f0), nn_model, collect(loaderXtrain)[1], 2))

    Flux.reset!(nn_model)
    for data in collect(loaderXtrain)[2]
        nn_model.([[data]])
    end

    prediction = Vector{Float32}()
    for data in collect(loaderXtest)[2]
        y = nn_model.([[data]])
        append!(prediction, y[1])
    end

    plot(prediction; seriestype=:scatter)
    plot!(Float32.(collect(loaderXtest)[1]); seriestype=:scatter)

    ND(Float32.(collect(loaderXtest)[1])[1:200], prediction[1:200])

    RMSE(Float32.(collect(loaderXtest)[1])[1:200], prediction[1:200])

    y = collect(loaderYtest)[1]
    Flux.reset!(nn_model)
    nn_model.([collect(loaderXtest)[1]'])
    collect(loaderYtrain)[1]

    get_watson_durbin_test(y, ŷ)
end
