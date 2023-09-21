using LinearAlgebra
using ToeplitzMatrices

#
# AutoRegressive Process Utils
#

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

# Creates training and testing samples according to hyperparameters
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

## Utils to measure time series performance

ND(xₜ, x̂ₜ) = sum(abs.(xₜ .- x̂ₜ)) / sum(abs.(xₜ))

function RMSE(xₜ, x̂ₜ)
    return sqrt((1 / length(xₜ)^2) * sum((xₜ .- x̂ₜ) .^ 2)) /
           ((1 / length(xₜ)^2) * sum(abs.(xₜ)))
end

function QLρ(xₜ, x̂ₜ; ρ=0.5)
    return 2 *
           (sum(abs.(xₜ))^-1) *
           sum(ρ .* (xₜ .- x̂ₜ) .* (xₜ .> x̂ₜ) .+ (1 - ρ) .* (x̂ₜ .- xₜ) .* (xₜ .<= x̂ₜ))
end

function get_watson_durbin_test(y, ŷ)
    e = []
    for (yₜ, ŷₜ) in zip(y, ŷ)
        append!(e, yₜ - ŷₜ)
    end
    sum = 0
    for i in 2:2:length(e)
        sum += (e[i] - e[i - 1])^2
    end
    return sum / sum(e .^ 2)
end

function yule_walker(
    x::Vector{Real};
    order::Int64=1,
    method="adjusted",
    df::Union{Nothing,Int64}=nothing,
    inv=false,
    demean=true,
)
    method in ("adjusted", "mle") ||
        throw(ArgumentError("ACF estimation method must be 'adjusted' or 'MLE'"))

    x = copy(x)
    if demean
        x .-= mean(x)
    end
    n = isnothing(df) ? length(x) : df

    adj_needed = method == "adjusted"

    if ndims(x) > 1 || size(x, 2) != 1
        throw(ArgumentError("Expecting a vector to estimate AR parameters"))
    end

    r = zeros(Float64, order + 1)
    r[1] = sum(x .^ 2) / n
    for k in 1:order
        r[k + 1] = sum(x[1:(end - k)] .* x[(k + 1):end]) / (n - k * adj_needed)
    end
    R = Toeplitz(r[1:(end - 1)], conj(r[1:(end - 1)]))

    rho = 0
    try
        rho = R \ r[2:end]
    catch err
        if occursin("Singular matrix", string(err))
            @warn "Matrix is singular. Using pinv."
            rho = pinv(R) * r[2:end]
        else
            throw(err)
        end
    end

    sigmasq = r[1] - dot(r[2:end], rho)
    sigma = isnan(sigmasq) || sigmasq <= 0 ? NaN : sqrt(sigmasq)

    if inv
        return rho, sigma, inv(R)
    else
        return rho, sigma
    end
end

function plot_univariate_ts_prediction(nn_model, X_train, X_test, hparams)
    prediction = Vector{Float32}()
    Flux.reset!(nn_model)
    for data in X_train
        y = nn_model([data])
        append!(prediction, y[1])
    end

    ideal = vcat(X_train, X_test)
    t = 1:length(ideal)
    plot(
        t,
        ideal;
        xlabel="t",
        ylabel="y",
        label="Ideal",
        linecolor=:redsblues,
        plot_titlefontsize=12,
        fmt=:png,
    )

    prediction = Vector{Float32}()
    for data in X_test
        y = nn_model([data])
        append!(prediction, y[1])
    end

    t = (length(X_train):(length(X_train) + length(prediction))-1)
    plot!(t, prediction; label="Prediction", linecolor=get(ColorSchemes.rainbow, 0.2))

    return vline!([length(X_train)]; line=(:dash, :black))
end

function plot_multivariate_ts_prediction(nn_model, X_train, X_test, hparams)
    Flux.reset!(nn_model)
    for data in X_train
        nn_model([data])
    end

    prediction = Vector{Float32}()
    for data in X_test
        y = nn_model([data])
        append!(prediction, y[1])
    end

    plot(
        [x[1] for x in X_test];
        xlabel="t",
        ylabel="y",
        label="Ideal",
        linecolor=:redsblues,
        plot_titlefontsize=12,
        fmt=:png,
    )

    return plot!(prediction; label="Prediction", linecolor=get(ColorSchemes.rainbow, 0.2))
end

function plot_ts(nn_model, Xₜ, Xₜ₊₁, hparams)
    Flux.reset!(nn_model)
    nn_model([Xₜ.data[1]])
    plot(Xₜ.data[1:15000]; seriestype=:scatter)
    return plot!(vec(nn_model.([xX.data[1:15000]]')...); seriestype=:scatter)
end
