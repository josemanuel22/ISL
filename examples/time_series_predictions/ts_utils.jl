using LinearAlgebra
using ToeplitzMatrices

include("../utils.jl")

#
# AutoRegressive Process Utils
#

"""
    struct ARParams

A mutable struct to hold parameters for generating AutoRegressive (AR) processes and datasets.

# Fields
- `ϕ::Vector{Float32} = [0.4f0, 0.3f0, 0.2f0]`: An array of AR coefficients. By default, it represents an AR(3) process.
- `proclen::Int = 10000`: The length of the AR process to be generated.
- `x₁::Float32 = 0.0f0`: The initial value of the AR process.
- `noise = Normal(0.0f0, 1.0f0)`: The noise distribution to add to the generated data. It is a Normal distribution
  with mean 0 and standard deviation 1.
- `seqshift::Int = 1`: The shift between sequences, which may be used in other parts of the code (see `utils.jl`).
- `train_ratio::Float64 = 0.8`: The percentage of data to be included in the training set. By default, it is set to 80%.

"""
Base.@kwdef mutable struct ARParams
    seed::Int = 1234                            # Seed for reproducibility
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

"""
    generate_batch_train_test_data(n_series, arparams)

Generate batches of training and testing data for multiple AR(p) processes. Each batch is of the DataLoader returned is a realization of the AR(p) process.

# Arguments
- `n_series::Int`: Number of AR(p) to generate and create training/testing data for.

- `arparams::NamedTuple`: A structure containing various parameters and settings for data generation. The structure should include the following fields:
    - `seed::Int`: Seed for random number generation.
    - `train_ratio::Float64`: The ratio of data to be used for training (between 0 and 1).
    - `proclen::Int`: Length of each time series.

# Returns
- `loaderXtrain::Flux.DataLoader`: DataLoader for the training input data (Xtrain).

- `loaderYtrain::Flux.DataLoader`: DataLoader for the training target data (Ytrain).

- `loaderXtest::Flux.DataLoader`: DataLoader for the testing input data (Xtest).

- `loaderYtest::Flux.DataLoader`: DataLoader for the testing target data (Ytest).
"""
function generate_batch_train_test_data(n_series, arparams)
    Random.seed!(arparams.seed)
    # Get data
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    for _ in 1:(n_series)
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

"""
    ND(xₜ, x̂ₜ)

Calculate the Normalized Deviation (ND) between true and predicted values.

# Arguments
- `xₜ::Vector{Float64}`: The true values (actual observations).
- `x̂ₜ::Vector{Float64}`: The predicted values.

# Returns
- `nd::Float64`: The Normalized Deviation (ND) between xₜ and x̂ₜ.

# Details
The Normalized Deviation (ND) is a metric that quantifies the relative deviation between true values (xₜ) and
predicted values (x̂ₜ). It is calculated as the sum of absolute differences between xₜ and x̂ₜ, normalized by
the sum of absolute values of xₜ.

The function calculates ND as follows:
- Computes the absolute differences between xₜ and x̂ₜ.
- Calculates the sum of absolute differences.
- Normalizes the sum by dividing it by the sum of absolute values of xₜ.

ND provides a measure of the relative error between the true and predicted values, taking into account the scale of
the true values.

Example:
```julia
true_values = [1.2, 2.3, 3.4, 4.5]
predicted_values = [1.0, 2.2, 3.3, 4.7]
nd = ND(true_values, predicted_values)
"""
ND(xₜ, x̂ₜ) = sum(abs.(xₜ .- x̂ₜ)) / sum(abs.(xₜ))

"""
    RMSE(xₜ, x̂ₜ)

Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

# Arguments
- `xₜ::Vector{Float64}`: The true values (actual observations).
- `x̂ₜ::Vector{Float64}`: The predicted values.

# Returns
- `rmse::Float64`: The Root Mean Squared Error (RMSE) between xₜ and x̂ₜ.

# Details
The Root Mean Squared Error (RMSE) is a widely used metric for evaluating the accuracy of predictive models.
It measures the square root of the mean of the squared differences between true values (xₜ) and predicted values (x̂ₜ).

The function calculates RMSE as follows:
- Computes the squared differences between xₜ and x̂ₜ.
- Calculates the mean of the squared differences.
- Takes the square root of the mean to obtain the RMSE.

Example:
```julia
true_values = [1.2, 2.3, 3.4, 4.5]
predicted_values = [1.0, 2.2, 3.3, 4.7]
rmse = RMSE(true_values, predicted_values)
"""
function RMSE(xₜ, x̂ₜ)
    return sqrt((1 / length(xₜ)^2) * sum((xₜ .- x̂ₜ) .^ 2)) /
           ((1 / length(xₜ)^2) * sum(abs.(xₜ)))
end

"""
    QLρ(xₜ, x̂ₜ; ρ=0.5)

Calculate the Quantile Loss function with a quantile level ρ.

# Arguments
- `xₜ::Vector{Float64}`: The true values (actual observations).
- `x̂ₜ::Vector{Float64}`: The predicted values.
- `ρ::Float64=0.5`: The quantile level (default: 0.5, which corresponds to the median).

# Returns
- `loss::Float64`: The Quantile Loss at the specified quantile level ρ.

# Details
The Quantile Loss function measures the discrepancy between true values (xₜ) and predicted values (x̂ₜ)
at a specified quantile level ρ. It is a robust loss function that is less sensitive to outliers
compared to the mean squared error (MSE) loss.

The function calculates the Quantile Loss as follows:
- For each observation, it computes the absolute difference between the true and predicted values.
- It then applies a weighted absolute difference, giving more weight to observations where xₜ > x̂ₜ for ρ,
  and less weight when xₜ ≤ x̂ₜ for (1 - ρ).
- Finally, it scales the loss by 2 divided by the sum of the absolute values of the true observations.

Example:
```julia
true_values = [1.2, 2.3, 3.4, 4.5]
predicted_values = [1.0, 2.2, 3.3, 4.7]
quantile_level = 0.75
loss = QLρ(true_values, predicted_values, ρ=quantile_level)
```
"""
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


"""
    yule_walker(x::Vector{Float32};
                order::Int64=1,
                method="adjusted",
                df::Union{Nothing,Int64}=nothing,
                inv=false,
                demean=true,
    )

Estimate AutoRegressive (AR) parameters using the Yule-Walker equations.

# Arguments
- `x::Vector{Float32}`: The input time series data.
- `order::Int64=1`: The order of the AR model to estimate (default: 1).
- `method::String="adjusted"`: The method for estimating the autocorrelation function (ACF).
  - "adjusted" (default): Use adjusted ACF estimation.
  - "mle": Use maximum likelihood estimation.
- `df::Union{Nothing,Int64}=nothing`: The degrees of freedom for ACF estimation (default: nothing).
- `inv::Bool=false`: If true, return the inverse of the Toeplitz matrix in addition to AR parameters (default: false).
- `demean::Bool=true`: If true, demean the input data by subtracting its mean (default: true).

# Returns
- If `inv` is false (default), returns a tuple containing:
  - `rho::Vector{Float64}`: The estimated AR coefficients.
  - `sigma::Float64`: The estimated standard deviation.
- If `inv` is true, returns a tuple containing:
  - `rho::Vector{Float64}`: The estimated AR coefficients.
  - `sigma::Float64`: The estimated standard deviation.
  - `R_inv::Matrix{Float64}`: The inverse of the Toeplitz matrix used in estimation.

# Details
This function estimates AR parameters using the Yule-Walker equations. It supports different ACF estimation methods
(adjusted or maximum likelihood) and can optionally return the inverse of the Toeplitz matrix.

Example:
```julia
data = [0.1, 0.2, 0.3, 0.4, 0.5]
order = 2
rho, sigma = yule_walker(data, order=order, method="mle")
```
"""
function yule_walker(
    x::Vector{Float32};
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

function ts_forecasting(rec, gen, ts, t₀, τ, n_average)
    prediction = Vector{Float32}()
    stdevss = Vector{Float32}()
    Flux.reset!(rec)
    s = 0
    for data in ts[1:t₀]
        s = rec([data])
    end

    for data in ts[t₀:(t₀ + τ - 1)]
        y, stdev = average_prediction(gen, s, n_average)
        s = rec([y[1]])
        append!(prediction, y[1])
        append!(stdevss, stdev)
    end
    return prediction, stdevss
end

function average_prediction(gen, s, n_average)
    elements = Float32[]
    for _ in 1:n_average
        ϵ = rand(Normal(0.0f0, 1.0f0))
        y = gen([ϵ, s...])
        push!(elements, y[1])
    end
    return mean(elements), std(elements)
end

function plot_univariate_ts_prediction(rec, gen, X_train, X_test, hparams; n_average=1000)
    prediction = Vector{Float32}()
    Flux.reset!(rec)
    s = 0
    for data in X_train
        s = rec([data])
        y, _ = average_prediction(gen, s, n_average)
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
    std_prediction = Vector{Float32}()
    for data in X_test
        y, std = average_prediction(gen, s, n_average)
        s = rec([data])
        append!(prediction, y[1])
        append!(std_prediction, std)
    end

    t = (length(X_train):((length(X_train) + length(prediction)) - 1))
    plot!(
        t,
        prediction;
        label="Prediction",
        linecolor=get(ColorSchemes.rainbow, 0.2),
        ribbon=std_prediction,
    )

    nd = ND(X_test, prediction)
    rmse = RMSE(X_test, prediction)
    qlρ = QLρ(X_test, prediction; ρ=0.9)

    return vline!(
        [length(X_train)];
        line=(:dash, :black),
        label="",
        plot_title="ND: " *
                   format_numbers(nd) *
                   " "^4 *
                   "RMSE: " *
                   format_numbers(rmse) *
                   " "^4 *
                   "QLₚ₌₀.₉: " *
                   format_numbers(qlρ),
    )
end

function plot_univariate_ts_forecasting(rec, gen, X_train, X_test, hparams; n_average=1000)
    prediction = Vector{Float32}()
    Flux.reset!(rec)
    s = 0
    for data in X_train
        s = rec([data])
        y, _ = average_prediction(gen, s, n_average)
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
    std_prediction = Vector{Float32}()
    for data in X_test
        y, std = average_prediction(gen, s, n_average)
        s = rec([y[1]])
        append!(prediction, y[1])
        append!(std_prediction, std)
    end

    t = (length(X_train):((length(X_train) + length(prediction)) - 1))
    plot!(
        t,
        prediction;
        label="Prediction",
        linecolor=get(ColorSchemes.rainbow, 0.2),
        ribbon=std_prediction,
    )

    nd = ND(X_test, prediction)
    rmse = RMSE(X_test, prediction)
    qlρ = QLρ(X_test, prediction; ρ=0.9)

    return vline!(
        [length(X_train)];
        line=(:dash, :black),
        label="",
        plot_title="ND: " *
                   format_numbers(nd) *
                   " "^4 *
                   "RMSE: " *
                   format_numbers(rmse) *
                   " "^4 *
                   "QLₚ₌₀.₉: " *
                   format_numbers(qlρ),
    )
end

function plot_multivariate_ts_prediction(rec, gen, X_train, X_test, hparams; n_average=1000)
    prediction = Vector{Float32}()
    Flux.reset!(rec)
    for data in X_train
        s = rec(data)
        y, _ = average_prediction(gen, s, n_average)
        append!(prediction, y[1])
    end

    ideal = vcat(X_train, X_test)
    t = 1:length(ideal)
    plot(
        t,
        [x[1] for x in ideal];
        xlabel="t",
        ylabel="y",
        label="Ideal",
        linecolor=:redsblues,
        plot_titlefontsize=12,
        fmt=:png,
    )

    #=     t = 1:length(X_train)
        plot!(
            t,
            prediction;
            xlabel="t",
            ylabel="y",
            label="Prediction",
            linecolor=:darkblue,
            plot_titlefontsize=12,
            fmt=:png,
        ) =#

    prediction = Vector{Float32}()
    std_prediction = Vector{Float32}()
    for data in X_test
        s = rec(data)
        y, std = average_prediction(gen, s, n_average)
        append!(prediction, y[1])
        append!(std_prediction, std[1])
    end

    t = (length(X_train):((length(X_train) + length(prediction)) - 1))
    plot!(
        t,
        prediction;
        label="Prediction",
        linecolor=get(ColorSchemes.rainbow, 0.2),
        ribbon=std_prediction,
    )

    X₁_test = [x[1] for x in X_test]
    nd = ND(X₁_test, prediction)
    rmse = RMSE(X₁_test, prediction)

    return vline!(
        [length(X_train)];
        line=(:dash, :black),
        label="",
        plot_title="ND: " * format_numbers(nd) * " "^4 * "RMSE: " * format_numbers(rmse),
    )
end

"""
    save_model_ts(rec, gen, hparams, ar_hparams)

Save model training results and parameters to a BSON file with an incremental filename.

# Arguments
- `rec`: A trained reconstruction model.
- `gen`: A trained generation model.
- `hparams`: Hyperparameters for the training.
- `ar_hparams`: Hyperparameters for the AutoRegressive (AR) process used in the data generation.

# Details
This function saves the trained reconstruction and generation models, along with the hyperparameters used in training
and AR process configuration, to a BSON file with an incremental filename. The incremental filename is generated to avoid
overwriting existing files.

The generated filename is constructed based on various parameters, including the AR process parameters, process length, noise
distribution, training ratio, training epochs, learning rate, window size, and number of components (K).

Example:
```julia
rec_model, gen_model, hparams, ar_hparams = train_models()
save_model_ts(rec_model, gen_model, hparams, ar_hparams)
```
"""
function save_model_ts(rec, gen, hparams, ar_hparams)
    function get_incremental_filename(base_name)
        i = 1
        while true
            new_filename = base_name * "-$i.bson"
            if !isfile(new_filename)
                return i
            end
            i += 1
        end
    end

    epochs = hparams.epochs
    lr = hparams.η
    window_size = hparams.window_size
    K = hparams.K
    proclen = ar_hparams.proclen
    ar_params = ar_hparams.ϕ
    noise = ar_hparams.noise
    train_ratio = ar_hparams.train_ratio

    base_name = "$(ar_params)_$(proclen)_$(noise)_$(train_ratio)_$(epochs)_$(lr)_$(window_size)_$(K)"
    i = get_incremental_filename(base_name)
    new_filename = base_name * "-$i.bson"
    @info "name file: " * new_filename
    @save new_filename rec gen hparams ar_hparams
end
