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

function generate_batch_train_test_data(hparams, arparams1, arparams2)
    Random.seed!(hparams.seed)
    # Get data
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    for _ in 1:(hparams.epochs)
        if rand(Bernoulli(0.5))
            xtrain, xtest, ytrain, ytest = generate_train_test_data(arparams1)
            append!(Xtrain, xtrain)
            append!(Ytrain, ytrain)
            append!(Xtest, xtest)
            append!(Ytest, ytest)
        else
            xtrain, xtest, ytrain, ytest = generate_train_test_data(arparams2)
            append!(Xtrain, xtrain)
            append!(Ytrain, ytrain)
            append!(Xtest, xtest)
            append!(Ytest, ytest)
        end
    end

    loaderXtrain = Flux.DataLoader(
        Xtrain;
        batchsize=round(Int, arparams1.train_ratio * arparams1.proclen),
        shuffle=false,
        partial=false,
    )
    loaderYtrain = Flux.DataLoader(
        Ytrain;
        batchsize=round(Int, arparams1.train_ratio * arparams1.proclen - 1),
        shuffle=false,
        partial=false,
    )
    loaderXtest = Flux.DataLoader(
        Xtest;
        batchsize=round(Int, (1 - arparams1.train_ratio) * arparams1.proclen),
        shuffle=false,
        partial=false,
    )
    loaderYtest = Flux.DataLoader(
        Ytest;
        batchsize=round(Int, (1 - arparams1.train_ratio) * arparams1.proclen - 1),
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

    function format_numbers(x)
        if abs(x) < 0.01
            formatted_x = @sprintf("%.2e", x)
        else
            formatted_x = @sprintf("%.4f", x)
        end
        return formatted_x
    end

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

    function format_numbers(x)
        if abs(x) < 0.01
            formatted_x = @sprintf("%.2e", x)
        else
            formatted_x = @sprintf("%.4f", x)
        end
        return formatted_x
    end

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

    function format_numbers(x)
        if abs(x) < 0.01
            formatted_x = @sprintf("%.2e", x)
        else
            formatted_x = @sprintf("%.4f", x)
        end
        return formatted_x
    end

    return vline!(
        [length(X_train)];
        line=(:dash, :black),
        label="",
        plot_title="ND: " * format_numbers(nd) * " "^4 * "RMSE: " * format_numbers(rmse),
    )
end

function syntetic_data(t, σ)
    A₁₁, A₁₂, A₂₁, A₂₂ = 2.0, 2.0, 2.0, 2.0

    #f₁₁ = rand(1:5)
    f₁₁ = 2.0
    #f₁₂ = rand(20:25)
    f₁₂ = 15.0
    #f₂₁ = rand(10:15)
    f₂₁ = 6.0
    #f₂₂ = rand(20:25)
    f₂₂ = 20.0
    f₁(t, ν) = A₁₁ * sin(2 * π * f₁₁ * t) + A₁₂ * sin(2 * π * f₁₂ * t) + ν
    f₂(t, ν) = A₂₁ * sin(2 * π * f₂₁ * t) + A₂₂ * sin(2 * π * f₂₂ * t) + ν

    if rand(Bernoulli(1 / 2))
        return f₁(t, rand(Normal(0.0f0, σ))) + f₂(t, rand(Normal(0.0f0, σ)))
    else
        return f₂(t, rand(Normal(0.0f0, σ))) + f₂(t, rand(Normal(0.0f0, σ)))
    end
end

function get_statistics(rec, gen, hparams, ar_hparams, n_average)
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    nds = []
    rmses = []
    @showprogress for i in 1:(99)
        X_train = collect(loaderXtrain)[i]
        X_test = collect(loaderXtest)[i]
        prediction = Vector{Float32}()
        Flux.reset!(rec)
        for data in X_train
            s = rec([data])
            y, _ = average_prediction(gen, s, n_average)
            append!(prediction, y[1])
        end
        for data in X_test
            s = rec([data])
            y, _ = average_prediction(gen, s, n_average)
            append!(prediction, y[1])
        end
        push!(nds, ND(vcat(X_train, X_test), prediction))
        push!(rmses, RMSE(vcat(X_train, X_test), prediction))
    end
    return mean(nds), mean(rmses)
end

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
