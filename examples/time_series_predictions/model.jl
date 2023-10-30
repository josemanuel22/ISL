using Flux
using Random
using Statistics

using AdaptativeBlockLearning
using Distributions
using DataFrames
using CSV
using Plots

include("../utils.jl")
include("ts_utils.jl")

@test_experiments "testing AutoRegressive Model 1" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.5f0, 0.1f0),
        train_ratio=0.8,
    )

    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    n_series = 100
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        n_series, ar_hparams
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=n_series, window_size=1000, K=10)
    loss = ts_invariant_statistical_loss_one_step_prediction(
        rec, gen, loaderXtrain, loaderYtrain, hparams
    )

    plot_univariate_ts_prediction(
        rec,
        gen,
        collect(loaderXtrain)[15],
        collect(loaderXtest)[15],
        hparams;
        n_average=1000,
    )
end

@test_experiments "testing AutoRegressive Model 2" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.0f0, 0.2f0),
    )

    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    n_series = 200
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        n_series, ar_hparams
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=n_series, window_size=1000, K=10)
    loss = ts_invariant_statistical_loss_one_step_prediction(
        rec, gen, loaderXtrain, loaderYtrain, hparams
    )

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=1000
    )
end

@test_experiments "testing AutoRegressive Model 3" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=4000,
        noise=Normal(0.0f0, 0.2f0),
        train_ratio=0.6,
    )

    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    n_series = 200
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        n_series, ar_hparams
    )

    loss = ts_invariant_statistical_loss_one_step_prediction(
        rec, gen, loaderXtrain, loaderYtrain, hparams
    )

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=1000
    )
end

@test_experiments "testing AutoRegressive Model 4" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.0f0, 0.5f0),
        train_ratio=0.8,
    )

    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    n_series = 500
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        n_series, ar_hparams
    )

    loss = ts_invariant_statistical_loss_one_step_prediction(
        rec, gen, loaderXtrain, loaderYtrain, hparams
    )

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[2], collect(loaderXtest)[2], hparams; n_average=100
    )
end

@test_experiments "testing AutoRegressive Model 5" begin
    ar_hparams = ARParams(;
        ϕ=[0.8f0, -0.4f0, -0.4f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.0f0, 0.2f0),
    )

    #nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 32, relu), Dense(32, 1, identity))

    n_series = 200
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        n_series, ar_hparams
    )

    loss = ts_invariant_statistical_loss_one_step_prediction(
        rec, gen, loaderXtrain, loaderYtrain, hparams
    )

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=100
    )
end

@test_experiments "testing AutoRegressive Model 6" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=4000,
        noise=Normal(0.0f0, 0.5f0),
        train_ratio=0.8,
    )

    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 32, relu), Dense(32, 1, identity))

    n_series = 100
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        n_series, ar_hparams
    )

    loss = ts_invariant_statistical_loss_one_step_prediction(
        rec, gen, loaderXtrain, loaderYtrain, hparams
    )

    plot_univariate_ts_prediction(
        rec,
        gen,
        collect(loaderXtrain)[1],
        collect(loaderXtest)[1],
        hparams;
        n_average=10000,
    )
end

@test_experiments "testing AutoRegressive Model 7" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=4000,
        noise=Normal(0.0f0, 0.5f0),
        train_ratio=0.6,
    )

    #nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 32, relu), Dense(32, 1, identity))

    n_series = 200
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        n_series, ar_hparams
    )

    loss = ts_invariant_statistical_loss_one_step_prediction(
        rec, gen, loaderXtrain, loaderYtrain, hparams
    )

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[3], collect(loaderXtest)[3], hparams; n_average=1000
    )
end

@test_experiments "testing LD2011_2014" begin
    csv_file_path = "/Users/jmfrutos/github/AdaptativeBlockLearning/examples/time_series_predictions/data/LD2011_2014.txt"

    df = CSV.File(
        csv_file_path;
        delim=';',
        header=true,
        decimal=',',
        types=Dict(
            #"MT_331" => Float32,
            #"MT_332" => Float32,
            "MT_333" => Float32,
            "MT_334" => Float32,
            "MT_335" => Float32,
            "MT_336" => Float32,
            "MT_338" => Float32,
        ),
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=10)

    rec = Chain(
        RNN(2 => 2, relu; init=Flux.randn32(MersenneTwister(1))),
        RNN(2 => 2, relu; init=Flux.randn32(MersenneTwister(1))),
    )
    gen = Chain(
        Dense(3, 8, identity; init=Flux.randn32(MersenneTwister(1))),
        Dense(8, 1, identity; init=Flux.randn32(MersenneTwister(1))),
    )

    start = 36000
    num_training_data = 1000
    loaderXtrain = Flux.DataLoader(
        [[df.MT_333[i], df.MT_334[i]] for i in start:(start + num_training_data)];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        [[df.MT_333[i], df.MT_334[i]] for i in (start + 1):(start + num_training_data + 1)];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test = 2000
    loaderXtest = Flux.DataLoader(
        [
            [df.MT_333[i], df.MT_334[i]] for
            i in (start + num_training_data - 1):(start + num_training_data + num_test)
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loss = ts_covariates_adaptative_block_learning(
        rec, gen, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
    )

    plot_multivariate_ts_prediction(
        rec,
        gen,
        collect(loaderXtrain)[1],
        collect(loaderXtest)[1],
        hparams;
        n_average=10000,
    )

    prediction = Vector{Float32}()
    Flux.reset!(rec)
    s = rec(X_train[1])
    i = 2
    ideal = X_train[2:(end - 950)]
    for data in ideal
        y, _ = average_prediction(gen, s, n_average)
        s = rec(vcat(y, X_train[i][2]))
        i += 1
        append!(prediction, y[1])
    end

    t = 1:(length(ideal))
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
    plot!(
        2:((length(prediction))),
        prediction;
        label="Prediction",
        linecolor=get(ColorSchemes.rainbow, 0.2),
        ribbon=std_prediction,
    )

    prediction = Vector{Float32}()
    std_prediction = Vector{Float32}()
    s = rec(X_test[1])
    i = 2
    for data in X_test[1:100]
        y, std = average_prediction(gen, s, n_average)
        s = rec(vcat(y[1], X_test[i][2]))
        append!(prediction, y[1])
        append!(std_prediction, std[1])
        i += 1
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
end

"""
Run experiments for testing electricity-f consumption time series forecasting (variant F).

This code block performs experiments to forecast electricity consumption time series data. It loads the data from a CSV file, preprocesses it, trains a recurrent neural network (RNN) model, and makes predictions. This variant specifically tests electricity consumption forecasting.

Parameters:
- `csv_file_path`: A string representing the file path to the CSV file containing electricity consumption data.
- `start`: The starting index in the time series data for training and testing.
- `num_training_data`: The number of data points used for training.
- `hparams`: An instance of HyperParamsTS with hyperparameters for training the model.
- `rec`: The recurrent neural network (RNN) model used for encoding.
- `gen`: The neural network model used for decoding.
- `predictions`: An array to store the forecasted values.
- `stds`: An array to store the standard deviations of forecasts.

The function performs the following steps:
1. Load the electricity consumption data from the CSV file and preprocess it.
2. Split the data into training and testing sets.
3. Train the model using training data and hyperparameters.
4. Make predictions for the test data.

Note: This code assumes a specific data structure for the CSV file, including column names like "MT_005," "MT_006," etc., and specific model architecture for `rec` and `gen`.

Example:
```julia
@test_experiments "testing electricity-f" begin
    csv_file_path = "/Users/jmfrutos/github/AdaptativeBlockLearning/examples/time_series_predictions/data/LD2011_2014.txt"
    start = 35040
    num_training_data = 1000
    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=20)
    rec = ...  # Define the RNN model
    gen = ...  # Define the neural network model
    predictions,
```
"""
@test_experiments "testing electricity-f" begin
    csv_file_path = "/Users/jmfrutos/github/AdaptativeBlockLearning/examples/time_series_predictions/data/LD2011_2014.txt"

    df = CSV.File(
        csv_file_path;
        delim=';',
        header=true,
        decimal=',',
        types=Dict(
            "MT_005" => Float32,
            "MT_006" => Float32,
            "MT_007" => Float32,
            "MT_008" => Float32,
            "MT_167" => Float32,
            "MT_168" => Float32,
            "MT_333" => Float32,
            "MT_334" => Float32,
            "MT_335" => Float32,
            "MT_336" => Float32,
            "MT_338" => Float32,
        ),
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=20)

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(
        RNN(1 => 3, elu; init=Flux.randn32(MersenneTwister(1))),
        RNN(3 => 3, elu; init=Flux.randn32(MersenneTwister(1))),
    )
    gen = Chain(
        Dense(4, 16, identity; init=Flux.randn32(MersenneTwister(1))),
        Dense(16, 1, identity; init=Flux.randn32(MersenneTwister(1))),
    )

    ts = df.MT_005

    start = 35040
    num_training_data = 1000
    loaderXtrain = Flux.DataLoader(
        ts[start:(start + num_training_data)];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        ts[(start + 1):(start + num_training_data + 1)];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test = 1000
    loaderXtest = Flux.DataLoader(
        ts[(start + num_training_data - 1):(start + num_training_data + num_test)];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    @showprogress for _ in 1:1000
        loss = ts_invariant_statistical_loss(rec, gen, loaderXtrain, loaderYtrain, hparams)
        append!(losses, loss)
    end

    τ = 24
    xtrain = collect(loaderXtrain)[1]
    xtest = collect(loaderXtest)[1]
    prediction, stds = ts_forecast(
        rec, gen, xtrain, xtest, τ; n_average=1000, noise_model=Normal(0.0f0, 1.0f0)
    )
end

"""
Run experiments for testing electricity-c consumption time series forecasting.

This code block performs experiments to forecast electricity consumption time series data. It loads the data from a CSV file, preprocesses it, trains a recurrent neural network (RNN) model, and makes predictions.

Parameters:
- `csv_file_path`: A string representing the file path to the CSV file containing electricity consumption data.
- `start`: The starting index in the time series data for training and testing.
- `num_training_data`: The number of data points used for training.
- `hparams`: An instance of HyperParamsTS with hyperparameters for training the model.
- `rec`: The recurrent neural network (RNN) model used for encoding.
- `gen`: The neural network model used for decoding.
- `predictions`: An array to store the forecasted values.
- `stds`: An array to store the standard deviations of forecasts.

The function performs the following steps:
1. Load the electricity consumption data from the CSV file.
2. Preprocess the data, including type conversion and aggregation.
3. Split the data into training and testing sets.
4. Train the model using training data and hyperparameters.
5. Make predictions for the test data.

Note: This code assumes a specific data structure for the CSV file, including column names like "MT_005," "MT_006," etc., and specific model architecture for `rec` and `gen`.

Example:
```julia
@test_experiments "testing electricity-c" begin
    csv_file_path = "/Users/jmfrutos/github/AdaptativeBlockLearning/examples/time_series_predictions/data/LD2011_2014.txt"
    start = 35040
    num_training_data = 35040
    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=10)
    rec = ...  # Define the RNN model
    gen = ...  # Define the neural network model
    predictions, stds = run_electricity_experiments(csv_file_path, start, num_training_data, hparams, rec, gen)
end
```
"""
@test_experiments "testing electricity-c" begin
    csv_file_path = "/Users/jmfrutos/github/AdaptativeBlockLearning/examples/time_series_predictions/data/LD2011_2014.txt"

    df = CSV.File(
        csv_file_path;
        delim=';',
        header=true,
        decimal=',',
        types=Dict(
            "MT_005" => Float32,
            "MT_006" => Float32,
            "MT_007" => Float32,
            "MT_008" => Float32,
            "MT_168" => Float32,
            #"MT_331" => Float32,
            #"MT_332" => Float32,
            "MT_333" => Float32,
            "MT_334" => Float32,
            "MT_335" => Float32,
            "MT_336" => Float32,
            "MT_338" => Float32,
        ),
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=10)

    rec = Chain(
        RNN(1 => 3, relu; init=Flux.randn32(MersenneTwister(1))),
        RNN(3 => 3, relu; init=Flux.randn32(MersenneTwister(1))),
    )
    gen = Chain(
        Dense(4, 10, identity; init=Flux.randn32(MersenneTwister(1))),
        Dense(10, 1, identity; init=Flux.randn32(MersenneTwister(1))),
    )

    # start and finish of the training data
    start = 35040
    num_training_data = 1000

    ts = df.MT_006

    loaderXtrain = ts[start:(start + num_training_data)]
    loaderYtrain = ts[(start + 1):(start + num_training_data + 1)]
    loaderXtest = ts[(start + num_training_data - 1):length(ts)]

    # coarse grain the data, electricity-c
    aggregated_data_xtrain = Vector{Float32}()
    aggregated_data_ytrain = Vector{Float32}()
    aggregated_data_xtest = Vector{Float32}()

    for i in 1:4:length(loaderXtrain)
        push!(aggregated_data_xtrain, Float32(sum(loaderXtrain[i:min(i + 3, end)])))
    end

    for i in 1:4:length(loaderYtrain)
        push!(aggregated_data_ytrain, Float32(sum(loaderYtrain[i:min(i + 3, end)])))
    end

    for i in 1:4:length(loaderXtest)
        push!(aggregated_data_xtest, Float32(sum(loaderXtest[i:min(i + 3, end)])))
    end

    loader_xtrain = Flux.DataLoader(
        aggregated_data_xtrain;
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )
    loader_ytrain = Flux.DataLoader(
        aggregated_data_ytrain;
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loader_xtest = Flux.DataLoader(
        aggregated_data_xtest;
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    #train the model
    losses = []
    @showprogress for _ in 1:200
        loss = ts_invariant_statistical_loss(
            rec, gen, loader_xtrain, loader_ytrain, hparams
        )
        append!(losses, loss)
    end

    predictions, stds = ts_forecast(
        rec, gen, collect(loader_xtrain)[1], collect(loader_xtest)[1], 24, 1000
    )
end

@test_experiments "Mixture time series syntetic 1" begin

    # Define a function to generate synthetic data.
    """
    generate_synthetic(range)

    Generate synthetic time series data according to a composite time series model.

    This function generates synthetic time series data based on a composite time series model. The model consists of two underlying functions, y1(t) and y2(t), each of which is selected with equal probability at each time step from a Bernoulli distribution with a parameter of 1/2. The functions y1(t) and y2(t) are defined as follows:
        - y1(t) = 10 * cos(t - 0.5) + ξ, where ξ follows a standard Gaussian distribution (N(0, 1)).
        - y2(t) = 10 * sin(t - 0.5) + ξ, where ξ follows a standard Gaussian distribution (N(0, 1)).

    Parameters:
        - `range`: A range of time values over which the synthetic time series data will be generated.

    Returns:
        - An array of time-value pairs representing the synthetic time series data.

    Example:
    ```julia
        range = -4:0.1:4
        data = generate_synthetic(range)
    ```
    """
    function generate_syntetic(range)
        data = []
        for x in range
            ϵ = Float32(rand(Normal(0.0f0, 1.0f0)))
            # Randomly select one of two functions based on a Bernoulli distribution with parameter 1/2.
            if rand(Bernoulli(0.5))
                y = 10 * cos(x - 0.5) + ϵ # Function y1(t) = 10 cos(t − 0.5) + ξ
            else
                y = 10 * sin(x - 0.5) + ϵ # Function y2(t) = 10 sin(t − 0.5) + ξ
            end
            push!(data, [x, y])
        end
        return data
    end

    #generating train data
    range = -4:0.01:4
    data = generate_syntetic(range)

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=20, window_size=800, K=10)

    rec = Chain(RNN(1 => 2, relu), RNN(2 => 2, relu))
    gen = Chain(Dense(3, 16, relu), Dense(16, 1, identity))

    loaderXtrain = Flux.DataLoader(
        [Float32(y[2]) for y in data[1:(end - 1)]];
        batchsize=round(Int, length(data)),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        [Float32(y[2]) for y in data[2:end]];
        batchsize=round(Int, length(data)),
        shuffle=false,
        partial=false,
    )

    losses = []
    @showprogress for i in 1:2000
        loss = ts_invariant_statistical_loss(rec, gen, loaderXtrain, loaderYtrain, hparams)
        append!(losses, loss)
    end

    #generating test data
    range = -4:0.02:4
    data = generate_syntetic(range)
    X_test = [Float32(y[2]) for y in data[1:(end - 1)]]
    Y_test = [Float32(y[2]) for y in data[2:end]]

    τ = 20
    prediction, _ = ts_forecast(rec, gen, X_test, τ; n_average=1000)

    #plot the results
    scatter(
        X_test;
        label="ideal",
        color=get(ColorSchemes.rainbow, 0.2),
        legend=:topright,
        marker=:circle,
        markersize=3,
    )
    scatter!(prediction; label="prediction", color=:redsblues, marker=:circle, markersize=3)
    xlabel!("t")
    ylabel!("y")
end
