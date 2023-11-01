using Flux
using Random
using Statistics

using ISL
using Distributions
using DataFrames
using CSV
using Plots

include("../utils.jl")
include("ts_utils.jl")

@test_experiments "testing AutoRegressive Model 1" begin
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
    csv_file_path = "examples/time_series_predictions/data/LD2011_2014.txt"
    start = 35040
    num_training_data = 1000
    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=20)
    rec = ...  # Define the RNN model
    gen = ...  # Define the neural network model
    predictions,
```
"""
@test_experiments "testing electricity-f" begin
    csv_file_path = "examples/time_series_predictions/data/LD2011_2014.txt"

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
    csv_file_path = "/examples/time_series_predictions/data/LD2011_2014.txt"
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
    csv_file_path = "examples/time_series_predictions/data/LD2011_2014.txt"

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

@test_experiments "testing 30 years european wind generation" begin
    #=
    https://www.nrel.gov/grid/solar-power-data.html
    =#

    csv1 = "examples/time_series_predictions/data/30-years-of-european-wind-generation/TS.CF.N2.30yr.csv"

    df1 = CSV.File(csv1; delim=',', header=true, decimal='.', types=Dict("AT11" => Float32))

    hparams = HyperParamsTS(; seed=1234, η=1e-6, epochs=200, window_size=200, K=20)

    rec = Chain(
        RNN(1 => 32, relu;),
        RNN(32 => 32, relu;),
        RNN(32 => 32, relu;),
        RNN(32 => 32, relu;),
    )
    gen = Chain(Dense(33, 64, relu;), Dense(64, 1, identity;))

    start = 1
    num_training_data = 1000

    ts = df1.FR63
    xtrain = ts[start:(start + num_training_data)]
    ytrain = ts[(start + 1):(start + num_training_data)]

    start_test = 1000
    num_test_data = 1000
    ztrain = ts[(start_test + start + num_training_data):(start_test + start + num_training_data + num_test_data)]

    loaderXtrain = Flux.DataLoader(
        map(x -> Float32.(x), xtrain);
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        map(x -> Float32.(x), ytrain);
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test_data = 10000
    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    @showprogress for _ in 1:1000
        loss = ts_invariant_statistical_loss(rec, gen, loaderXtrain, loaderYtrain, hparams)
        append!(losses, loss)
    end

    τ = 24
    prediction, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], 24; n_average=1000
    )

    #plot results
    ideal_target = Vector{Float32}()
    append!(ideal_target, collect(loaderXtrain)[1][(end - 12):end])
    append!(ideal_target, ideal[1:180])
    t₁ = (1:1:length(ideal_target))
    plot(t₁, ideal_target; label="Ideal Target", linecolor=:redsblues)
    t₂ = (14:1:(13 + length(prediction[1:180])))
    plot!(
        t₂,
        prediction[1:length(t₂)];
        ribbon=stdss[1:length(t₂)],
        fillalpha=0.1,
        label="Prediction",
        color=get(ColorSchemes.rainbow, 0.2),
    )
    xlabel!("t")
    ylabel!("% power plant's maximum output")
    vline!(
        [length(collect(loaderXtrain)[1][(end - 13):end])]; line=(:dash, :black), label=""
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

@test_experiments "Exchange TS" begin
    csv1 = "/Users/jmfrutos/Desktop/EXANCHE_RATE/exchange_rate.txt"

    column_names = [:col1, :col2, :col3, :col4, :col5, :col6, :col7, :col8]

    df1 = CSV.File(csv1; delim=',', header=column_names, decimal='.')

    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=1000, K=5)

    rec = Chain(
        RNN(1 => 32, elu; init=Flux.randn32(MersenneTwister(1))),
        RNN(32 => 32, elu; init=Flux.randn32(MersenneTwister(1))),
        RNN(32 => 32, elu; init=Flux.randn32(MersenneTwister(1))),
    )
    gen = Chain(
        Dense(33, 64, elu; init=Flux.randn32(MersenneTwister(1))),
        Dense(64, 1, identity; init=Flux.randn32(MersenneTwister(1))),
    )

    ts = df1.col1
    start = 1
    num_training_data = 1000
    loaderXtrain = Flux.DataLoader(
        Float32.(ts[start:num_training_data]);
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        Float32.(ts[(start + 1):(num_training_data + 1)]);
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test = 5000
    loaderXtest = Flux.DataLoader(
        Float32.(
            ts[(start + num_training_data - 1):(start + num_training_data + num_test)]
        );
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    @showprogress for _ in 1:1000
        loss = ts_invariant_statistical_loss(rec, gen, loaderXtrain, loaderYtrain, hparams)
        append!(losses, loss)
    end

    window_size = 100
    ma_result = moving_average([x[1] for x in losses], window_size)
    plot(ma_result)



    τ = 1000
    predictions, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=100
    )

end

@test_experiments "PEMS_train" begin
    """
    15 months worth of daily data (440 daily records) that describes the occupancy rate,
    between 0 and 1, of different car lanes of the San Francisco bay area freeways across time.

    We consider each day in this database as a single time series of dimension 963
    (the number of sensors which functioned consistently throughout the studied period) and
    length 6 x 24=144.
    """
    csv_file_path = "examples/time_series_predictions/data/pems+sf/PEMS_train"

    # Read the lines of the file
    n_days = 10
    values = []
    for _ in 1:n_days
        first_line = readline(csv_file_path)
        values_str = replace(first_line, "[" => "", "]" => "", ";" => " ")
        values_array = split(values_str, ' ')
        append!(values, map(x -> parse(Float32, x), values_array))
    end
    valuesX = values[1:(end - 1)]
    valuesY = values[2:end]

    sensor = 1
    total_sensors = 963
    result_vector = values[sensor:total_sensors:end]

    valuesX = result_vector[1:(end - 1)]
    valuesY = result_vector[2:end]

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=5)

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(
        RNN(1 => 16, relu; init=Flux.randn32(MersenneTwister(1))),
        RNN(16 => 32, relu; init=Flux.randn32(MersenneTwister(1))),
        #RNN(16 => 32, elu; init=Flux.randn32(MersenneTwister(1))),
        RNN(32 => 32, relu; init=Flux.randn32(MersenneTwister(1))),
    )
    gen = Chain(
        Dense(33, 64, elu; init=Flux.randn32(MersenneTwister(1))),
        Dense(64, 64, elu; init=Flux.randn32(MersenneTwister(1))),
        Dense(64, 1, sigmoid; init=Flux.randn32(MersenneTwister(1))),
    )

    start = 1
    num_training_data = 57780
    loaderXtrain = Flux.DataLoader(
        Float32.(valuesX[start:(start + num_training_data)]);
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        Float32.(valuesY[(start + 1):(start + num_training_data + 1)]);
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    #num_test = 138672
    num_test = 80000
    loaderXtest = Flux.DataLoader(
        Float32.(
            valuesX[(start + num_training_data - 1):(start + num_training_data + num_test)]
        );
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    @showprogress for _ in 1:10
        loss = ts_invariant_statistical_loss(rec, gen, loaderXtrain, loaderYtrain, hparams)
        append!(losses, loss)
    end

    window_size = 50
    ma_result = moving_average([x[1] for x in losses], window_size)
    plot(ma_result)

    τ = 24
    predictions, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=100
    )

    #s = rec(collect(loaderXtrain)[1][1])
    #ideal = collect(loaderx)[1]

    hparams.window_size = 24
    Flux.reset!(rec)
    ideal = collect(loaderXtrain)[1]
    s = []
    for j in (0:(hparams.window_size):(length(ideal) - hparams.window_size))
        s = rec(ideal[(j + 1):(j + hparams.window_size)]')
    end

    n_average = 100
    prediction = Vector{Float32}()
    stdss = Vector{Float32}()
    #hparams.window_size = 24
    ideal = collect(loaderXtest)[1]
    for j in (0:(hparams.window_size):(length(ideal) - hparams.window_size))
        s = rec(ideal[(j + 1):(j + hparams.window_size)]')
        for i in 1:(hparams.window_size)
            xₖ = rand(hparams.noise_model, n_average)
            yₖ = hcat([gen(vcat(x, s[:, i])) for x in xₖ]...)
            y = mean(yₖ)
            #s = rec(ideal[j+1:(j + hparams.window_size)]')
            append!(prediction, y[1])
            append!(stdss, std(yₖ))
        end
    end
    QLρ([x[1] for x in ideal][1:800], prediction[1:800]; ρ=0.5)
end
