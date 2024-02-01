using Flux
using Random
using Statistics

using ISL
using Distributions
using DataFrames
using CSV
using Plots
using StatsBase
using RollingFunctions

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

    cols = [
        "MT_005",
        "MT_006",
        "MT_007",
        "MT_008",
        "MT_168",
        #"MT_331",
        #"MT_332",
        "MT_333",
        "MT_334",
        "MT_335",
        "MT_336",
        "MT_338",
    ]

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
    rec = Chain(LSTM(1 => 16), LayerNorm(16))
    gen = Chain(Dense(17, 32, elu), Dropout(0.05), Dense(32, 1, identity))

    start = 35040
    num_training_data = 1000
    num_test = 1000

    # coarse grain the data, electricity-c
    aggregated_data_xtrain = Vector{Float32}()
    aggregated_data_ytrain = Vector{Float32}()
    aggregated_data_xtest = Vector{Float32}()
    for name in select_names
        if name != "Column1"
            println(name)
            ts = getproperty(df, Symbol(name))

            loaderXtrain = ts[start:(start + num_training_data)]
            loaderYtrain = ts[(start + 1):(start + num_training_data + 1)]
            loaderXtest = ts[(start + num_training_data - 1):(start + +num_training_data + num_test_data)]

            append!(aggregated_data_xtrain, loaderXtrain)

            append!(aggregated_data_ytrain, loaderYtrain)

            append!(aggregated_data_xtest, loaderXtest)
        end
    end

    v_mean = mean(aggregated_data_xtrain)
    v_std = std(aggregated_data_xtrain)
    aggregated_data_xtrain = (aggregated_data_xtrain .- v_mean) ./ v_std

    v_mean = mean(aggregated_data_ytrain)
    v_std = std(aggregated_data_ytrain)
    aggregated_data_ytrain = (aggregated_data_ytrain .- v_mean) ./ v_std

    v_mean = mean(aggregated_data_xtest)
    v_std = std(aggregated_data_xtest)
    aggregated_data_xtest = (aggregated_data_xtest .- v_mean) ./ v_std

    loaderXtrain = Flux.DataLoader(
        aggregated_data_xtrain;
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        aggregated_data_ytrain;
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test = 1000
    loaderXtest = Flux.DataLoader(
        aggregated_data_xtest;
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    ql5 = []
    losses = []
    @showprogress for _ in 1:10
        loss, ql5_ = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest
        )
        #loss, ql5_ = ts_invariant_statistical_loss(rec, gen, loaderXtrain, loaderYtrain, hparams)
        append!(losses, loss)
        append!(ql5, ql5_)
    end

    mse = 0.0
    mae = 0.0
    count = 0
    for ts in 1:length(loader_xtrain)
        xtrain = collect(loader_xtrain)[ts]
        prediction = Vector{Float32}()
        stds = Vector{Float32}()
        Flux.reset!(rec)
        s = []
        for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
            s = rec([xtrain[j + 1]])
        end

        τ = 48
        xtest = collect(loader_xtest)[ts]
        noise_model = Normal(0.0f0, 1.0f0)
        n_average = 1000
        for j in (0:(τ):(length(xtest) - τ))
            #s = rec(xtest[(j + 1):(j + τ)]')
            #s = rec([xtest[j + 1]])
            for i in 1:(τ)
                xₖ = rand(noise_model, n_average)
                y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                ȳ = mean(y)
                σ = std(y)
                s = rec([ȳ])
                append!(prediction, ȳ)
                append!(stds, σ)
            end
        end
        count += 1
        ideal = collect(loader_xtest)[ts]
        #QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
        println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
        mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
    end
    mae / count
    mse / count

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

    rec = Chain(RNN(1 => 3, relu), RNN(3 => 3, relu))
    gen = Chain(Dense(4, 10, identity), Dense(10, 1, identity))

    # start and finish of the training data
    start = 35040
    num_training_data = 35040
    num_test_data = 1000

    # coarse grain the data, electricity-c
    aggregated_data_xtrain = Vector{Float32}()
    aggregated_data_ytrain = Vector{Float32}()
    aggregated_data_xtest = Vector{Float32}()

    df = DataFrame(df)

    col_indices = map(name -> findfirst(isequal(name), names(df)), cols)

    select_names = [name for name in names(df) if name in cols]

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
    @showprogress for _ in 1:1000
        loss = ts_invariant_statistical_loss(
            rec, gen, loader_xtrain, loader_ytrain, hparams
        )
        append!(losses, loss)
    end

    l = moving_average(losses, 200)
    plot(l)

    xtrain = collect(loader_xtrain)[1]
    prediction = Vector{Float32}()
    stds = Vector{Float32}()
    Flux.reset!(rec)
    s = []
    for j in (0:1:(length(xtrain) - 1))
        s = rec([xtrain[j + 1]])
    end

    τ = 720
    xtest = collect(loader_xtest)[1]
    noise_model = Normal(0.0f0, 1.0f0)
    n_average = 1000
    for j in (0:(τ):(length(xtest) - τ))
        #s = rec(xtest[(j + 1):(j + τ)]')
        for i in 1:(τ)
            xₖ = rand(noise_model, n_average)
            y = hcat([gen(vcat(x, s)) for x in xₖ]...)
            ȳ = mean(y)
            σ = std(y)
            s = rec([ȳ])
            append!(prediction, ȳ)
            append!(stds, σ)
        end
    end

    predictions, stds = ts_forecast(
        rec, gen, collect(loader_xtrain)[1], collect(loader_xtest)[1], 24; n_average=1000
    )
end

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

    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=1000, K=10)

    rec = Chain(RNN(1 => 3, relu), RNN(3 => 3, relu))
    gen = Chain(Dense(4, 10, identity), Dense(10, 1, identity))

    # start and finish of the training data
    start = 35040
    num_training_data = 35040

    # coarse grain the data, electricity-c
    aggregated_data_xtrain = Vector{Float32}()
    aggregated_data_ytrain = Vector{Float32}()
    aggregated_data_xtest = Vector{Float32}()

    df = DataFrame(df)

    for name in names(df)[1:100]
        if name != "Column1"
            println(name)
            ts = getproperty(df, Symbol(name))

            loaderXtrain = ts[start:(start + num_training_data)]
            loaderYtrain = ts[(start + 1):(start + num_training_data + 1)]
            loaderXtest = ts[(start + num_training_data - 1):length(ts)]

            for i in 1:4:length(loaderXtrain)
                push!(aggregated_data_xtrain, Float32(sum(loaderXtrain[i:min(i + 3, end)])))
            end

            for i in 1:4:length(loaderYtrain)
                push!(aggregated_data_ytrain, Float32(sum(loaderYtrain[i:min(i + 3, end)])))
            end

            for i in 1:4:length(loaderXtest)
                push!(aggregated_data_xtest, Float32(sum(loaderXtest[i:min(i + 3, end)])))
            end
        end
    end

    losses = []
    @showprogress for _ in 1:100
        loss = ts_invariant_statistical_loss(
            rec, gen, loader_xtrain, loader_ytrain, hparams
        )
        append!(losses, loss)
    end

    ts = df.MT_005

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
    @showprogress for _ in 1:100
        loss = ts_invariant_statistical_loss(
            rec, gen, loader_xtrain, loader_ytrain, hparams
        )
        append!(losses, loss)
    end

    l = moving_average(losses, 100)
    plot(l)

    xtrain = collect(loader_xtrain)[1]
    prediction = Vector{Float32}()
    stds = Vector{Float32}()
    Flux.reset!(rec)
    s = []
    for j in (0:1:(length(xtrain) - 1))
        s = rec([xtrain[j + 1]])
    end

    τ = 24
    xtest = collect(loader_xtest)[1]
    noise_model = Normal(0.0f0, 1.0f0)
    n_average = 1000
    for j in (0:(τ):(length(xtest) - τ))
        #s = rec(xtest[(j + 1):(j + τ)]')
        for i in 1:(τ)
            xₖ = rand(noise_model, n_average)
            y = hcat([gen(vcat(x, s)) for x in xₖ]...)
            ȳ = mean(y)
            σ = std(y)
            s = rec([ȳ])
            append!(prediction, ȳ)
            append!(stds, σ)
        end
    end

    predictions, stds = ts_forecast(
        rec,
        gen,
        collect(loader_xtrain)[1][1:100],
        collect(loader_xtrain)[1][100:200],
        24;
        n_average=1000,
    )
end

@test_experiments "testing electricity-c" begin
    csv_file_path = "examples/time_series_predictions/data/LD2011_2014.txt"

    cols = [
        "MT_005",
        "MT_006",
        "MT_007",
        "MT_008",
        "MT_168",
        #"MT_331",
        #"MT_332",
        "MT_333",
        "MT_334",
        "MT_335",
        "MT_336",
        "MT_338",
    ]

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

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=2000, K=10)

    rec = Chain(LSTM(1 => 16), LayerNorm(16))
    gen = Chain(Dense(17, 32, elu), Dropout(0.05), Dense(32, 1, identity))

    # start and finish of the training data
    start = 35040
    num_training_data = 35040
    num_test_data = 1000

    # coarse grain the data, electricity-c
    aggregated_data_xtrain = Vector{Float32}()
    aggregated_data_ytrain = Vector{Float32}()
    aggregated_data_xtest = Vector{Float32}()

    df = DataFrame(df)

    col_indices = map(name -> findfirst(isequal(name), names(df)), cols)

    select_names = [name for name in names(df) if name in cols]

    for name in select_names
        if name != "Column1"
            println(name)
            ts = getproperty(df, Symbol(name))

            loaderXtrain = ts[start:(start + num_training_data)]
            loaderYtrain = ts[(start + 1):(start + num_training_data + 1)]
            loaderXtest = ts[(start + num_training_data - 1):(start + +num_training_data + num_test_data)]

            for i in 1:4:length(loaderXtrain)
                push!(aggregated_data_xtrain, Float32(sum(loaderXtrain[i:min(i + 3, end)])))
            end

            for i in 1:4:length(loaderYtrain)
                push!(aggregated_data_ytrain, Float32(sum(loaderYtrain[i:min(i + 3, end)])))
            end

            for i in 1:4:length(loaderXtest)
                push!(aggregated_data_xtest, Float32(sum(loaderXtest[i:min(i + 3, end)])))
            end
        end
    end

    v_mean = mean(aggregated_data_xtrain)
    v_std = std(aggregated_data_xtrain)
    aggregated_data_xtrain = (aggregated_data_xtrain .- v_mean) ./ v_std

    v_mean = mean(aggregated_data_ytrain)
    v_std = std(aggregated_data_ytrain)
    aggregated_data_ytrain = (aggregated_data_ytrain .- v_mean) ./ v_std

    v_mean = mean(aggregated_data_xtest)
    v_std = std(aggregated_data_xtest)
    aggregated_data_xtest = (aggregated_data_xtest .- v_mean) ./ v_std

    loader_xtrain = Flux.DataLoader(
        aggregated_data_xtrain;
        batchsize=round(Int, num_training_data / 4),
        shuffle=false,
        partial=false,
    )
    loader_ytrain = Flux.DataLoader(
        aggregated_data_ytrain;
        batchsize=round(Int, num_training_data / 4),
        shuffle=false,
        partial=false,
    )

    loader_xtest = Flux.DataLoader(
        aggregated_data_xtest;
        batchsize=round(Int, num_test_data / 4),
        shuffle=false,
        partial=false,
    )

    losses = []
    ql5 = []
    @showprogress for _ in 1:40
        loss, ql5_ = ts_invariant_statistical_loss(
            rec, gen, loader_xtrain, loader_ytrain, hparams, loader_xtest
        )
        append!(losses, loss)
        append!(ql5, ql5_)
    end

    #train the model
    losses = []
    @showprogress for _ in 1:100
        loss = ts_invariant_statistical_loss(
            rec, gen, loader_xtrain, loader_ytrain, hparams
        )
        append!(losses, loss)
    end

    l = moving_average(losses, 100)
    plot(l)

    mse = 0.0
    mae = 0.0
    count = 0
    for ts in 1:length(loader_xtrain)
        xtrain = collect(loader_xtrain)[ts]
        prediction = Vector{Float32}()
        stds = Vector{Float32}()
        Flux.reset!(rec)
        s = []
        for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
            s = rec([xtrain[j + 1]])
        end

        τ = 48
        xtest = collect(loader_xtest)[ts]
        noise_model = Normal(0.0f0, 1.0f0)
        n_average = 1000
        for j in (0:(τ):(length(xtest) - τ))
            #s = rec(xtest[(j + 1):(j + τ)]')
            #s = rec([xtest[j + 1]])
            for i in 1:(τ)
                xₖ = rand(noise_model, n_average)
                y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                ȳ = mean(y)
                σ = std(y)
                s = rec([ȳ])
                append!(prediction, ȳ)
                append!(stds, σ)
            end
        end
        count += 1
        ideal = collect(loader_xtest)[ts]
        #QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
        println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
        mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
    end
    mae / count
    mse / count

    predictions, stds = ts_forecast(
        rec,
        gen,
        collect(loader_xtrain)[1][1:100],
        collect(loader_xtrain)[1][100:200],
        24;
        n_average=1000,
    )
end

@test_experiments "testing 30 years european wind generation" begin
    #=
    https://www.nrel.gov/grid/solar-power-data.html
    =#

    csv1 = "examples/time_series_predictions/data/30-years-of-european-wind-generation/TS.CF.N2.30yr.csv"

    df1 = CSV.File(csv1; delim=',', header=true, decimal='.', types=Dict("AT11" => Float32))

    hparams = HyperParamsTS(; seed=1234, η=1e-5, epochs=200, window_size=200, K=20)

    rec = Chain(
        RNN(1 => 32, relu;),
        RNN(32 => 64, relu;),
        RNN(64 => 32, relu;),
        RNN(32 => 32, relu;),
        RNN(32 => 32, relu;),
    )
    gen = Chain(Dense(33, 128, relu;), Dense(128, 1, identity;))

    ts = df1.FR22

    df = DataFrame(df1)

    xtrain = []
    ytrain = []
    ztrain = []
    start = 1
    num_training_data = 2000

    start_test = 1000
    num_test_data = 1000
    for name in names(df)
        println(name)
        if name != "Column1"
            ts = getproperty(df, Symbol(name))
            append!(xtrain, ts[start:(start + num_training_data)])
            append!(ytrain, ts[(start + 1):(start + num_training_data)])

            append!(
                ztrain,
                ts[(start_test + start + num_training_data):(start_test + start + num_training_data + num_test_data)],
            )
        end
    end

    xtrain = filtered_array = filter(!ismissing, xtrain)
    ytrain = filtered_array = filter(!ismissing, ytrain)
    ztrain = filtered_array = filter(!ismissing, ztrain)

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

    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    ql5 = []
    @showprogress for _ in 1:1
        loss, _ql5 = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest
        )
        append!(losses, loss)
        append!(ql5, _ql5)
    end

    l = moving_average(ql5, 200)
    plot(l)

    xtrain = collect(loaderXtrain)[1]
    prediction = Vector{Float32}()
    stds = Vector{Float32}()
    Flux.reset!(rec)
    s = []
    for j in ((length(xtrain) - 100):1:(length(xtrain) - 1))
        s = rec([xtrain[j + 1]])
    end

    τ = 30
    xtest = collect(loaderXtest)[1]
    noise_model = Normal(0.0f0, 1.0f0)
    n_average = 1000
    for j in (0:(τ):(length(xtest) - τ))
        #s = rec(xtest[(j + 1):(j + τ)]')
        #s = rec([xtest[j + 1]])
        for i in 1:(τ)
            xₖ = rand(noise_model, n_average)
            y = hcat([gen(vcat(x, s)) for x in xₖ]...)
            ȳ = mean(y)
            σ = std(y)
            s = rec([ȳ])
            append!(prediction, ȳ)
            append!(stds, σ)
        end
    end

    τ = 30
    prediction, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=1000
    )

    #plot results
    ideal = collect(loaderXtest)[1]
    ideal_target = Vector{Float32}()
    append!(ideal_target, collect(loaderXtrain)[2][(end - 12):end])
    append!(ideal_target, ideal[1:τ])
    t₁ = (1:1:length(ideal_target))
    plot(t₁, ideal_target; label="Ideal Target", linecolor=:redsblues)
    t₂ = (14:1:(13 + length(prediction[1:τ])))
    plot!(
        t₂,
        prediction[1:length(t₂)];
        #ribbon=stds[1:length(t₂)],
        fillalpha=0.1,
        label="Prediction",
        color=get(ColorSchemes.rainbow, 0.2),
    )
    xlabel!("t")
    ylabel!("% power plant's maximum output")
    vline!(
        [length(collect(loaderXtrain)[2][(end - 13):end])]; line=(:dash, :black), label=""
    )
    ideal = collect(loaderXtest)[1]
    QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
    MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
    MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
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

    hparams = HyperParamsTS(; seed=1234, η=1e-4, epochs=20, window_size=800, K=10)

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

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=10)

    rec = Chain(
        RNN(1 => 16, elu),
        LayerNorm(16),
        RNN(16 => 16, elu),
        LayerNorm(16),
        RNN(16 => 16, elu),
    )
    gen = Chain(Dense(17, 32, sigmoid), LayerNorm(32), Dense(32, 1, sigmoid))

    ts = df1.col1
    df = DataFrame(df1)

    xtrain = []
    ytrain = []
    ztrain = []
    start = 1
    num_training_data = 1000
    start_test = num_training_data
    num_test_data = 1000

    for name in names(df)
        println(name)
        if name != "Column1"
            ts = getproperty(df, Symbol(name))
            append!(xtrain, ts[start:(start + num_training_data)])
            append!(ytrain, ts[(start + 1):(start + num_training_data)])
            append!(
                ztrain,
                ts[(start + num_training_data):(start + num_training_data + num_test_data)],
            )
        end
    end

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

    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    ql5 = []
    @showprogress for _ in 1:100
        loss, _ql5 = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest
        )
        append!(losses, loss)
        append!(ql5, _ql5)
    end

    window_size = 20
    ma_result = moving_average([x[1] for x in ql5], window_size)
    plot(ma_result)

    τ = 30
    predictions, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=1000
    )

    mse = 0.0
    mae = 0.0
    count = 0
    for ts in 1:length(column_names)
        xtrain = collect(loaderXtrain)[ts]
        prediction = Vector{Float32}()
        stds = Vector{Float32}()
        Flux.reset!(rec)
        s = []
        for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
            s = rec([xtrain[j + 1]])
        end

        τ = 96
        xtest = collect(loaderXtest)[ts]
        noise_model = Normal(0.0f0, 1.0f0)
        n_average = 1000
        for j in (0:(τ):(length(xtest) - τ))
            #s = rec(xtest[(j + 1):(j + τ)]')
            #s = rec([xtest[j + 1]])
            for i in 1:(τ)
                xₖ = rand(noise_model, n_average)
                y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                ȳ = mean(y)
                σ = std(y)
                s = rec([ȳ])
                append!(prediction, ȳ)
                append!(stds, σ)
            end
        end
        count += 1
        ideal = collect(loaderXtest)[ts]
        QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
        println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
        mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
    end
    mae / count
    mse / count
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
        RNN(1 => 16, relu),
        RNN(16 => 32, relu),
        #RNN(16 => 32, elu),
        RNN(32 => 32, relu),
    )
    gen = Chain(Dense(33, 64, elu), Dense(64, 64, elu), Dense(64, 1, sigmoid))

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
    @showprogress for _ in 1:1
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

@test_experiments "PEMS_train 2" begin
    """
    15 months worth of daily data (440 daily records) that describes the occupancy rate,
    between 0 and 1, of different car lanes of the San Francisco bay area freeways across time.

    We consider each day in this database as a single time series of dimension 963
    (the number of sensors which functioned consistently throughout the studied period) and
    length 6 x 24=144.
    """
    csv_file_path = "/Users/jmfrutos/Desktop/data_/traffic.txt"

    df1 = CSV.File(csv_file_path; delim=',', header=false, decimal='.')

    df = DataFrame(df1)
    #df = select(df, Not(:date))
    #new_df = normalize_df(df, 5000, 7500)
    new_df = mapcols(zscore, df)

    xtrain = []
    ytrain = []
    ztrain = []
    start = 1
    num_training_data = 16000
    num_test_data = 100

    ma_df = DataFrame()
    window_size = 25
    for name in names(new_df)
        ts = getproperty(new_df, Symbol(name))
        rolled = rollmean(ts, window_size)
        ma_df[!, Symbol(name)] = rolled
    end

    for name in names(ma_df)[1:100]
        println(name)
        if name != "date"
            ts = getproperty(ma_df, Symbol(name))
            append!(xtrain, ts[start:(start + num_training_data)])
            append!(ytrain, ts[(start + 1):(start + num_training_data)])
            append!(
                ztrain,
                ts[(start + num_training_data):(start + num_training_data + num_test_data)],
            )
        end
    end

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

    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-4, epochs=2000, window_size=1000, K=10)

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(
        RNN(1 => 3, elu),
        RNN(3 => 3, elu),
        #RNN(16 => 32, elu),
    )
    gen = Chain(Dense(4, 10, elu), Dense(10, 1, identity))

    losses = []
    ql5 = []
    @showprogress for _ in 1:2
        loss, _ql5 = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest
        )
        append!(losses, loss)
        append!(ql5, _ql5)
    end

    window_size = 10
    ma_result = moving_average([x[1] for x in ql5], window_size)
    plot(ma_result)

    τ = 1000
    predictions, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=100
    )

    #s = rec(collect(loaderXtrain)[1][1])
    #ideal = collect(loaderx)[1]

    prediction = Vector{Float32}()
    stds = Vector{Float32}()
    mse = 0.0
    mae = 0.0
    count = 0
    for ts in (1:(length(names(df)) - 1))
        xtrain = collect(loaderXtrain)[ts]
        Flux.reset!(rec)
        s = []
        for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
            s = rec([xtrain[j + 1]])
        end

        τ = 96
        xtest = collect(loaderXtest)[ts]
        noise_model = Normal(0.0f0, 1.0f0)
        n_average = 1000
        for j in (0:(τ):(length(xtest) - τ))
            #s = rec(xtest[(j + 1):(j + τ)]')
            #s = rec([xtest[j + 1]])
            for i in 1:(τ)
                xₖ = rand(noise_model, n_average)
                y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                ȳ = mean(y)
                σ = std(y)
                s = rec([ȳ])
                append!(prediction, ȳ)
                append!(stds, σ)
            end
        end
        count += 1
        ideal = collect(loaderXtest)[ts]
        QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
        println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
        mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
    end
end

@test_experiments "Solar" begin
    """
    15 months worth of daily data (440 daily records) that describes the occupancy rate,
    between 0 and 1, of different car lanes of the San Francisco bay area freeways across time.

    We consider each day in this database as a single time series of dimension 963
    (the number of sensors which functioned consistently throughout the studied period) and
    length 6 x 24=144.
    """
    csv_file_path = "/Users/jmfrutos/Desktop/archive/solar_AL.csv"

    df1 = CSV.File(csv_file_path; delim=',', header=false, decimal='.')

    df = DataFrame(df1)

    xtrain = []
    ytrain = []
    ztrain = []
    start = 1
    num_training_data = 5000
    num_test_data = 2000
    start_test = num_training_data + num_test_data

    for name in names(df)[1:10]
        println(name)
        #if name != "Column1"
        ts = getproperty(df, Symbol(name))
        append!(xtrain, ts[start:(start + num_training_data)])
        append!(ytrain, ts[(start + 1):(start + num_training_data)])
        append!(
            ztrain,
            ts[(start_test + start + num_training_data):(start_test + start + num_training_data + num_test_data)],
        )
        #end
    end

    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=1000, K=10)

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 3, relu), RNN(3 => 3, relu))
    gen = Chain(Dense(4, 10, relu), Dense(10, 1, identity))

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

    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    ql5 = []
    @showprogress for _ in 1:10
        loss, _ql5 = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest
        )
        append!(losses, loss)
        append!(ql5, _ql5)
    end

    window_size = 200
    ma_result = moving_average([x[1] for x in ql5], window_size)
    plot(ma_result)

    τ = 24
    predictions, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=100
    )

    #s = rec(collect(loaderXtrain)[1][1])
    #ideal = collect(loaderx)[1]

    hparams.window_size = 30
    Flux.reset!(rec)
    ideal = collect(loaderXtrain)[1]
    s = []
    for j in (0:(hparams.window_size):(length(ideal) - hparams.window_size))
        s = rec(ideal[(j + 1):(j + hparams.window_size)]')
    end

    n_average = 1000
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
    QLρ([x[1] for x in ideal][1:30], prediction[1:30]; ρ=0.5)
end

@test_experiments "ETDataset" begin
    csv1 = "/Users/jmfrutos/github/ETDataset/ETT-small/ETTh1.csv"

    #column_names = [:col1, :col2, :col3, :col4, :col5, :col6, :col7, :col8]

    df1 = CSV.File(csv1; delim=',', header=true, decimal='.')

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=2000, K=50)

    rec = Chain(RNN(1 => 3, relu), LayerNorm(3), Dropout(0.1))
    gen = Chain(Dense(4, 10, relu), Dropout(0.1), Dense(10, 1, identity))

    df = DataFrame(df1)
    df = select(df, Not(:date))
    #new_df = normalize_df(df, 5000, 7500)
    #new_df = mapcols(zscore, df)

    function normalize_df(df::DataFrame, init::Int, final::Int)
        normalized_df = copy(df[init:final, :])
        for col in names(normalized_df)
            normalized_df[!, col] = zscore(normalized_df[!, col])
        end
        return normalized_df
    end

    # Initialize a new DataFrame with the same number of rows
    ma_df = DataFrame()

    window_size = 25
    for name in names(df)
        ts = getproperty(df, Symbol(name))
        rolled = rollmean(ts, window_size)
        ma_df[!, Symbol(name)] = rolled
    end

    xtrain = []
    ytrain = []
    ztrain = []
    start = 1
    num_training_data = 16000
    start_test = num_training_data
    num_test_data = 720

    for name in names(ma_df)
        println(name)
        if name != "date"
            ts = getproperty(ma_df, Symbol(name))
            append!(xtrain, ts[start:(start + num_training_data)])
            append!(ytrain, ts[(start + 1):(start + num_training_data)])
            append!(
                ztrain,
                ts[(start + num_training_data):(start + num_training_data + num_test_data)],
            )
        end
    end

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ytrain)
    v_std = std(ytrain)
    ytrain = (ytrain .- v_mean) ./ v_std

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ztrain)
    v_std = std(ztrain)
    ztrain = (ztrain .- v_mean) ./ v_std

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

    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    ql5 = []
    mses = []
    maes = []
    @showprogress for _ in 1:5
        loss, _ql5 = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest; cond=0.3
        )
        append!(losses, loss)
        append!(ql5, _ql5)

        τ = 96
        predictions, stds = ts_forecast(
            rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=1000
        )

        mse = 0.0
        mae = 0.0
        for ts in (1:(length(names(df)) - 1))
            prediction = Vector{Float32}()
            stds = Vector{Float32}()
            xtrain = collect(loaderXtrain)[ts]
            Flux.reset!(rec)
            s = []
            for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
                s = rec([xtrain[j + 1]])
            end

            τ = 336
            xtest = collect(loaderXtest)[ts]
            noise_model = Normal(0.0f0, 1.0f0)
            n_average = 1000
            for j in (0:(τ):(length(xtest) - τ))
                #s = rec(xtest[(j + 1):(j + τ)]')
                #s = rec([xtest[j + 1]])
                for i in 1:(τ)
                    xₖ = rand(noise_model, n_average)
                    y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                    ȳ = mean(y)
                    σ = std(y)
                    s = rec([ȳ])
                    append!(prediction, ȳ)
                    append!(stds, σ)
                end
            end
            ideal = collect(loaderXtest)[ts]
            QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
            println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
            println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
            mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
            mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
        end
        append!(mses, mse / 7.0)
        append!(maes, mae / 7.0)
    end

    window_size = 50
    ma_result = moving_average([x[1] for x in ql5], window_size)
    plot(ma_result)

    τ = 96
    predictions, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=1000
    )

    mse = 0.0
    mae = 0.0
    for ts in (1:(length(names(df)) - 1))
        prediction = Vector{Float32}()
        stds = Vector{Float32}()
        xtrain = collect(loaderXtrain)[ts]
        Flux.reset!(rec)
        s = []
        for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
            s = rec([xtrain[j + 1]])
        end

        τ = 336
        xtest = collect(loaderXtest)[ts]
        noise_model = Normal(0.0f0, 1.0f0)
        n_average = 1000
        for j in (0:(τ):(length(xtest) - τ))
            #s = rec(xtest[(j + 1):(j + τ)]')
            #s = rec([xtest[j + 1]])
            for i in 1:(τ)
                xₖ = rand(noise_model, n_average)
                y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                ȳ = mean(y)
                σ = std(y)
                s = rec([ȳ])
                append!(prediction, ȳ)
                append!(stds, σ)
            end
        end
        ideal = collect(loaderXtest)[ts]
        QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
        println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
        mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
    end
    mse / 7.0
    mae / 7.0
end

@test_experiments "ETDataset" begin
    csv1 = "/Users/jmfrutos/github/ETDataset/ETT-small/ETTh2.csv"

    #column_names = [:col1, :col2, :col3, :col4, :col5, :col6, :col7, :col8]

    df1 = CSV.File(csv1; delim=',', header=true, decimal='.')

    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=2000, K=10)

    rec = Chain(RNN(1 => 3, relu), LayerNorm(3), Dropout(0.1))
    gen = Chain(Dense(4, 10, relu), Dropout(0.1), Dense(10, 1, identity))

    df = DataFrame(df1)
    df = select(df, Not(:date))
    #new_df = normalize_df(df, 5000, 7500)
    #new_df = mapcols(zscore, df)

    function normalize_df(df::DataFrame, init::Int, final::Int)
        normalized_df = copy(df[init:final, :])
        for col in names(normalized_df)
            normalized_df[!, col] = zscore(normalized_df[!, col])
        end
        return normalized_df
    end

    # Initialize a new DataFrame with the same number of rows
    ma_df = DataFrame()

    window_size = 25
    for name in names(df)
        ts = getproperty(df, Symbol(name))
        rolled = rollmean(ts, window_size)
        ma_df[!, Symbol(name)] = rolled
    end

    xtrain = []
    ytrain = []
    ztrain = []
    start = 1
    num_training_data = 16000
    start_test = num_training_data
    num_test_data = 720

    for name in names(ma_df)
        println(name)
        if name != "date"
            ts = getproperty(ma_df, Symbol(name))
            append!(xtrain, ts[start:(start + num_training_data)])
            append!(ytrain, ts[(start + 1):(start + num_training_data)])
            append!(
                ztrain,
                ts[(start + num_training_data):(start + num_training_data + num_test_data)],
            )
        end
    end

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ytrain)
    v_std = std(ytrain)
    ytrain = (ytrain .- v_mean) ./ v_std

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ztrain)
    v_std = std(ztrain)
    ztrain = (ztrain .- v_mean) ./ v_std

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

    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    ql5 = []
    @showprogress for _ in 1:5
        loss, _ql5 = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest; cond=0.35
        )
        append!(losses, loss)
        append!(ql5, _ql5)
    end

    window_size = 10
    ma_result = moving_average([x[1] for x in ql5], window_size)
    plot(ma_result)

    τ = 30
    predictions, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=1000
    )

    #mapcols(zscore, collect(loaderXtest)[1])

    mse = 0.0
    mae = 0.0
    for ts in (1:(length(names(df)) - 1))
        prediction = Vector{Float32}()
        stds = Vector{Float32}()
        xtrain = collect(loaderXtrain)[ts]
        Flux.reset!(rec)
        s = []
        for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
            s = rec([xtrain[j + 1]])
        end

        τ = 96
        xtest = collect(loaderXtest)[ts]
        noise_model = Normal(0.0f0, 1.0f0)
        n_average = 1000
        for j in (0:(τ):(length(xtest) - τ))
            #s = rec(xtest[(j + 1):(j + τ)]')
            #s = rec([xtest[j + 1]])
            for i in 1:(τ)
                xₖ = rand(noise_model, n_average)
                y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                ȳ = mean(y)
                σ = std(y)
                s = rec([ȳ])
                append!(prediction, ȳ)
                append!(stds, σ)
            end
        end
        ideal = collect(loaderXtest)[ts]
        QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
        println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
        mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
    end
    mse / 7.0
    mae / 7.0
end

@test_experiments "ETDataset" begin
    csv1 = "/Users/jmfrutos/github/ETDataset/ETT-small/ETTm1.csv"

    #column_names = [:col1, :col2, :col3, :col4, :col5, :col6, :col7, :col8]

    df1 = CSV.File(csv1; delim=',', header=true, decimal='.')

    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=8000, K=10)

    rec = Chain(RNN(1 => 3, relu), LayerNorm(3))
    gen = Chain(Dense(4, 10, relu), Dropout(0.1), Dense(10, 1, identity))

    df = DataFrame(df1)
    df = select(df, Not(:date))
    #new_df = normalize_df(df, 5000, 7500)
    #new_df = mapcols(zscore, df)

    function normalize_df(df::DataFrame, init::Int, final::Int)
        normalized_df = copy(df[init:final, :])
        for col in names(normalized_df)
            normalized_df[!, col] = zscore(normalized_df[!, col])
        end
        return normalized_df
    end

    # Initialize a new DataFrame with the same number of rows
    ma_df = DataFrame()

    window_size = 25
    for name in names(df)
        ts = getproperty(df, Symbol(name))
        rolled = rollmean(ts, window_size)
        ma_df[!, Symbol(name)] = rolled
    end

    xtrain = []
    ytrain = []
    ztrain = []
    start = 1
    num_training_data = 10000
    start_test = num_training_data
    num_test_data = 720

    for name in names(ma_df)
        println(name)
        if name != "date"
            ts = getproperty(ma_df, Symbol(name))
            append!(xtrain, ts[start:(start + num_training_data)])
            append!(ytrain, ts[(start + 1):(start + num_training_data)])
            append!(
                ztrain,
                ts[(start + num_training_data):(start + num_training_data + num_test_data)],
            )
        end
    end

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ytrain)
    v_std = std(ytrain)
    ytrain = (ytrain .- v_mean) ./ v_std

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ztrain)
    v_std = std(ztrain)
    ztrain = (ztrain .- v_mean) ./ v_std

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

    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    ql5 = []
    @showprogress for _ in 1:10
        loss, _ql5 = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest; cond=0.5
        )
        append!(losses, loss)
        append!(ql5, _ql5)
    end

    window_size = 10
    ma_result = moving_average([x[1] for x in ql5], window_size)
    plot(ma_result)

    τ = 30
    predictions, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=1000
    )

    mse = 0.0
    mae = 0.0
    for ts in (1:(length(names(df)) - 1))
        prediction = Vector{Float32}()
        stds = Vector{Float32}()
        xtrain = collect(loaderXtrain)[ts]
        Flux.reset!(rec)
        s = []
        for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
            s = rec([xtrain[j + 1]])
        end

        τ = 720
        xtest = collect(loaderXtest)[ts]
        noise_model = Normal(0.0f0, 1.0f0)
        n_average = 1000
        for j in (0:(τ):(length(xtest) - τ))
            #s = rec(xtest[(j + 1):(j + τ)]')
            #s = rec([xtest[j + 1]])
            for i in 1:(τ)
                xₖ = rand(noise_model, n_average)
                y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                ȳ = mean(y)
                σ = std(y)
                s = rec([ȳ])
                append!(prediction, ȳ)
                append!(stds, σ)
            end
        end
        ideal = collect(loaderXtest)[ts]
        QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
        println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
        mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
    end
    mse / 7.0
    mae / 7.0
end

@test_experiments "ETDataset" begin
    csv1 = "/Users/jmfrutos/github/ETDataset/ETT-small/ETTm2.csv"

    #column_names = [:col1, :col2, :col3, :col4, :col5, :col6, :col7, :col8]

    df1 = CSV.File(csv1; delim=',', header=true, decimal='.')

    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=10000, K=10)

    rec = Chain(
        LSTM(1 => 10),
        #Dropout(0.05),
        LayerNorm(10),
    )
    gen = Chain(Dense(11, 32, relu), Dropout(0.1), Dense(32, 1, identity))

    df = DataFrame(df1)
    df = select(df, Not(:date))
    #new_df = normalize_df(df, 5000, 7500)
    #new_df = mapcols(zscore, df)

    function normalize_df(df::DataFrame, init::Int, final::Int)
        normalized_df = copy(df[init:final, :])
        for col in names(normalized_df)
            normalized_df[!, col] = zscore(normalized_df[!, col])
        end
        return normalized_df
    end

    # Initialize a new DataFrame with the same number of rows
    ma_df = DataFrame()

    window_size = 25
    for name in names(df)
        ts = getproperty(df, Symbol(name))
        rolled = rollmean(ts, window_size)
        ma_df[!, Symbol(name)] = rolled
    end

    xtrain = []
    ytrain = []
    ztrain = []
    start = 1
    num_training_data = 12000
    start_test = num_training_data
    num_test_data = 96

    for name in names(ma_df)
        println(name)
        if name != "date"
            ts = getproperty(ma_df, Symbol(name))
            append!(xtrain, ts[start:(start + num_training_data)])
            append!(ytrain, ts[(start + 1):(start + num_training_data)])
            append!(
                ztrain,
                ts[(start + num_training_data):(start + num_training_data + num_test_data)],
            )
        end
    end

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ytrain)
    v_std = std(ytrain)
    ytrain = (ytrain .- v_mean) ./ v_std

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ztrain)
    v_std = std(ztrain)
    ztrain = (ztrain .- v_mean) ./ v_std

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

    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    ql5 = []
    @showprogress for _ in 1:10
        loss, _ql5 = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest; cond=0.3
        )
        append!(losses, loss)
        append!(ql5, _ql5)
    end

    window_size = 10
    ma_result = moving_average([x[1] for x in ql5], window_size)
    plot(ma_result)

    τ = 30
    predictions, stds = ts_forecast(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], τ; n_average=1000
    )

    mse = 0.0
    mae = 0.0
    for ts in (1:(length(names(df)) - 6))
        prediction = Vector{Float32}()
        stds = Vector{Float32}()
        xtrain = collect(loaderXtrain)[ts]
        Flux.reset!(rec)
        s = []
        for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
            s = rec([xtrain[j + 1]])
        end

        τ = 96
        xtest = collect(loaderXtest)[ts]
        noise_model = Normal(0.0f0, 1.0f0)
        n_average = 100
        for j in (0:(τ):(length(xtest) - τ))
            #s = rec(xtest[(j + 1):(j + τ)]')
            #s = rec([xtest[j + 1]])
            for i in 1:(τ)
                xₖ = rand(noise_model, n_average)
                y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                ȳ = mean(y)
                σ = std(y)
                s = rec([ȳ])
                append!(prediction, ȳ)
                append!(stds, σ)
            end
        end
        ideal = collect(loaderXtest)[ts]
        QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
        println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
        mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
    end
    mse / 7.0
    mae / 7.0
end

@test_experiments "ETDataset" begin
    csv1 = "/Users/jmfrutos/github/ETDataset/ETT-small/ETTh2.csv"

    #column_names = [:col1, :col2, :col3, :col4, :col5, :col6, :col7, :col8]

    df1 = CSV.File(csv1; delim=',', header=true, decimal='.')

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=40)

    rec = Chain(RNN(7 => 10, relu), LayerNorm(10))
    gen = Chain(Dense(11, 15, relu), Dropout(0.05), Dense(15, 7, identity))

    df = DataFrame(df1)
    df = select(df, Not(:date))

    matriz = Float32.(Matrix(df))
    #new_df = normalize_df(df, 5000, 7500)
    #new_df = mapcols(zscore, df)

    function normalize_df(df::DataFrame, init::Int, final::Int)
        normalized_df = copy(df[init:final, :])
        for col in names(normalized_df)
            normalized_df[!, col] = zscore(normalized_df[!, col])
        end
        return normalized_df
    end

    # Initialize a new DataFrame with the same number of rows
    ma_df = DataFrame()

    window_size = 25
    for name in names(df)
        ts = getproperty(df, Symbol(name))
        rolled = rollmean(ts, window_size)
        ma_df[!, Symbol(name)] = rolled
    end

    xtrain = []
    ytrain = []
    ztrain = []
    start = 1
    num_training_data = 10000
    start_test = num_training_data
    num_test_data = 96

    for name in names(ma_df)
        println(name)
        if name != "date"
            ts = getproperty(ma_df, Symbol(name))
            push!(xtrain, ts[start:(start + num_training_data)])
            push!(ytrain, ts[(start + 1):(start + num_training_data)])
            push!(
                ztrain,
                ts[(start + num_training_data):(start + num_training_data + num_test_data)],
            )
        end
    end

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ytrain)
    v_std = std(ytrain)
    ytrain = (ytrain .- v_mean) ./ v_std

    v_mean = mean(xtrain)
    v_std = std(xtrain)
    xtrain = (xtrain .- v_mean) ./ v_std

    v_mean = mean(ztrain)
    v_std = std(ztrain)
    ztrain = (ztrain .- v_mean) ./ v_std

    mean_vals = mean(matriz; dims=1)
    std_vals = std(matriz; dims=1)
    matriz = (matriz .- mean_vals) ./ std_vals

    dataX = [matriz[i, :] for i in 1:size(matriz, 1)]
    dataY = [matriz[i, :] for i in 2:size(matriz, 1)]

    # Define the batch size
    batch_size = 4000

    # Create the DataLoader
    loaderXtrain = DataLoader(dataX; batchsize=batch_size, shuffle=false, partial=false)
    loaderYtrain = DataLoader(dataY; batchsize=batch_size, shuffle=false, partial=false)

    maes = []
    mses = []
    losses = []
    @showprogress for i in 1:1000
        loss = ts_invariant_statistical_loss_multivariate(
            rec, gen, loaderXtrain, loaderYtrain, hparams
        )
        append!(losses, loss)

        mse = 0.0
        mae = 0.0
        count = 0
        τ = 336
        for ts in (1:length(collect(loaderXtrain)[1][1]))
            #τ = 96
            s = 0
            Flux.reset!(rec)
            xtrain = collect(loaderXtrain)[1]
            prediction = []
            for i in 1:96
                s = rec(xtrain[i])
                #xₖ = rand(Normal(0.0f0, 1.0f0), 1000)
                #ŷ = mean([gen(vcat(x, s)) for x in xₖ])
                #push!(prediction, ŷ)
            end
            for i in (96:(96 + τ))
                xₖ = rand(Normal(0.0f0, 1.0f0), 100)
                y = mean([gen(vcat(x, s)) for x in xₖ])
                #y = mean([gen(vcat(x, s)) for x in xₖ])
                #println(ŷ)
                s = rec(y)
                push!(prediction, y)
            end
            primeraColumnaY = [fila[ts] for fila in prediction]
            primeraColumna = [fila[ts] for fila in collect(loaderXtrain)[1]]
            count += 1
            #l = moving_average(primeraColumnaY, 10)
            #println(MSE(primeraColumna[96:96+τ], primeraColumnaY))
            #println(MAE(primeraColumna[96:96+τ], primeraColumnaY))
            mse += MSE(primeraColumna[96:(96 + τ)], primeraColumnaY)
            mae += MAE(primeraColumna[96:(96 + τ)], primeraColumnaY)
        end
        append!(mses, mse / count)
        append!(maes, mae / count)
    end

    l = moving_average(losses, 100)
    plot(l)

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

    loaderXtest = Flux.DataLoader(
        map(x -> Float32.(x), ztrain);
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    losses = []
    ql5 = []
    @showprogress for _ in 1:10
        loss, _ql5 = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams
        )
        append!(losses, loss)
        append!(ql5, _ql5)
    end

    Flux.reset!(rec)
    xtrain = collect(loaderXtrain)[1]
    prediction = []
    for i in 1:48
        s = rec(xtrain[i])
        xₖ = rand(Normal(0.0f0, 1.0f0), 1000)
        ŷ = mean([gen(vcat(x, s)) for x in xₖ])
        push!(prediction, ŷ)
    end
    primeraColumnaY = [fila[1] for fila in prediction]
    l = moving_average(primeraColumnaY, 10)
    plot(primeraColumna[1:48])
    plot!(l)

    mse = 0.0
    mae = 0.0
    count = 0
    τ = 720
    s = 0
    for ts in (1:length(collect(loaderXtrain)[1][1]))
        #τ = 96
        Flux.reset!(rec)
        xtrain = collect(loaderXtrain)[1]
        prediction = []
        for i in 1:96
            s = rec(xtrain[i])
            xₖ = rand(Normal(0.0f0, 1.0f0), 1000)
            ŷ = mean([gen(vcat(x, s)) for x in xₖ])
            push!(prediction, ŷ)
        end
        for i in (96:(96 + τ))
            xₖ = rand(Normal(0.0f0, 1.0f0), 1000)
            ŷ = mean([gen(vcat(x, s)) for x in xₖ])
            #println(ŷ)
            s = rec(ŷ)
            push!(prediction, ŷ)
        end
        primeraColumnaY = [fila[ts] for fila in prediction]
        primeraColumna = [fila[ts] for fila in collect(loaderXtrain)[1]]
        count += 1
        l = moving_average(primeraColumnaY, 10)
        println(MSE(primeraColumna[96:(96 + τ)], primeraColumnaY))
        println(MAE(primeraColumna[96:(96 + τ)], primeraColumnaY))
        mse += MSE(primeraColumna[96:(96 + τ)], primeraColumnaY)
        mae += MAE(primeraColumna[96:(96 + τ)], primeraColumnaY)
    end
    mse / count
    mae / count

    plot(primeraColumna[1:144])
    plot!(l)

    MSE(primeraColumna[1:145], l)

    mse = 0.0
    mae = 0.0
    for ts in (1:(length(names(df))))
        prediction = Vector{Float32}()
        stds = Vector{Float32}()
        xtrain = collect(loaderXtrain)[ts]
        Flux.reset!(rec)
        s = []
        for j in ((length(xtrain) - 96):1:(length(xtrain) - 1))
            s = rec([xtrain[j + 1]])
        end

        τ = 96
        xtest = collect(loaderXtest)[ts]
        noise_model = Normal(0.0f0, 1.0f0)
        n_average = 100
        for j in (0:(τ):(length(xtest) - τ))
            #s = rec(xtest[(j + 1):(j + τ)]')
            #s = rec([xtest[j + 1]])
            for i in 1:(τ)
                xₖ = rand(noise_model, n_average)
                y = hcat([gen(vcat(x, s)) for x in xₖ]...)
                ȳ = mean(y)
                σ = std(y)
                s = rec([ȳ])
                append!(prediction, ȳ)
                append!(stds, σ)
            end
        end
        ideal = collect(loaderXtest)[ts]
        QLρ([x[1] for x in ideal][1:τ], prediction[1:τ]; ρ=0.5)
        println(MSE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        println(MAE([x[1] for x in ideal][1:τ], prediction[1:τ]))
        mse += MSE([x[1] for x in ideal][1:τ], prediction[1:τ])
        mae += MAE([x[1] for x in ideal][1:τ], prediction[1:τ])
    end
end
