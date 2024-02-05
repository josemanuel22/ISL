# Core functionality
using Flux           # For neural networks
using Random         # For random number generation
using Statistics     # For basic statistical functions

# Data handling and manipulation
using DataFrames     # For data manipulation and representation
using CSV            # For reading and writing CSV files
using Distributions  # For probability distributions
using StatsBase      # For basic statistical support
using RollingFunctions  # For applying functions with a rolling window

# Plotting and visualization
using Plots          # For creating plots

# External utilities and custom functions
include("../utils.jl")       # Include general utility functions
include("ts_utils.jl")       # Include time series utility functions

# Additional packages for specific tasks (if any used in your project)
using ISL             # Assuming this is a custom or specific package related to your project

standardize!(data) = (data .- mean(data)) ./ std(data)

function create_data_loader(data, batchsize; shuffle=false, partial=false)
    return Flux.DataLoader(data; batchsize=batchsize, shuffle=shuffle, partial=partial)
end

function aggregate_time_series_data(
    df::DataFrame,
    select_names::Vector{String},
    start::Int,
    num_training_data::Int,
    num_test_data::Int,
    coarse::Int=4,
)
    aggregated_data_xtrain = Float32[]
    aggregated_data_ytrain = Float32[]
    aggregated_data_xtest = Float32[]

    for name in select_names
        ts = getproperty(df, Symbol(name))

        loaderXtrain = ts[start:(start + num_training_data)]
        loaderYtrain = ts[(start + 1):(start + num_training_data + 1)]
        loaderXtest = ts[(start + num_training_data - 1):(start + num_training_data + num_test_data)]

        for i in 1:coarse:length(loaderXtrain)
            push!(
                aggregated_data_xtrain,
                Float32(sum(loaderXtrain[i:min(i + coarse - 1, length(loaderXtrain))])),
            )
        end

        for i in 1:coarse:length(loaderYtrain)
            push!(
                aggregated_data_ytrain,
                Float32(sum(loaderYtrain[i:min(i + coarse - 1, length(loaderYtrain))])),
            )
        end

        for i in 1:coarse:length(loaderXtest)
            push!(
                aggregated_data_xtest,
                Float32(sum(loaderXtest[i:min(i + coarse - 1, length(loaderXtest))])),
            )
        end
    end

    # Remove missing values
    aggregated_data_xtrain = filter(!ismissing, aggregated_data_xtrain)
    aggregated_data_ytrain = filter(!ismissing, aggregated_data_ytrain)
    aggregated_data_xtest = filter(!ismissing, aggregated_data_xtest)

    # Standardize data
    aggregated_data_xtrain = standardize!(aggregated_data_xtrain)
    aggregated_data_ytrain = standardize!(aggregated_data_ytrain)
    aggregated_data_xtest = standardize!(aggregated_data_xtest)

    loader_xtrain = create_data_loader(aggregated_data_xtrain, num_training_data)
    loader_ytrain = create_data_loader(aggregated_data_ytrain, num_training_data)
    loader_xtest = create_data_loader(aggregated_data_xtest, num_test_data)

    return loader_xtrain, loader_ytrain, loader_xtest
end

function aggregate_data_from_columns(
    df::DataFrame, cols::Vector{String}, start::Int, num_training_data::Int, num_test::Int
)
    # Initialize vectors to store the aggregated data
    aggregated_data_xtrain = Vector{Float32}()
    aggregated_data_ytrain = Vector{Float32}()
    aggregated_data_xtest = Vector{Float32}()

    # Iterate over the specified columns and aggregate the data
    for name in cols
        ts = getproperty(df, Symbol(name))

        # Append training data
        append!(aggregated_data_xtrain, ts[start:(start + num_training_data - 1)])

        # Append training target data (offset by 1)
        append!(aggregated_data_ytrain, ts[(start + 1):(start + num_training_data)])

        # Append testing data
        append!(
            aggregated_data_xtest,
            ts[(start + num_training_data):(start + num_training_data + num_test - 1)],
        )
    end

    # Standardize data
    aggregated_data_xtrain = standardize!(aggregated_data_xtrain)
    aggregated_data_ytrain = standardize!(aggregated_data_ytrain)
    aggregated_data_xtest = standardize!(aggregated_data_xtest)

    # DataLoader setup
    loaderXtrain = DataLoader(
        aggregated_data_xtrain;
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )
    loaderYtrain = DataLoader(
        aggregated_data_ytrain;
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )
    loaderXtest = DataLoader(
        aggregated_data_xtest; batchsize=round(Int, num_test), shuffle=false, partial=false
    )

    return loaderXtrain, loaderYtrain, loaderXtest
end

@test_experiments "testing AutoRegressive Model 1" begin
    # --- Model Parameters and Data Generation ---

    # Define AR model parameters
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],  # Autoregressive coefficients
        x₁=rand(Normal(0.0f0, 1.0f0)),  # Initial value from a Normal distribution
        proclen=2000,  # Length of the process
        noise=Normal(0.0f0, 0.2f0),  # Noise in the AR process
    )

    # Define the recurrent and generative models
    recurrent_model = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    generative_model = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    # Generate training and testing data
    n_series = 200  # Number of series to generate
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        n_series, ar_hparams
    )

    # --- Training Configuration ---

    # Define hyperparameters for time series prediction
    ts_hparams = HyperParamsTS(;
        seed=1234,
        η=1e-3,  # Learning rate
        epochs=n_series,
        window_size=1000,  # Size of the window for prediction
        K=10,  # Hyperparameter K (if it has a specific use, add a comment)
    )

    # Train model and calculate loss
    loss = ts_invariant_statistical_loss_one_step_prediction(
        recurrent_model, generative_model, loaderXtrain, loaderYtrain, ts_hparams
    )

    # --- Visualization ---

    # Plotting the time series prediction
    plot_univariate_ts_prediction(
        recurrent_model,
        generative_model,
        collect(loaderXtrain)[1],  # Extract the first batch for plotting
        collect(loaderXtest)[1],  # Extract the first batch for plotting
        ts_hparams;
        n_average=1000,  # Number of predictions to average
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
    # Load dataset
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

    # Model setup
    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=2000, window_size=1000, K=20)
    rec = Chain(LSTM(1 => 16), LayerNorm(16))
    gen = Chain(Dense(17, 32, elu), Dropout(0.05), Dense(32, 1, identity))

    # Data preprocessing
    cols = [
        "MT_005",
        "MT_006",
        "MT_007",
        "MT_008",
        "MT_168",
        "MT_333",
        "MT_334",
        "MT_335",
        "MT_336",
        "MT_338",
    ]
    start, num_training_data, num_test = 35040, 1000, 1000

    loaderXtrain, loaderYtrain, loaderXtest = aggregate_data_from_columns(
        df, cols, start, num_training_data, num_test
    )

    # Model training
    losses = []
    @showprogress for _ in 1:10
        loss = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest
        )
        append!(losses, loss)
    end
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
    # Data loading and preprocessing
    csv_file_path = "examples/time_series_predictions/data/LD2011_2014.txt"

    # Specify custom types for certain columns to ensure proper data handling
    column_types = Dict(
        "MT_005" => Float32,
        "MT_006" => Float32,
        "MT_007" => Float32,
        "MT_008" => Float32,
        "MT_168" => Float32,
        "MT_333" => Float32,
        "MT_334" => Float32,
        "MT_335" => Float32,
        "MT_336" => Float32,
        "MT_338" => Float32,
    )

    df = DataFrame(
        CSV.File(csv_file_path; delim=';', header=true, decimal=',', types=column_types)
    )

    # Hyperparameters setup
    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=1000, K=10)

    # Model definition
    rec = Chain(RNN(1 => 3, relu), RNN(3 => 3, relu))
    gen = Chain(Dense(4, 10, identity), Dense(10, 1, identity))

    # Training and testing data setup
    start, num_training_data, num_test_data = 35040, 35040, 1000

    # Aggregate time series data for training and testing
    selected_columns = [
        "MT_005",
        "MT_006",
        "MT_007",
        "MT_008",
        "MT_168",
        "MT_333",
        "MT_334",
        "MT_335",
        "MT_336",
        "MT_338",
    ]

    loader_xtrain, loader_ytrain, loader_xtest = aggregate_time_series_data(
        df, selected_columns, start, num_training_data, num_test_data
    )

    # Model training
    losses = []
    @showprogress for _ in 1:100
        loss = ts_invariant_statistical_loss(
            rec, gen, loader_xtrain, loader_ytrain, hparams
        )
        append!(losses, loss)
    end
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
    csv1 = "/Users/jmfrutos/github/ETDataset/ETT-small/ETTh1.csv"

    #column_names = [:col1, :col2, :col3, :col4, :col5, :col6, :col7, :col8]

    df1 = CSV.File(csv1; delim=',', header=true, decimal='.')

    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=2000, K=40)

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
    mses = []
    maes = []
    @showprogress for _ in 1:20
        loss = ts_invariant_statistical_loss(
            rec, gen, loaderXtrain, loaderYtrain, hparams, loaderXtest; cond=0.5
        )
        append!(losses, loss)
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
        append!(maes, mae / 7.0)
        append!(mses, mse / 7.0)
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

@test_experiments "ETDataset multivariated" begin
    # Load CSV file
    csv_file = "/Users/jmfrutos/github/ETDataset/ETT-small/ETTh2.csv"
    df = DataFrame(CSV.File(csv_file; delim=',', header=true, decimal='.'))

    # Select relevant columns and standardize data
    df = select(df, Not(:date))
    matrix = Float32.(Matrix(df))
    mean_vals, std_vals = mean(matrix; dims=1), std(matrix; dims=1)
    matrix = (matrix .- mean_vals) ./ std_vals

    # Preparing data for training
    dataX = [matrix[i, :] for i in 1:size(matrix, 1)]
    dataY = [matrix[i, :] for i in 2:size(matrix, 1)]

    # Model hyperparameters and architecture
    hparams = HyperParamsTS(; seed=1234, η=1e-4, epochs=2000, window_size=2000, K=30)
    rec = Chain(RNN(7 => 20, relu), LayerNorm(20), Dropout(0.10))
    gen = Chain(Dense(21, 15, relu), Dropout(0.10), Dense(15, 7, identity), Dropout(0.10))

    # DataLoader setup
    batch_size = 2000
    loaderXtrain = DataLoader(dataX; batchsize=batch_size, shuffle=false, partial=false)
    loaderYtrain = DataLoader(dataY; batchsize=batch_size, shuffle=false, partial=false)

    maes = []
    mses = []
    losses = []
    @showprogress for i in 1:10000
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
end
