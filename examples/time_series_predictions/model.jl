using Flux
using Random
using Statistics

using AdaptativeBlockLearning
using Distributions
using DataFrames
using CSV
using Plots

include("../../benchmarks/benchmark_utils.jl")
include("ts_utils.jl")

function ts_covariates_mse_learning(rec, gen, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim_rec = Flux.setup(Flux.Adam(hparams.η), rec)
    optim_gen = Flux.setup(Flux.Adam(hparams.η), gen)
    @showprogress for epoch in (1:(hparams.epochs))
        Flux.reset!(nn_model)
        loss, grads = Flux.withgradient(rec, gen) do rec, gen
            for i in 1:(length(Xₜ) - 1)
                s = rec(Xₜ[i])
                yₖ = gen(vcat(Xₜ[i], s))
                totla
            end
            s = rec(Xₜ[1])
            sum(Flux.Losses.mse.([gen(x)[1] for x in Xₜ[1:end]], Xₜ₊₁[1:end]))
        end
        Flux.update!(optim_rec, rec, grads[1])
        Flux.update!(optim_gen, gen, grads[2])
        push!(losses, loss)
    end
    return losses
end

function ts_mse_learning(nn_model, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for epoch in (1:(hparams.epochs))
        Flux.reset!(rec)
        loss, grads = Flux.withgradient(nn_model) do nn
            nn([Xₜ[1]])
            sum(Flux.Losses.mse.([nn([x])[1] for x in Xₜ[2:(end - 1)]], Xₜ₊₁[2:end]))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end

function ts_mse_learning_2(nn_model, data_Xₜ, data_Xₜ₊₁, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for (Xₜ, Xₜ₊₁) in zip(data_Xₜ, data_Xₜ₊₁)
        Flux.reset!(rec)
        loss, grads = Flux.withgradient(nn_model) do nn
            nn([Xₜ[1]])
            sum(Flux.Losses.mse.([nn([x])[1] for x in Xₜ[2:(end - 1)]], Xₜ₊₁[2:end]))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end

function plot_mse(nn_model, Xₜ)
    function format_numbers(x)
        if abs(x) < 0.01
            formatted_x = @sprintf("%.2e", x)
        else
            formatted_x = @sprintf("%.4f", x)
        end
        return formatted_x
    end

    plot(Xₜ)
    prediction = [nn_model([x])[1] for x in Xₜ[1:end]]

    nd = ND(Xₜ, prediction)
    rmse = RMSE(Xₜ, prediction)
    qlρ = QLρ(Xₜ, prediction; ρ=0.9)
    return plot!(
        prediction;
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


function mean_yule_walker(hparams, ar_hparams, rec, gen, n_average)
    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    coefs = []
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
        coef, _ = yule_walker(prediction; order=3)
        push!(coefs, coef)
    end
    return mean(coefs)
end

@test_experiments "testing AutoRegressive Model 0" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.5f0, 0.1f0),
        train_ratio=0.8,
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=100, window_size=1000, K=10)

    nn_model = Chain(
        RNN(1 => 10, relu), RNN(10 => 10, relu), Dense(10, 16, relu), Dense(16, 1, identity)
    )
    #rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    #gen = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)

    plot_univariate_ts_prediction(
        rec,
        gen,
        collect(loaderXtrain)[15],
        collect(loaderXtest)[15],
        hparams;
        n_average=1000,
    )
end

@test_experiments "testing AutoRegressive Model 0" begin
    ar_hparams1 = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.0f0, 0.2f0),
    )

    ar_hparams2 = ARParams(;
        ϕ=[0.8f0, 0.1f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.0f0, 0.2f0),
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=200, window_size=1000, K=5)

    nn_model = Chain(
        RNN(1 => 10, relu), RNN(10 => 10, relu), Dense(10, 16, relu), Dense(16, 1, identity)
    )
    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams1, ar_hparams2
    )

    loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)

    plot_univariate_ts_prediction(
        rec,
        gen,
        collect(loaderXtrain)[15],
        collect(loaderXtest)[15],
        hparams;
        n_average=1000,
    )

    loss = plot_mse(nn_model, collect(loaderXtest)[15])

    plot_univariate_ts_forecasting(
        rec, gen, collect(loaderXtrain)[15], collect(loaderXtest)[15], hparams; n_average=1
    )

    ts_mse_learning_2(nn_model, loaderXtrain, loaderYtrain, hparams)

    mean_yule_walker(hparams, ar_hparams, rec, gen, 100)
end

@test_experiments "testing AutoRegressive Model 1" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.0f0, 0.2f0),
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=1000, window_size=1000, K=10)

    #nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=1000
    )

    plot_univariate_ts_forecasting(
        rec, gen, collect(loaderXtrain)[15], collect(loaderXtest)[15], hparams; n_average=1
    )

    mean_yule_walker(hparams, ar_hparams, rec, gen, 100)


    using CSV
    using DataFrames

    # Initialize DataFrame
    df = DataFrame(timestamp=Int32[], a=Float32[])

    i=1
    for batch in loaderXtrain
        # Assuming that batch is a collection of rows, where each row is a tuple (or any Iterable) of two elements
        sub_df = DataFrame(timestamp=Int64[], a=Float32[])
        for row in batch
            try
                push!(sub_df, Dict(:timestamp => i, :a => row[1]))
                i+=1
            catch
                break
            end
        end
        append!(df, sub_df)
    end

    CSV.write("output_file.csv", df)


end

@test_experiments "testing AutoRegressive Model 2" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=4000,
        noise=Normal(0.0f0, 0.2f0),
        train_ratio=0.6,
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=200, window_size=1000, K=5)

    #nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=1000
    )
end

@test_experiments "testing AutoRegressive Model 2" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.0f0, 0.5f0),
        train_ratio=0.8,
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=500, window_size=1000, K=10)

    #nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[2], collect(loaderXtest)[2], hparams; n_average=100
    )

    t₀ = 100
    τ = 20
    nd = []
    rmse = []
    stdevs = []
    predictionss = []
    for X_data in loaderXtrain
        predictions, stdev = ts_forecasting(rec, gen, X_data, t₀, τ, 1000)
        push!(predictionss, predictions)
        push!(stdevs, stdev)
        push!(nd, ND(X_data[t₀+1:t₀+τ], predictions))
        push!(rmse, RMSE(X_data[t₀+1:t₀+τ], predictions))
    end
    mean(nd)
    mean(rmse)
    mean(stdevs)

    mss = []
    sdevss = []
    for X_data in loaderXtrain
        m = mean(X_data[t₀+1:t₀+τ])
        stdv  =  std(X_data[t₀+1:t₀+τ])
        push!(mss, m)
        push!(sdevss, stdv)
    end


end

@test_experiments "testing AutoRegressive Model" begin
    ar_hparams = ARParams(;
        ϕ=[0.8f0, -0.4f0, -0.4f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=2000,
        noise=Normal(0.0f0, 0.2f0),
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=200, window_size=1000, K=10)

    #nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 32, relu), Dense(32, 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=100
    )

    mean_yule_walker(hparams, ar_hparams, rec, gen, 100)
end

@test_experiments "testing AutoRegressive Model 3" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=4000,
        noise=Normal(0.0f0, 0.5f0),
        train_ratio=0.8,
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=100, window_size=1000, K=10)

    #nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 32, relu), Dense(32, 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)

    plot_univariate_ts_prediction(
        rec,
        gen,
        collect(loaderXtrain)[1],
        collect(loaderXtest)[1],
        hparams;
        n_average=10000,
    )
end

@test_experiments "testing AutoRegressive Model 4" begin
    ar_hparams = ARParams(;
        ϕ=[0.5f0, 0.3f0, 0.2f0],
        x₁=rand(Normal(0.0f0, 1.0f0)),
        proclen=4000,
        noise=Normal(0.0f0, 0.5f0),
        train_ratio=0.6,
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=200, window_size=1000, K=10)

    #nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
    gen = Chain(Dense(11, 32, relu), Dense(32, 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[3], collect(loaderXtest)[3], hparams; n_average=1000
    )

    c = mean_yule_walker(hparams, ar_hparams, rec, gen, 100)
end

@test_experiments "testing al-pv-2006" begin
    #=
    https://www.nrel.gov/grid/solar-power-data.html
    =#

    csv1 = "examples/time_series_predictions/data/al-pv-2006/Actual_30.45_-88.25_2006_UPV_70MW_5_Min.csv"

    df1 = CSV.File(
        csv1; delim=',', header=true, decimal='.', types=Dict("Power(MW)" => Float32)
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=100, window_size=1000, K=10)

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))

    rec = Chain(RNN(1 => 5, elu), RNN(5 => 5, elu))
    gen = Chain(Dense(6, 32, elu), Dense(32, 1, identity))

    num_training_data = 10000

    loaderXtrain = Flux.DataLoader(
        [getproperty(df1, Symbol("Power(MW)"))[1:num_training_data]];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        [getproperty(df1, Symbol("Power(MW)"))[2:(num_training_data + 1)]];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test_data = 1000
    loaderXtest = Flux.DataLoader(
        [
            getproperty(df1, Symbol("Power(MW)"))[num_training_data:(num_training_data + num_test_data)],
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loss = ts_adaptative_block_learning(
        rec, gen, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
    )

    plot_univariate_ts_prediction(
        rec,
        gen,
        collect(loaderXtrain)[1][1],
        collect(loaderXtest)[1][1],
        hparams;
        n_average=10,
    )
end

@test_experiments "testing al-pv-2006" begin
    #=
    https://www.nrel.gov/grid/solar-power-data.html
    =#

    csv1 = "examples/time_series_predictions/data/al-pv-2006/Actual_30.45_-88.25_2006_UPV_70MW_5_Min.csv"
    csv2 = "examples/time_series_predictions/data/al-pv-2006/Actual_30.55_-87.55_2006_UPV_80MW_5_Min.csv"
    csv3 = "examples/time_series_predictions/data/al-pv-2006/Actual_30.55_-87.75_2006_DPV_36MW_5_Min.csv"
    csv4 = "examples/time_series_predictions/data/al-pv-2006/Actual_30.55_-88.15_2006_DPV_38MW_5_Min.csv"
    csv5 = "examples/time_series_predictions/data/al-pv-2006/Actual_30.55_-88.25_2006_DPV_38MW_5_Min.csv"

    df1 = CSV.File(
        csv1; delim=',', header=true, decimal='.', types=Dict("Power(MW)" => Float32)
    )

    df2 = CSV.File(
        csv2; delim=',', header=true, decimal='.', types=Dict("Power(MW)" => Float32)
    )

    df3 = CSV.File(
        csv3; delim=',', header=true, decimal='.', types=Dict("Power(MW)" => Float32)
    )

    df4 = CSV.File(
        csv4; delim=',', header=true, decimal='.', types=Dict("Power(MW)" => Float32)
    )

    df5 = CSV.File(
        csv5; delim=',', header=true, decimal='.', types=Dict("Power(MW)" => Float32)
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=100, window_size=1000, K=10)

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))

    rec = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu))
    gen = Chain(Dense(33, 64, relu), Dense(64, 1, identity))

    num_training_data = 1000

    loaderXtrain = Flux.DataLoader(
        [
            [
                getproperty(df1, Symbol("Power(MW)"))[i],
                getproperty(df2, Symbol("Power(MW)"))[i],
                getproperty(df3, Symbol("Power(MW)"))[i],
                getproperty(df4, Symbol("Power(MW)"))[i],
                getproperty(df5, Symbol("Power(MW)"))[i],
            ] for i in 1:num_training_data
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        [
            [
                getproperty(df1, Symbol("Power(MW)"))[i],
                getproperty(df2, Symbol("Power(MW)"))[i],
                getproperty(df3, Symbol("Power(MW)"))[i],
                getproperty(df4, Symbol("Power(MW)"))[i],
                getproperty(df5, Symbol("Power(MW)"))[i],
            ] for i in 2:(num_training_data + 1)
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test_data = 10000
    loaderXtest = Flux.DataLoader(
        [
            [
                getproperty(df1, Symbol("Power(MW)"))[i],
                getproperty(df2, Symbol("Power(MW)"))[i],
                getproperty(df3, Symbol("Power(MW)"))[i],
                getproperty(df4, Symbol("Power(MW)"))[i],
                getproperty(df5, Symbol("Power(MW)"))[i],
            ] for i in (num_training_data - 1):(num_training_data + num_test_data)
        ];
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    loss = ts_covariates_adaptative_block_learning(
        rec, gen, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
    )

    plot_multivariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=1000
    )
end

@test_experiments "testing 30 years european wind generation" begin
    #=
    https://www.nrel.gov/grid/solar-power-data.html
    =#

    csv1 = "examples/time_series_predictions/data/30-years-of-european-wind-generation/TS.CF.N2.30yr.csv"

    df1 = CSV.File(csv1; delim=',', header=true, decimal='.', types=Dict("AT11" => Float32))

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=200, window_size=200, K=5)

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))

    rec = Chain(RNN(2 => 32, relu), RNN(32 => 32, relu))
    gen = Chain(Dense(33, 64, relu), Dense(64, 1, relu))

    start = 1
    num_training_data = 1000

    loaderXtrain = Flux.DataLoader(
        map(
            x -> Float32.(x),
            cat(
                [[df1.FR82[i], df1.FR71[i]] for i in start:(start + num_training_data)];
                dims=1,
            ),
        );
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        map(
            x -> Float32.(x),
            cat(
                [
                    [df1.FR82[i], df1.FR71[i]] for
                    i in (start + 1):(start + num_training_data + 1)
                ];
                dims=1,
            ),
        );
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test_data = 1000
    loaderXtest = Flux.DataLoader(
        map(
            x -> Float32.(x),
            cat(
                [
                    [df1.FR82[i], df1.FR71[i]] for i in
                    (start + num_training_data):(start + num_training_data + num_test_data)
                ];
                dims=1,
            ),
        );
        batchsize=round(Int, num_test_data),
        shuffle=false,
        partial=false,
    )

    loss = ts_covariates_adaptative_block_learning(
        rec, gen, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
    )

    plot_multivariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=100
    )

    Flux.reset!(nn_model)
    for data in collect(loaderXtrain)[1]
        nn_model.([data])
    end

    prediction = Vector{Float32}()
    for data in collect(loaderXtest)[1]
        y = nn_model.([data])
        append!(prediction, y[1])
    end

    real_seriel = [x[1] for x in collect(loaderXtest)[1]]
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

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
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

    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=20, window_size=1000, K=10)

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(
        RNN(3 => 10, relu; init=Flux.glorot_uniform(; gain=1)),
        RNN(10 => 5, relu; init=Flux.glorot_uniform(; gain=1)),
        RNN(5 => 3, relu; init=Flux.glorot_uniform(; gain=1)),
    )
    gen = Chain(
        Dense(4, 32, identity; init=Flux.glorot_uniform(; gain=1)),
        Dense(32, 1, identity; init=Flux.glorot_uniform(; gain=1)),
    )

    start = 36000
    num_training_data = 1000
    loaderXtrain = Flux.DataLoader(
        [
            [df.MT_333[i], df.MT_334[i], df.MT_335[i]] for
            i in start:(start + num_training_data)
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        [
            [df.MT_333[i], df.MT_334[i], df.MT_335[i]] for
            i in (start + 1):(start + num_training_data + 1)
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test = 2000
    loaderXtest = Flux.DataLoader(
        [
            [df.MT_333[i], df.MT_334[i], df.MT_335[i]] for
            i in (start + num_training_data - 1):(start + num_training_data + num_test)
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loss = ts_covariates_adaptative_block_learning(
        rec, gen, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
    )

    ts_covariates_mse_learning(nn_model, Xₜ, Xₜ₊₁, hparams)

    plot_multivariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=100
    )
end

@test_experiments "syntetic data Fernando" begin
    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=1000, window_size=120, K=10)

    #nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(LSTM(1 => 10), LSTM(10 => 10), LSTM(10 => 10), LSTM(10 => 1))
    gen = Chain(Dense(2, 32, relu), Dense(32, 32, relu), Dense(32, 1, identity))

    Xtrain = Vector{Float32}()
    Ytrain = Vector{Float32}()
    Xtest = Vector{Float32}()
    num_training_data = 120
    τ = 80
    for _ in 1:500
        x = syntetic_data.(1:(num_training_data + τ), 0.5)
        append!(Xtrain, Float32.(x[1:num_training_data]))
        append!(Ytrain, Float32.(x[2:(num_training_data + 1)]))
        append!(Xtest, Float32.(x[num_training_data:(num_training_data + τ)]))
    end

    loaderXtrain = Flux.DataLoader(
        Xtrain; batchsize=round(Int, num_training_data), shuffle=false, partial=false
    )

    loaderYtrain = Flux.DataLoader(
        Ytrain; batchsize=round(Int, num_training_data), shuffle=false, partial=false
    )

    num_test = 100
    loaderXtest = Flux.DataLoader(
        Xtest; batchsize=round(Int, num_test), shuffle=false, partial=false
    )

    loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)

    plot_univariate_ts_prediction(
        rec, gen, collect(loaderXtrain)[1], collect(loaderXtest)[1], hparams; n_average=10
    )
end


@test_experiments "testing " begin

    function generate_syntetic(range)
        data = []
        for x in range
            ϵ = Float32(rand(Normal(0.0f0, 1.0f0)))
            if rand(Bernoulli(0.5))
                y = 10 * cos(x - 0.5) + ϵ
            else
                y = 10 * sin(x - 0.5) + ϵ
            end
            push!(data, [x,y])
        end
        return data
    end


    range = -4:0.01:4
    data = generate_syntetic(range)

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=20, window_size=800, K=10)

    #nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))
    rec = Chain(RNN(1 => 2, relu), RNN(2 => 2, relu))
    gen = Chain(Dense(3, 16, relu), Dense(16, 1, identity))

    loaderXtrain = Flux.DataLoader(
        [Float32(y[2]) for y in data[1:end-1]];
        batchsize=round(Int, 10000),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        [Float32(y[2]) for y in data[2:end]];
        batchsize=round(Int, 10000),
        shuffle=false,
        partial=false,
    )

    losses = []
    @showprogress for i in 1:1000
        loss = ts_adaptative_block_learning(rec, gen, loaderXtrain, loaderYtrain, hparams)
        append!(losses, loss)
    end

    prediction = Vector{Float32}()
    Flux.reset!(rec)
    s = 0
    X_train = collect(loaderXtrain)[1]
    for data in X_train
        s = rec([data])
        y, _ = average_prediction(gen, s, n_average)
        append!(prediction, y[1])
    end

    ideal = X_train
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
        seriestype = :scatter
    )
    plot!(prediction; seriestype = :scatter)

end
