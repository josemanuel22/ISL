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

function ts_covariates_mse_learning(nn_model, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for epoch in (1:(hparams.epochs))
        Flux.reset!(nn_model)
        loss, grads = Flux.withgradient(nn_model) do nn
            nn(Xₜ[1])
            sum(Flux.Losses.mse.([nn(x)[1] for x in Xₜ[1:end]], Xₜ₊₁[1:end]))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end

function ts_mse_learning(nn_model, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for epoch in (1:(hparams.epochs))
        Flux.reset!(nn_model)
        loss, grads = Flux.withgradient(nn_model) do nn
            nn([Xₜ[1]]')
            sum(Flux.Losses.mse.([nn([x]')[1] for x in Xₜ[1:(end - 1)]], Xₜ₊₁[1:end]))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end

@test_experiments "testing AutoRegressive Model" begin
    ar_hparams = ARParams(;
        ϕ=[0.7f0, 0.2f0, 0.1f0],
        x₁=rand(Normal(0.0f0, 0.5f0)),
        proclen=10000,
        noise=Normal(0.0f0, 0.5f0),
    )
    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=1000, window_size=200, K=5)

    nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))

    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
        hparams, ar_hparams
    )

    loss = ts_adaptative_block_learning(nn_model, loaderXtrain, loaderYtrain, hparams)

    plot_ts(nn_model, loaderXtrain, loaderYtrain, hparams)

    loss = ts_mse_learning(
        nn_model, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
    )

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

    res = []
    for i in (1:(hparams.epochs - 1))
        Flux.reset!(nn_model)
        for data in collect(loaderXtrain)[i]
            nn_model.([[data]])
        end

        prediction = Vector{Float32}()
        for data in collect(loaderXtest)[i]
            y = nn_model.([[data]])
            append!(prediction, y[1])
        end
        r, _ = yule_walker(Float64.(prediction); order=3)
        push!(res, r)
    end

    plot(prediction; seriestype=:scatter)
    plot!(Float32.(collect(loaderXtest)[1]); seriestype=:scatter)

    ND(Float32.(collect(loaderXtest)[1])[1:200], prediction[1:200])

    RMSE(Float32.(collect(loaderXtest)[1])[1:200], prediction[1:200])

    yule_walker(Float64.(collect(loaderYtest)[2]); order=3)
    yule_walker(Float64.(prediction); order=3)

    y = collect(loaderYtest)[1]
    Flux.reset!(nn_model)
    nn_model.([collect(loaderXtest)[1]'])
    collect(loaderYtrain)[1]

    get_watson_durbin_test(y, ŷ)
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

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=100, window_size=1000, K=5)

    nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))

    num_training_data = 10000

    loaderXtrain = Flux.DataLoader(
        [
            [
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
            ] for i in 1:length(df1["Power(MW)"][1:num_training_data])
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        [
            [
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
            ] for i in 1:length(df1["Power(MW)"][2:(num_training_data + 1)])
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    num_test_data = 200
    loaderXtest = Flux.DataLoader(
        [
            [
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
                df1["Power(MW)"][i],
            ] for i in
            1:length(
                df1["Power(MW)"][num_training_data:(num_training_data + num_test_data)]
            )
        ];
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loss = ts_covariates_adaptative_block_learning(
        nn_model, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
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
end

@test_experiments "testing 30 years european wind generation" begin
    #=
    https://www.nrel.gov/grid/solar-power-data.html
    =#

    csv1 = "examples/time_series_predictions/data/30-years-of-european-wind-generation/TS.CF.N2.30yr.csv"

    df1 = CSV.File(
        csv1;
        delim=',',
        header=true,
        decimal='.',
        stripwhitespace=true,
        types=Dict("AT11" => Float32),
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=100, window_size=1000, K=5)

    nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))

    num_training_data = 10000

    loaderXtrain = Flux.DataLoader(
        map(
            x -> Float32.(x),
            cat(
                [
                    [df1.FR43[i], df1.FR82[i], df1.FR71[i], df1.FR72[i], df1.FR26[i]] for
                    i in 1:length(df1[1:num_training_data])
                ];
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
                    [df1.FR43[i], df1.FR82[i], df1.FR71[i], df1.FR72[i], df1.FR26[i]] for
                    i in 1:length(df1[2:(num_training_data + 1)])
                ];
                dims=1,
            ),
        );
        batchsize=round(Int, num_training_data),
        shuffle=false,
        partial=false,
    )

    loss = ts_covariates_adaptative_block_learning(
        nn_model, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
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
    csv_file_path = "/AdaptativeBlockLearning/examples/time_series_predictions"

    df = CSV.File(
        csv_file_path;
        delim=';',
        header=true,
        decimal=',',
        types=Dict(
            "MT_331" => Float32,
            "MT_332" => Float32,
            "MT_333" => Float32,
            "MT_334" => Float32,
            "MT_335" => Float32,
            "MT_336" => Float32,
            "MT_338" => Float32,
        ),
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=100, window_size=100, K=5)

    nn_model = Chain(RNN(5 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))

    loaderXtrain = Flux.DataLoader(
        [
            [df.MT_333[i], df.MT_334[i], df.MT_335[i], df.MT_336[i], df.MT_338[i]] for
            i in 1:length(df.MT_335[1:40000])
        ];
        batchsize=round(Int, 40000),
        shuffle=false,
        partial=false,
    )

    loaderYtrain = Flux.DataLoader(
        [
            [df.MT_333[i], df.MT_334[i], df.MT_335[i], df.MT_336[i], df.MT_338[i]] for
            i in 1:length(df.MT_335[1:40000])
        ];
        batchsize=round(Int, 200),
        shuffle=false,
        partial=false,
    )

    loaderXtest = Flux.DataLoader(
        [
            [
                df.MT_333[40000 + i],
                df.MT_334[40000 + i],
                df.MT_335[40000 + i],
                df.MT_336[40000 + i],
                df.MT_338[40000 + i],
            ] for i in 1:length(df.MT_335[40000:40200])
        ];
        batchsize=round(Int, 40000),
        shuffle=false,
        partial=false,
    )

    loss = ts_covariates_adaptative_block_learning(
        nn_model, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams
    )

    res = []
    Flux.reset!(nn_model)
    for data in collect(loaderXtrain)[1]
        y = nn_model.([data])
        append!(prediction, y[1])
    end

    prediction = Vector{Float32}()
    for data in collect(loaderXtest)[1]
        y = nn_model.([data])
        append!(prediction, y[1])
    end

    prediction = Vector{Float32}()
    data = collect(loaderXtest)[1]
    y = nn_model(data[1])
    append!(prediction, y[1])
    for i in 1:200
        y = nn_model(vcat(y, data[i][2:end]))
        append!(prediction, y[1])
    end
    r, _ = yule_walker(Float64.(prediction); order=3)
    push!(res, r)
end
