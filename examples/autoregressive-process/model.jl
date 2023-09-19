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
        nn_model([batch_Xₜ[1]]) # Warm up recurrent model on first observation
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


function ts_covariates_adaptative_block_learning(nn_model, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for _ in 1:(hparams.epochs)
        j = 0
        Flux.reset!(nn_model)
        nn_model(Xₜ[1])
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(hparams.K + 1)
            for i in (1:(hparams.window_size))
                xₖ = rand(hparams.noise_model, hparams.K)
                nn_cp = deepcopy(nn)
                yₖ = nn_cp.([vcat(x, Xₜ[j+i][2:end]) for x in xₖ])
                aₖ += generate_aₖ(cat(yₖ..., dims=2),  Xₜ₊₁[j + i][1])
                nn(Xₜ[j+i])
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

function ts_covariates_mse_learning(nn_model, Xₜ, Xₜ₊₁, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for epoch in (1:(hparams.epochs))
        Flux.reset!(nn_model)   # Reset the hidden state of the RNN
        loss, grads = Flux.withgradient(nn_model) do nn
            nn(Xₜ[1])    # Warm-up the model
            sum(Flux.Losses.mse.([nn(x)[1] for x in Xₜ[1:end]], Xₜ₊₁[1:end]))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end

function ts_adaptative_block_learning(nn_model, , tsₖ, s, hparams)
    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for _ in 1:(hparams.epochs)
        j = 0
        Flux.reset!(nn_model)
        nn_model(insert!([x[1] for x in tsₖ], s, ts₁[1]))
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(hparams.K + 1)
            for i in (1:(hparams.window_size))
                xₖ = rand(hparams.noise_model, hparams.K)
                nn_cp = deepcopy(nn)
                yₖ = nn_cp(insert!([[x[j + i] for x in tsₖ]], s, xₖ))
                aₖ += generate_aₖ(yₖ, [ts₁[j + i + 1], [x[j + i + 1] for x in tsₖ]])
                nn([ts₁[j + i], [x[j + i] for x in tsₖ]])
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

@test_experiments "testing" begin
    csv_file_path = "/Users/jmfrutos/Desktop/LD2011_2014.txt"

    df = CSV.File(
        csv_file_path;
        delim=';',
        header=true,
        decimal=',',
        types=Dict(
            "MT_330" => Float32,
            "MT_331" => Float32,
            "MT_332" => Float32,
            "MT_333" => Float32,
            "MT_334" => Float32,
            "MT_335" => Float32,
        ),
    )

    hparams = HyperParamsTS(; seed=1234, η=1e-3, epochs=1, window_size=1000, K=5)

    nn_model = Chain(RNN(1 => 32, relu), RNN(32 => 32, relu), Dense(32 => 1, identity))

    loaderXtrain = Flux.DataLoader(
        df.MT_334[1:40000]; batchsize=round(Int, 40000), shuffle=false, partial=false
    )

    loaderYtrain = Flux.DataLoader(
        df.MT_334[2:40001]; batchsize=round(Int, 40000), shuffle=false, partial=false
    )

    res = Vector{Float32}()
    @showprogress for i in 1:200
        loss = ts_adaptative_block_learning(nn_model, loaderXtrain, loaderYtrain, hparams)
        append!(res, loss)
    end

    Flux.reset!(nn_model)
    for data in collect(loaderXtrain)[1]
        nn_model.([[data]])
    end

    prediction = Vector{Float32}()
    for data in df.MT_334[39900:40100]
        y = nn_model.([[data]])
        append!(prediction, y[1])
    end
end

@test_experiments "testing" begin
    csv_file_path = "/Users/jmfrutos/Desktop/LD2011_2014.txt"

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
            [df.MT_333[40000+i], df.MT_334[40000+i], df.MT_335[40000+i], df.MT_336[40000+i], df.MT_338[40000+i]] for
            i in 1:length(df.MT_335[40000:40200])
        ];
        batchsize=round(Int, 40000),
        shuffle=false,
        partial=false,
    )

    #loss = ts_covariates_mse_learning(nn_model, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams)

    loss = ts_covariates_adaptative_block_learning(nn_model, collect(loaderXtrain)[1], collect(loaderYtrain)[1], hparams)

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
