using AdaptativeBlockLearning

using Flux
using StatsBase
using Distributions: Normal, rand
using HypothesisTests: pvalue, ChisqTest
using ProgressMeter
using LoopVectorization
using Parameters: @with_kw

model = gpu(Chain(Dense(1 => 20, tanh), Dense(20 => 1)))

function realModel(ϵ)
    μ₁ = -2
    σ₁ = 1
    μ₃ = 3
    σ₃ = 1
    if ϵ < 0.4
        return rand(Normal(μ₁, σ₁))
    else
        return rand(Normal(μ₃, σ₃))
    end
end

#Learning with custom loss


μ = 0;
stddev = 1;
η = 0.5;
num_epochs = 2000;
n_samples = 200;
K = 20;

@with_kw struct HyperParams
    μ::Float64 = 0
    stddev::Float64 = 1
    η::Float64 = 0.5
    epochs::Int = 2000
    samples::Int = 200
    K::Int = 20
end;

hparams = HyperParams()

losses = []
optim = Flux.setup(Flux.Adam(hparams.η), model)
@showprogress for epoch in 1:hparams.epochs
    loss, grads = Flux.withgradient(model) do m
        aₖ = zeros(hparams.K + 1)
        for _ in 1:hparams.samples
            x = rand(Normal(hparams.μ, hparams.stddev), hparams.K)
            yₖ = m(x')
            y = real_model(rand(Float64))
            aₖ += @avx generate_aₖ(yₖ, y)
        end
        #jensen_shannon_∇(aₖ ./ sum(aₖ))
        scalar_diff(aₖ ./ sum(aₖ))
    end
    Flux.update!(optim, model, grads[1])
    push!(losses, loss)
end;

#Learning with classical mse loss
η = 0.01;
num_epochs = 100000;
optim = Flux.setup(Flux.Adam(η), model)
losses_mse = []
for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        x = rand(Normal(μ, stddev), 1)
        ŷ = m(x')
        y = realModel(rand(Float64))
        Flux.mse(ŷ, y)
    end
    Flux.update!(optim, model, grads[1])
    push!(losses_mse, loss)
end;

function plot(real_model, model, range)
    x = rand(Normal(μ, stddev), 100000)
    ϵ = rand(Float64, 100000)
    y = real_model.(ϵ)
    histogram(y; bins=range)
    ŷ = model(x')
    histogram!(ŷ'; bins=range)
end

function get_statistic_model(real_model, model)
    windows = get_window_of_Aₖ(model, real_model, hparams.K, hparams.samples)
    aₖ = [count(x -> x == i, windows) for i in 0:hparams.K]
    histogram(windows; bins=0:1:(hparams.K + 1))

end
