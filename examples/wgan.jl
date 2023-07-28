using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy
using Images
using MLDatasets
using Statistics
using Parameters: @with_kw
using Random
using Printf
using CUDA
using Zygote

@with_kw struct HyperParams
    data_size::Int = 10000
    batch_size::Int = 100
    latent_dim::Int = 1
    epochs::Int = 100
    n_critic::Int = 5
    clip_value::Float32 = 0.01
    lr_dscr::Float64 =  0.00005
    lr_gen::Float64 =   0.00005
end

function real_model(ϵ)
    rand(Pareto(1, 2))
end

## Generator and Discriminator
function generator(args)
    return Chain(Dense(1, 5, tanh), Dense(5, 5, tanh), Dense(5, 1)) |> gpu
end

function discriminator(args)
    return Chain(Dense(100, 200, tanh), Dense(200, 500, tanh), Dense(500, 500, relu), Dense(500, 1, σ))  |> gpu
end

function wasserstein_loss_discr(real, fake)
    return -mean(real) + mean(fake)
end

function wasserstein_loss_gen(out)
    return -mean(out)
end

function train_discr(discr, original_data, fake_data, opt_discr, hparams)
    ps = Flux.params(discr)
    loss = 0.0
    for i in 1:hparams.n_critic
        loss, back = Zygote.pullback(ps) do
                        wasserstein_loss_discr(discr(original_data), discr(fake_data'))
        end
        grads = back(1f0)
        update!(opt_discr, ps, grads)
        for i in ps
            i = clamp.(i, -hparams.clip_value, hparams.clip_value)
        end
    end
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, opt_gen, opt_discr, hparams)
    noise = randn!(similar(original_data, (hparams.latent_dim, hparams.batch_size))) |> gpu
    loss = Dict()
    ps = Flux.params(gen)
    loss["gen"], back = Zygote.pullback(ps) do
                          fake_ = gen(noise)
                          loss["discr"] = train_discr(discr, original_data, fake_, opt_discr, hparams)
                          wasserstein_loss_gen(discr(fake_'))
    end
    grads = back(1f0)
    update!(opt_gen, ps, grads)
    return loss
end

function train()
    hparams = HyperParams()

    train_set = realModel.(rand(Float64, hparams.data_size))
    loader = Flux.DataLoader(train_set, batchsize = hparams.batch_size, shuffle = true, partial=false) |> gpu

    dscr = discriminator(hparams)
    gen =  generator(hparams) |> gpu

    opt_dscr = RMSProp(hparams.lr_dscr)
    opt_gen = RMSProp(hparams.lr_gen)
    losses_gen = []
    losses_dscr = []

    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        if train_steps % hparams.verbose_freq == 0
            @info "Epoch $ep"
        end
        for x in loader
            # Update discriminator and generator
            loss = train_gan(gen, dscr, x, opt_gen, opt_dscr, hparams)
            train_steps += 1
            push!(losses_gen, loss["gen"])
            push!(losses_dscr, loss["discr"])
        end
    end
end

function plot_model(real_model, model, range)
    μ = 0; stddev = 1
    x = rand(Normal(μ, stddev), 1000000)
    ϵ = rand(Float64, 1000000)
    y = real_model.(ϵ)
    histogram(y; bins=range)
    ŷ = model(x')
    histogram!(ŷ'; bins=range)
end
