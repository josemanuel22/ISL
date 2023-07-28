using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, binarycrossentropy
using Images
using MLDatasets
using Statistics
using Parameters: @with_kw
using Random
using Printf
using CUDA
using Zygote

if has_cuda()# Check if CUDA is available
    @info "CUDA is on"
    using CuArrays: CuArrays# If CUDA is available, import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw struct HyperParams
    data_size::Int = 10000
    batch_size::Int = 100
    latent_dim::Int = 1
    epochs::Int = 20
    verbose_freq::Int = 1000
    lr_dscr::Float64 = 0.000001
    lr_gen::Float64 = 0.000002

    dscr_steps::Int = 5
    gen_steps::Int = 1
end

function real_model(ϵ)
    return rand(LogNormal(3,1))
end

function generator(args)
    return gpu(Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1)))
end

function discriminator(args)
    return gpu(Chain(Dense(100, 200, tanh), Dense(200, 200, tanh), Dense(200, 1, σ)))
end

function discr_loss(real_output, fake_output)
    real_loss = mean(binarycrossentropy.(real_output, 1.0f0))
    fake_loss = mean(binarycrossentropy.(fake_output, 0.0f0))
    return (real_loss + fake_loss)
end

generator_loss(fake_output) = mean(binarycrossentropy.(fake_output, 1.0f0))

function train_discr(discr, original_data, fake_data, opt_discr)
    ps = Flux.params(discr)
    loss, back = Zygote.pullback(ps) do
        discr_loss(discr(original_data), discr(fake_data'))
    end
    grads = back(1.0f0)
    update!(opt_discr, ps, grads)
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, opt_gen, opt_discr, hparams)
    noise = gpu(randn!(similar(original_data, (hparams.batch_size, hparams.latent_dim))))
    loss = Dict()
    ps = Flux.params(gen)
    for _ in 1:hparams.dscr_steps
        loss["gen"], back = Zygote.pullback(ps) do
            fake_ = gen(noise')
            generator_loss(discr(fake_'))
        end
    end
    grads = back(1.0f0)
    update!(opt_gen, ps, grads)

    for _ in 1:hparams.gen_steps
        Zygote.pullback(ps) do
            fake_ = gen(noise')
            loss["discr"] = train_discr(discr, original_data, fake_, opt_discr)
        end
    end
    grads = back(1.0f0)
    update!(opt_gen, ps, grads)

    loss["gen"], back = Zygote.pullback(ps) do
        fake_ = gen(noise')
        loss["discr"] = train_discr(discr, original_data, fake_, opt_discr)
        generator_loss(discr(fake_'))
    end
    grads = back(1.0f0)
    update!(opt_gen, ps, grads)
    return loss
end

function train()
    hparams = HyperParams()

    train_set = real_model.(rand(Float64, hparams.data_size))
    loader = gpu(Flux.DataLoader(
        train_set; batchsize=hparams.batch_size, shuffle=true, partial=false
    ))

    dscr = discriminator(hparams)

    gen = gpu(generator(hparams))

    opt_dscr = ADAM(hparams.lr_dscr)
    opt_gen = ADAM(hparams.lr_gen)

    # Training
    losses_gen = []
    losses_dscr = []
    train_steps = 0
    for ep in 1:(hparams.epochs)
        if train_steps % hparams.verbose_freq == 0
            @info "Epoch $ep"
        end
        for x in loader
            # Update discriminator and generator
            loss = train_gan(gen, dscr, x, opt_gen, opt_dscr, hparams)
            push!(losses_gen, loss["gen"])
            push!(losses_dscr, loss["discr"])
            train_steps += 1
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
