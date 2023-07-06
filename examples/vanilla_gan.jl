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

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    import CuArrays		# If CUDA is available, import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw struct HyperParams
    data_size::Int = 10000
    batch_size::Int = 100
    latent_dim::Int = 1
    epochs::Int = 10000
    verbose_freq::Int = 1000
    output_x::Int = 6        # No. of sample images to concatenate along x-axis
    output_y::Int = 6        # No. of sample images to concatenate along y-axis
    lr_dscr::Float64 = 0.00002
    lr_gen::Float64 = 0.00002
end

function generator(args)
    return Chain(Dense(1, 20, tanh), Dense(20, 1)) |> gpu
end

function discriminator(args)
    return Chain(Dense(1, 20), x -> leakyrelu.(x,0.2f0), Dense(20, 1, σ)) |> gpu
end

function load_data(hparams)
    function realModel(ϵ)
        μ₁=-20; σ₁ = 2;
        μ₂=0; σ₂ = 1;
        μ₃=40; σ₃ = 2;
        if ϵ < 0.3
            return rand(Normal(μ₁, σ₁))
        elseif ϵ < 0.7
            return rand(Normal(μ₂, σ₂))
        else
            return rand(Normal(μ₃, σ₃))
    end

    data = collect(partition(realModel.(rand(Float64, hparams.data_size)), hparams.batch_size))
    return data
end

# Loss functions
function discr_loss(real_output, fake_output)
    real_loss = mean(binarycrossentropy.(real_output, 1f0))
    fake_loss = mean(binarycrossentropy.(fake_output, 0f0))
    return (real_loss + fake_loss)
end

generator_loss(fake_output) = mean(binarycrossentropy.(fake_output, 1f0))

function train_discr(discr, original_data, fake_data, opt_discr)
    ps = Flux.params(discr)
    loss, back = Zygote.pullback(ps) do
                      discr_loss(discr(original_data'), discr(fake_data))
    end
    grads = back(1f0)
    update!(opt_discr, ps, grads)
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, opt_gen, opt_discr, hparams)
    noise = randn!(similar(original_data, (hparams.batch_size, hparams.latent_dim))) |> gpu
    loss = Dict()
    ps = Flux.params(gen)
    loss["gen"], back = Zygote.pullback(ps) do
                          fake_ = gen(noise')
                          loss["discr"] = train_discr(discr, original_data, fake_, opt_discr)
                          generator_loss(discr(fake_))
    end
    grads = back(1f0)
    update!(opt_gen, ps, grads)
    return loss
end

function train()
    hparams = HyperParams()

    data = load_data(hparams)

    # Discriminator
    dscr = discriminator(hparams)

    # Generator
    gen =  generator(hparams) |> gpu

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr)
    opt_gen = ADAM(hparams.lr_gen)

    # Training
    losses_gen = []
    losses_dscr = []
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for x in data
            # Update discriminator and generator
            loss = train_gan(gen, dscr, x, opt_gen, opt_dscr, hparams)

            #if train_steps % hparams.verbose_freq == 0
            #    @info("Train step $(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"])")
                # Save generated fake image
            #    x = rand(Normal(μ, stddev), 100000)
            #    ϵ = rand(Float64, 100000)
            #    y = realModel.(ϵ)
            #    histogram(y, bins=-50:1:70, label = "distrubución real", title = "proxy error cuadrático")
            #    ŷ = gen(x')
            #    histogram!(ŷ', bins=-50:1:70, label = "distrubución aproximada")
            #    #output_image = create_output_image(gen, fixed_noise, hparams)
            #    #save(@sprintf("output/gan_steps_%06d.png", train_steps), output_image)
            #end
            push!(losses_gen, loss["gen"])
            push!(losses_dscr, loss["discr"])
            train_steps += 1
        end
    end
end

x = rand(Normal(μ, stddev), 100000)
ϵ = rand(Float64, 100000)
y = realModel.(ϵ)
histogram(y, bins=-50:1:70, label = "distrubución real", title = "proxy error cuadrático")
ŷ = gen(x')
histogram!(ŷ', bins=-50:1:70, label = "distrubución aproximada")
