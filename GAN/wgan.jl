@with_kw struct HyperParamsWGAN
    noise_model = Normal(0.0f0, 1.0f0)
    target_model = Normal(23.0f0, 1.0f0)
    data_size::Int = 10000
    batch_size::Int = 100
    latent_dim::Int = 1
    epochs::Int = 20
    n_critic::Int = 5
    clip_value::Float32 = 0.01
    lr_dscr::Float64 = 0.00005
    lr_gen::Float64 = 0.00005
end

function real_model(ϵ)
    return rand(Normal(1.0f0, 2.0f0))
end

## Generator and Discriminator
function generator(args)
    return gpu(Chain(Dense(1, 5, tanh), Dense(5, 5, tanh), Dense(5, 1)))
end

function discriminator(args)
    return gpu(
        Chain(
            Dense(100, 200, tanh),
            Dense(200, 500, tanh),
            Dense(500, 500, relu),
            Dense(500, 1, σ),
        ),
    )
end

function wasserstein_loss_discr(real, fake)
    return -mean(real) + mean(fake)
end

function wasserstein_loss_gen(out)
    return -mean(out)
end

function train_discr(discr, original_data, fake_data, opt_discr, hparams::HyperParamsWGAN)
    loss = 0.0
    for i in 1:(hparams.n_critic)
        loss, grads = Flux.withgradient(discr) do discr
            wasserstein_loss_discr(discr(original_data), discr(fake_data'))
        end
        update!(opt_discr, discr, grads[1])
        for i in Flux.params(discr)
            i = clamp.(i, -hparams.clip_value, hparams.clip_value)
        end
    end
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, opt_gen, opt_discr, hparams::HyperParamsWGAN)
    noise = gpu(
        rand!(
            hparams.noise_model,
            similar(original_data, (hparams.batch_size, hparams.latent_dim)),
        ),
    )
    loss = Dict()
    loss["gen"], grads = Flux.withgradient(gen) do gen
        fake_ = gen(noise)
        loss["discr"] = train_discr(discr, original_data, fake_, opt_discr, hparams)
        wasserstein_loss_gen(discr(fake_'))
    end
    update!(opt_gen, gen, grads[1])
    return loss
end

function train_wgan(dscr, gen, hparams::HyperParamsWGAN)
    #hparams = HyperParams()

    train_set = Float32.(rand(hparams.target_model, hparams.data_size))
    loader = gpu(
        Flux.DataLoader(
            train_set; batchsize=hparams.batch_size, shuffle=true, partial=false
        ),
    )

    #dscr = discriminator(hparams)
    #gen = gpu(generator(hparams))

    opt_dscr = Flux.setup(Flux.RMSProp(hparams.lr_dscr), dscr)
    opt_gen = Flux.setup(Flux.RMSProp(hparams.lr_gen), gen)
    losses_gen = []
    losses_dscr = []

    train_steps = 0
    @showprogress for epoch in 1:(hparams.epochs)
        for x in loader
            loss = train_gan(gen, dscr, x, opt_gen, opt_dscr, hparams)
            train_steps += 1
            push!(losses_gen, loss["gen"])
            push!(losses_dscr, loss["discr"])
        end
    end
end

function plot_model(real_model, model, range)
    μ = 0.0f0
    stddev = 1.0f0
    x = rand(Normal(μ, stddev), 1000000)
    ϵ = rand(Float32, 1000000)
    y = real_model.(ϵ)
    histogram(y; bins=range)
    ŷ = model(x')
    return histogram!(ŷ'; bins=range)
end
