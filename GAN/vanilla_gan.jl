@with_kw struct HyperParamsVanillaGan
    noise_model = Normal(0.0f0, 1.0f0)
    target_model = Normal(23.0f0, 1.0f0)
    data_size::Int = 10000
    batch_size::Int = 100
    latent_dim::Int = 1
    epochs::Int = 1000
    lr_dscr::Float64 = 0.000001
    lr_gen::Float64 = 0.000002
    dscr_steps::Int = 5
    gen_steps::Int = 1
end

function discr_loss(real_output, fake_output)
    real_loss = mean(binarycrossentropy.(real_output, 1.0f0))
    fake_loss = mean(binarycrossentropy.(fake_output, 0.0f0))
    return (real_loss + fake_loss)
end

generator_loss(fake_output) = mean(binarycrossentropy.(fake_output, 1.0f0))

function train_discr(discr, original_data, fake_data, opt_discr)
    loss, grads = Flux.withgradient(discr) do discr
        discr_loss(discr(original_data), discr(fake_data'))
    end
    update!(opt_discr, discr, grads[1])
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, opt_gen, opt_discr, hparams::HyperParamsVanillaGan)
    noise = gpu(
        rand!(
            hparams.noise_model,
            similar(original_data, (hparams.batch_size, hparams.latent_dim)),
        ),
    )
    loss = Dict()
    for _ in 1:(hparams.gen_steps)
        loss["gen"], grads = Flux.withgradient(gen) do gen
            fake_ = gen(noise')
            generator_loss(discr(fake_'))
        end
        update!(opt_gen, gen, grads[1])
    end

    for _ in 1:(hparams.dscr_steps)
        fake_ = gen(noise')
        loss["discr"] = train_discr(discr, original_data, fake_, opt_discr)
    end

    loss["gen"], grads = Flux.withgradient(gen) do gen
        fake_ = gen(noise')
        loss["discr"] = train_discr(discr, original_data, fake_, opt_discr)
        generator_loss(discr(fake_'))
    end
    update!(opt_gen, gen, grads[1])
    return loss
end

function train_vanilla_gan(dscr, gen, hparams::HyperParamsVanillaGan)
    #hparams = HyperParamsVanillaGan()

    train_set = Float32.(rand(hparams.target_model, hparams.data_size))
    loader = gpu(
        Flux.DataLoader(
            train_set; batchsize=hparams.batch_size, shuffle=true, partial=false
        ),
    )

    #dscr = discriminator(hparams)
    #gen = gpu(generator(hparams))

    opt_dscr = Flux.setup(Flux.Adam(hparams.lr_dscr), dscr)
    opt_gen = Flux.setup(Flux.Adam(hparams.lr_gen), gen)

    # Training
    losses_gen = []
    losses_dscr = []
    train_steps = 0
    @showprogress for epoch in 1:(hparams.epochs)
        for x in loader
            loss = train_gan(gen, dscr, x, opt_gen, opt_dscr, hparams)
            push!(losses_gen, loss["gen"])
            push!(losses_dscr, loss["discr"])
            train_steps += 1
        end
    end
    return (losses_gen, losses_dscr)
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
