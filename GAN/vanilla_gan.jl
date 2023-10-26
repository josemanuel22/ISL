
"""
HyperParamsVanillaGan

Hyper-parameters for the Vanilla GAN (Generative Adversarial Network).

This structure defines hyper-parameters commonly used in training a Vanilla GAN model. These hyper-parameters include noise and target models, data size, batch size, latent dimension, number of epochs, learning rates for the discriminator and generator, and the number of steps for the discriminator and generator updates.

# Fields
- `noise_model::Distribution`: The noise model used for generating noise data.
- `target_model::Distribution`: The target model representing the real data distribution.
- `data_size::Int`: The size of the dataset.
- `batch_size::Int`: The batch size used during training.
- `latent_dim::Int`: The dimension of the latent space for generating samples.
- `epochs::Int`: The number of training epochs.
- `lr_dscr::Float64`: The learning rate for the discriminator.
- `lr_gen::Float64`: The learning rate for the generator.
- `dscr_steps::Int`: The number of discriminator steps per training iteration.
- `gen_steps::Int`: The number of generator steps per training iteration.

# Example
```julia
# Define hyper-parameters for Vanilla GAN
hyperparameters = HyperParamsVanillaGan(
    noise_model = Normal(0.0f0, 1.0f0),
    target_model = Normal(23.0f0, 1.0f0),
    data_size = 10000,
    batch_size = 100,
    latent_dim = 1,
    epochs = 1000,
    lr_dscr = 0.000001,
    lr_gen = 0.000002,
    dscr_steps = 5,
    gen_steps = 1
)
```
"""
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

function train_gan(
    gen, discr, original_data, opt_gen, opt_discr, hparams::HyperParamsVanillaGan
)
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

"""
Train the Vanilla GAN (Generative Adversarial Network).

This function trains a Vanilla GAN model with a provided discriminator neural network (`dscr`), generator neural network (`gen`), and hyper-parameters (`hparams`). The hyper-parameters include settings for data loading, batch size, learning rates, and training epochs.

# Arguments
- `dscr`: The neural network model for the discriminator.
- `gen`: The neural network model for the generator.
- `hparams::HyperParamsVanillaGan`: The hyper-parameters for training the Vanilla GAN.

# Returns
- A tuple of two arrays containing the generator and discriminator losses during training.

# Example
```julia
# Define the discriminator, generator, and hyper-parameters
discriminator_model = ...
generator_model = ...
hyperparameters = HyperParamsVanillaGan(...)

# Train the Vanilla GAN
losses = train_vanilla_gan(discriminator_model, generator_model, hyperparameters)
```
"""
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
