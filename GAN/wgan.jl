"""
HyperParamsWGAN

Hyper-parameters for the Wasserstein GAN (WGAN).

This structure defines hyper-parameters commonly used in training a Wasserstein GAN model. These hyper-parameters include noise and target models, data size, batch size, latent dimension, number of epochs, number of critic (discriminator) steps, clip value for weight clipping, and learning rates for the discriminator and generator.

# Fields
- `noise_model::Distribution`: The noise model used for generating noise data.
- `target_model::Distribution`: The target model representing the real data distribution.
- `data_size::Int`: The size of the dataset.
- `batch_size::Int`: The batch size used during training.
- `latent_dim::Int`: The dimension of the latent space for generating samples.
- `epochs::Int`: The number of training epochs.
- `n_critic::Int`: The number of critic (discriminator) steps per generator step.
- `clip_value::Float32`: The clip value for weight clipping in the critic.
- `lr_dscr::Float64`: The learning rate for the discriminator.
- `lr_gen::Float64`: The learning rate for the generator.

# Example
```julia
# Define hyper-parameters for Wasserstein GAN
hyperparameters = HyperParamsWGAN(
    noise_model = Normal(0.0f0, 1.0f0),
    target_model = Normal(23.0f0, 1.0f0),
    data_size = 10000,
    batch_size = 100,
    latent_dim = 1,
    epochs = 20,
    n_critic = 5,
    clip_value = 0.01,
    lr_dscr = 0.00005,
    lr_gen = 0.00005
)
```
"""
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
            wasserstein_loss_discr(discr(original_data), discr(fake_data))
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
    noise = cpu(
        rand!(
            hparams.noise_model,
            similar(original_data, (hparams.latent_dim, hparams.batch_size)),
        ),
    )
    loss = Dict()
    loss["gen"], grads = Flux.withgradient(gen) do gen
        fake_ = gen(noise)
        loss["discr"] = train_discr(discr, original_data, fake_, opt_discr, hparams)
        wasserstein_loss_gen(discr(fake_))
    end
    update!(opt_gen, gen, grads[1])
    return loss
end

"""
Train the Wasserstein GAN (WGAN) using provided discriminator and generator models.

This function trains a Wasserstein GAN model with a provided discriminator neural network (`dscr`), generator neural network (`gen`), and hyper-parameters (`hparams`). The hyper-parameters include settings for data loading, batch size, learning rates, training epochs, critic (discriminator) steps, clip value for weight clipping, and more.

# Arguments
- `dscr`: The neural network model for the discriminator.
- `gen`: The neural network model for the generator.
- `hparams::HyperParamsWGAN`: The hyper-parameters for training the Wasserstein GAN.

# Returns
- A tuple of two arrays containing the generator and discriminator losses during training.

# Example
```julia
# Define the discriminator, generator, and hyper-parameters
discriminator_model = ...
generator_model = ...
hyperparameters = HyperParamsWGAN(...)

# Train the Wasserstein GAN
losses = train_wgan(discriminator_model, generator_model, hyperparameters)
```
"""
function train_wgan(dscr, gen, hparams::HyperParamsWGAN, loader)
    #hparams = HyperParams()

    #=
    train_set = Float32.(rand(hparams.target_model, hparams.data_size))
    loader = gpu(
        Flux.DataLoader(
            train_set; batchsize=hparams.batch_size, shuffle=true, partial=false
        ),
    )
    =#

    #dscr = discriminator(hparams)
    #gen = gpu(generator(hparams))

    opt_dscr = Flux.setup(Flux.Adam(hparams.lr_dscr), dscr)
    opt_gen = Flux.setup(Flux.Adam(hparams.lr_gen), gen)
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
    return (losses_gen, losses_dscr)
end
