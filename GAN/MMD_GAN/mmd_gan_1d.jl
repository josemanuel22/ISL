
include("./mmd.jl")

"""
    HyperParamsMMD1D

Hyper-parameters for the MMD GAN 1D.

```julia
@with_kw struct HyperParamsMMD1D
    target_model = Normal(23.0f0, 1.0f0)
    noise_model = Normal(0.0f0, 1.0f0)

    data_size::Int = 1000
    batch_size::Int = 100
    latent_dim::Int = 1
    epochs::Int = 1000
    num_gen::Int = 1
    num_enc_dec::Int = 1
    lr_enc::Float64 = 1.0e-10
    lr_dec::Float64 = 1.0e-10
    lr_gen::Float64 = 1.0e-10

    lambda_AE::Float64 = 8.0

    base::Float64 = 1.0
    sigma_list::Array{Float64,1} = [1.0, 2.0, 4.0, 8.0, 16.0] ./ base
end
```
"""
@with_kw struct HyperParamsMMD1D
    target_model = Normal(23.0f0, 1.0f0)
    noise_model = Normal(0.0f0, 1.0f0)

    data_size::Int = 1000
    batch_size::Int = 100
    latent_dim::Int = 1
    epochs::Int = 1000
    num_gen::Int = 1
    num_enc_dec::Int = 1
    lr_enc::Float64 = 1.0e-3
    lr_dec::Float64 = 1.0e-3
    lr_gen::Float64 = 1.0e-3

    lambda_AE::Float64 = 8.0

    base::Float64 = 1.0
    sigma_list::Array{Float64,1} = [1.0, 2.0, 4.0, 8.0, 16.0] ./ base
end


"""
    train_mmd_gan_1d(enc, dec, gen, hparams::HyperParamsMMD1D)

Train the MMD GAN 1D. ``enc`` is the encoder, ``dec`` is the decoder used as dscr,
and ``gen`` is the generator. hparams is the hyper-parameters.
"""
function train_mmd_gan_1d(enc, dec, gen, hparams::HyperParamsMMD1D)
    #hparams = HyperParams()

    #gen = generator()
    #enc = encoder()
    #dec = decoder()

    # Optimizers
    gen_opt = Flux.setup(Flux.Adam(hparams.lr_gen), gen)
    enc_opt = Flux.setup(Flux.Adam(hparams.lr_enc), enc)
    dec_opt = Flux.setup(Flux.Adam(hparams.lr_dec), dec)

    # Training
    losses_gen = []
    losses_dscr = []
    @showprogress for epoch in 1:(hparams.epochs)
        for _ in 1:(hparams.num_enc_dec)
            target = Float32.(rand(hparams.target_model, hparams.batch_size))
            noise = Float32.(rand(hparams.noise_model,  hparams.batch_size))
            loss, grads = Flux.withgradient(enc, dec) do enc, dec
                Flux.reset!(enc)
                Flux.reset!(dec)
                encoded_target = enc(target')
                decoded_target = dec(encoded_target)
                L2_AE_target = Flux.mse(decoded_target', target)
                transformed_noise = gen(noise')
                encoded_noise = enc(transformed_noise)
                decoded_noise = dec(encoded_noise)
                L2_AE_noise = Flux.mse(decoded_noise, transformed_noise)
                MMD = mix_rbf_mmd2(encoded_target, encoded_noise, hparams.sigma_list)
                MMD = relu(MMD)
                L_MMD_AE =
                    -1.0 * (sqrt(MMD) - hparams.lambda_AE * (L2_AE_noise + L2_AE_target))
            end
            update!(enc_opt, enc, grads[1])
            update!(dec_opt, dec, grads[2])
            push!(losses_dscr, loss)
        end
        for _ in 1:(hparams.num_gen)
            target = Float32.(rand(hparams.target_model, hparams.batch_size))
            noise = Float32.(rand(hparams.noise_model,  hparams.batch_size))
            loss, grads = Flux.withgradient(gen) do gen
                Flux.reset!(gen)
                encoded_target = enc(target')
                encoded_noise = enc(gen(noise'))
                MMD = sqrt(
                    relu(mix_rbf_mmd2(encoded_target, encoded_noise, hparams.sigma_list)),
                )
            end
            update!(gen_opt, gen, grads[1])
            push!(losses_gen, loss)
        end
    end
    return (losses_gen, losses_dscr)
end
