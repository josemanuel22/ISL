using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, binarycrossentropy
using Statistics
using Parameters: @with_kw
using Random
using CUDA
using Zygote
using Distributions
using ProgressMeter

include("./mmd.jl")

@with_kw struct HyperParams
    data_size::Int = 100
    batch_size::Int = 1
    latent_dim::Int = 1
    epochs::Int = 10000
    num_gen::Int = 1
    num_enc_dec::Int = 5
    lr_enc::Float64 = 1.0e-4
    lr_dec::Float64 = 1.0e-4
    lr_gen::Float64 = 1.0e-4

    lambda_AE::Float64 = 8.0
    target_param::Tuple{Float64,Float64} = (23.0, 1.0)
    noise_param::Tuple{Float64,Float64} = (0.0, 1.0)
    base::Float64 = 1.0
    sigma_list::Array{Float64,1} = [1.0, 2.0, 4.0, 8.0, 16.0] ./ base
end

function generator()
    return Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
end

function encoder()
    return Chain(Dense(1, 11), elu, Dense(11, 29), elu)
end

function decoder()
    return Chain(Dense(29, 11), elu, Dense(11, 1))
end

function real_model(ϵ)
    return rand(Normal(23.0f0, 1.0f0), (hparams.batch_size, 1))
end

function data_sampler(hparams, target)
    return rand(Normal(1.0f0, target[2]), (hparams.batch_size, 1))
end

# Initialize models and optimizers
function train_mmd_gan()
    hparams = HyperParams()

    gen = generator()
    enc = encoder()
    dec = decoder()

    # Optimizers
    gen_opt = Flux.setup(Flux.Adam(hparams.lr_gen), gen)
    enc_opt = Flux.setup(Flux.Adam(hparams.lr_enc), enc)
    dec_opt = Flux.setup(Flux.Adam(hparams.lr_dec), dec)

    # Training
    losses_gen = []
    losses_dscr = []
    loader_target = Flux.DataLoader(
        real_model(rand(Float32));
        batchsize=hparams.batch_size,
        shuffle=true,
        partial=false
    )
    loader_noise = Flux.DataLoader(
        real_model(rand(Float32));
        batchsize=hparams.batch_size,
        shuffle=true,
        partial=false
    )
    @showprogress for epoch in 1:(hparams.epochs)
        for (target, noise) in zip(loader_target, loader_noise)
            for _ in 1:(hparams.num_enc_dec)
                loss, grads = Flux.withgradient(enc, dec) do enc, dec
                    encoded_target = enc(target)
                    decoded_target = dec(encoded_target)
                    L2_AE_target = Flux.mse(decoded_target, target)
                    transformed_noise = gen(noise)
                    encoded_noise = enc(transformed_noise)
                    decoded_noise = dec(encoded_noise)
                    L2_AE_noise = Flux.mse(decoded_noise, transformed_noise)
                    MMD = mix_rbf_mmd2(encoded_target, encoded_noise, hparams.sigma_list)
                    MMD = relu(MMD)
                    L_MMD_AE =
                        -1.0 *
                        (sqrt(MMD) - hparams.lambda_AE * (L2_AE_noise + L2_AE_target))
                end
                update!(enc_opt, enc, grads[1])
                update!(dec_opt, dec, grads[2])
                push!(losses_dscr, loss)
            end
            for _ in 1:(hparams.num_gen)
                loss, grads = Flux.withgradient(gen) do gen
                    #target = data_sampler(hparams, hparams.target_param)
                    #noise = data_sampler(hparams, hparams.noise_param)
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
    end
end

function plot_results(n_samples, range_)
    target = collect(Iterators.flatten(real_model.(rand(n_samples))))
    transformed_noise = vec(gen(rand(Normal(0.0f0, 1.0f0), n_samples)'))
    histogram(target; bins=range_)
    return histogram!(transformed_noise; bins=range_)
end
