using ISL
using GAN

include("../utils.jl")

"""
We showcase its qualitative advantage over classical generative models for independent time
settings, capturing multimodal and heavy-tailed distributions. ISL outperforms vanilla-GAN,
Wasserstein-GAN, and MMD-GAN baselines. Our experimental settings for this section are
grounded in the study by https://chunliangli.github.io/docs/dltp17gan.pdf
"""
@test_experiments "vanilla_gan" begin

    # Define an experiment test with noise model N(0,1)
    @test_experiments "Origin N(0,1)" begin
        # Initialize the noise model as a normal distribution N(0,1)
        noise_model = Normal(0.0f0, 1.0f0)
        n_samples = 10000

        # Define an experiment to transform N(0,1) to N(23,1)
        @test_experiments "N(0,1) to N(23,1)" begin
            # Generator: a neural network with ELU activation
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))

            # Discriminator: a neural network with ELU activation and σ activation function in the last layer
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            # Target model composed of a mixture of models
            target_model = Normal(23.0f0, 1.0f0)

            # Parameters for GAN training
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=1e4,
                lr_dscr=1e-4,
                lr_gen=1e-4,
                dscr_steps=4,
                gen_steps=0,
                noise_model=noise_model,
                target_model=target_model,
            )

            # Training the GAN
            train_vanilla_gan(dscr, gen, hparams)

            # Parameters for automatic invariant statistical loss
            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )

            # Preparing the training set and data loader
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            # Training using the automatic invariant statistical loss
            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "N(0,1) to Uniform(22,24)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Uniform(22.0f0, 24.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=1e4,
                lr_dscr=1e-4,
                lr_gen=1e-4,
                dscr_steps=3,
                gen_steps=0,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)

            plot_global(
                x -> -quantile.(-target_model, cdf(noise_model, x)),
                noise_model,
                target_model,
                gen,
                n_samples,
                (-3:0.1:3),
                (0:0.1:10),
            )
        end

        @test_experiments "N(0,1) to Cauchy(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Cauchy(23.0f0, 1.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=1e3,
                lr_dscr=1e-4,
                lr_gen=1e-4,
                dscr_steps=1,
                gen_steps=0,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)

            plot_global(
                x -> quantile.(target_model, cdf(noise_model, x)),
                noise_model,
                target_model,
                gen,
                n_samples,
                (-3:0.1:3),
                (0:0.1:10),
            )
        end

        @test_experiments "N(0,1) to Pareto(1,23)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Pareto(1.0f0, 23.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=2,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end
    end

    @test_experiments "Origin U(-1,1)" begin
        noise_model = Uniform(-1.0f0, 1.0f0)
        n_samples = 10000

        @test_experiments "Uniform(-1,1) to N(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Normal(23.0f0, 1.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=1000,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Uniform(22,24)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Uniform(22.0f0, 24.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Cauchy(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Cauchy(23.0f0, 1.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=1e4,
                lr_dscr=1e-4,
                lr_gen=1e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Pareto(1,23)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Pareto(1.0f0, 23.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end
    end
end

@test_experiments "wgan" begin
    @test_experiments "Origin N(0,1)" begin
        noise_model = Normal(0.0f0, 1.0f0)
        n_samples = 10000

        @test_experiments "N(0,1) to N(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Normal(23.0f0, 1.0f0)

            hparams = HyperParamsWGAN(;
                noise_model=noise_model,
                target_model=target_model,
                data_size=100,
                batch_size=1,
                epochs=1e3,
                n_critic=2,
                lr_dscr=1e-2,
                #lr_gen = 1.4e-2,
                lr_gen=1e-2,
            )

            loss = train_wgan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "N(0,1) to Uniform(22,24)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Uniform(22.0f0, 24.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "N(0,1) to Cauchy(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Cauchy(23.0f0, 1.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "N(0,1) to Pareto(1,23)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Pareto(1.0f0, 23.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end
    end

    @test_experiments "Origin U(-1,1)" begin
        noise_model = Uniform(-1.0f0, 1.0f0)
        n_samples = 10000

        @test_experiments "Uniform(-1,1) to N(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Normal(23.0f0, 1.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=1000,
                lr_dscr=1e-4,
                lr_gen=1e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Uniform(22,24)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Uniform(22.0f0, 24.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Cauchy(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Cauchy(23.0f0, 1.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Pareto(1,23)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Pareto(1.0f0, 23.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end
    end
end

@test_experiments "MMD-gan" begin
    @test_experiments "Origin N(0,1)" begin
        noise_model = Normal(0.0f0, 1.0f0)
        n_samples = 10000

        @test_experiments "N(0,1) to N(23,1)" begin
            enc = Chain(Dense(1, 11), elu, Dense(11, 29), elu)
            dec = Chain(Dense(29, 11), elu, Dense(11, 1))
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))

            target_model = Normal(4.0f0, 2.0f0)

            hparams = HyperParamsMMD1D(;
                noise_model=noise_model,
                target_model=target_model,
                data_size=100,
                batch_size=1,
                num_gen=1,
                num_enc_dec=5,
                epochs=1000000,
                lr_dec=1.0e-3,
                lr_enc=1.0e-3,
                lr_gen=1.0e-3,
            )

            train_mmd_gan_1d(enc, dec, gen, hparams)

            plot_global(
                x -> quantile.(target_model, cdf(noise_model, x)),
                noise_model,
                target_model,
                gen,
                n_samples,
                (-3:0.1:3),
                (-5:0.2:10),
            )

            hparams = HyperParams(; samples=100, K=10, epochs=2000, η=1e-3, noise_model)
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "N(0,1) to Uniform(22,24)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Uniform(22.0f0, 24.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = HyperParams(; samples=100, K=10, epochs=2000, η=1e-3, noise_model)
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "N(0,1) to Cauchy(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Cauchy(23.0f0, 1.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = HyperParams(; samples=100, K=10, epochs=2000, η=1e-3, noise_model)
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "N(0,1) to Pareto(1,23)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Pareto(1.0f0, 23.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = HyperParams(;
                samples=100, K=10, epochs=2000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end
    end

    @test_experiments "Origin U(-1,1)" begin
        noise_model = Uniform(-1.0f0, 1.0f0)
        n_samples = 10000

        @test_experiments "Uniform(-1,1) to N(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Normal(23.0f0, 1.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=1000,
                lr_dscr=1e-4,
                lr_gen=1e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = HyperParams(; samples=100, K=10, epochs=2000, η=1e-3, noise_model)
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Uniform(22,24)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Uniform(22.0f0, 24.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = HyperParams(; samples=100, K=10, epochs=2000, η=1e-3, noise_model)
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Cauchy(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Cauchy(23.0f0, 1.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = HyperParams(; samples=100, K=10, epochs=2000, η=1e-3, noise_model)
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Pareto(1,23)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = Pareto(1.0f0, 23.0f0)
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=100,
                lr_dscr=1e-4,
                lr_gen=2e-4,
                dscr_steps=5,
                gen_steps=1,
                noise_model=Normal(0.0f0, 1.0f0),
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = HyperParams(;
                samples=100, K=10, epochs=2000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end
    end
end
