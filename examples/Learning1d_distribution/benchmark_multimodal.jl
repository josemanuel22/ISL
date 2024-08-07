using ISL
using GAN

include("../utils.jl")

@test_experiments "vanilla_gan" begin
    @test_experiments "Origin N(0,1)" begin
        noise_model = Normal(0.0f0, 1.0f0)
        n_samples = 10000

        @test_experiments "N(0,1) to M(N(5.0f0, 2.0f0), N(-1.0f0, 1.0f0))" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = MixtureModel([Normal(5.0f0, 2.0f0), Normal(-1.0f0, 1.0f0)])
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=1e4,
                lr_dscr=1e-3,
                lr_gen=1e-4,
                dscr_steps=4,
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

            plot_global(
                x -> quantile.(target_model, cdf(noise_model, x)),
                noise_model,
                target_model,
                gen,
                n_samples,
                (-3:0.1:3),
                (0:0.2:10),
            )

            #@test js_divergence(hist1.weights, hist2.weights)/hparams.samples < 0.03

        end

        @test_experiments "N(0,1) to Uniform(22,24)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = MixtureModel([Normal(5.0f0, 2.0f0), Pareto(5.0f0, 1.0f0)])
            hparams = HyperParamsVanillaGan(;
                data_size=100,
                batch_size=1,
                epochs=1e4,
                lr_dscr=1e-4,
                lr_gen=1e-4,
                dscr_steps=2,
                gen_steps=1,
                noise_model=noise_model,
                target_model=target_model,
            )

            train_vanilla_gan(dscr, gen, hparams)

            hparams = AutoISLParams(;
                max_k=10, samples=10000, epochs=1000, η=1e-3, transform=noise_model
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
                (0:0.05:8),
            )
        end

        @test_experiments "N(0,1) to Cauchy(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = MixtureModel([Cauchy(7.0f0, 1.0f0), Normal(-1.0f0, 1.0f0)])
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

        @test_experiments "N(0,1) to Pareto(1,23)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = MixtureModel([Uniform(-4.0f0, 1.0f0), Pareto(1.0f0, 23.0f0)])

            Pareto(1.0f0, 23.0f0)
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
            target_model = MixtureModel([Normal(5.0f0, 2.0f0), Normal(-1.0f0, 1.0f0)])
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

            hparams = HyperParams(; samples=100, K=10, epochs=2000, η=1e-3, noise_model)
            train_set = rand(target_model, hparams.samples)
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            hparams = AutoISLParams(;
                max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model
            )
            train_set = Float32.(rand(target_model, hparams.samples))
            loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

            auto_invariant_statistical_loss(gen, loader, hparams)
        end

        @test_experiments "Uniform(-1,1) to Uniform(22,24)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = MixtureModel([
                Normal(5.0f0, 2.0f0), Normal(-1.0f0, 1.0f0), Normal(-7.0f0, 0.4f0)
            ])
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

        @test_experiments "Uniform(-1,1) to Cauchy(23,1)" begin
            gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
            dscr = Chain(
                Dense(1, 11), elu, Dense(11, 29), elu, Dense(29, 11), elu, Dense(11, 1, σ)
            )
            target_model = MixtureModel([Cauchy(7.0f0, 1.0f0), Normal(-1.0f0, 1.0f0)])
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
            target_model = MixtureModel([Uniform(-4.0f0, 1.0f0), Pareto(1.0f0, 23.0f0)])
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
