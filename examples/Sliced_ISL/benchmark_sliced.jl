using ISL
using KernelDensity
using Random

include("../utils.jl")

@test_experiments "sliced ISL" begin
    @test_experiments "N(0,1)" begin
        noise_model = gpu(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
        n_samples = 10000
        @test_experiments "N(0,1) to N(23,1)" begin
            gen = gpu(Chain(
                Dense(2, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 2)
            ))

            mean_vector = [2.0, 3.0]
            cov = [1.0 0.5; 0.5 1.0]
            target_model = gpu(MvNormal(mean_vector, cov))

            hparams = gpu(HyperParamsSlicedISL(;
                K=10, samples=1000, epochs=100, η=1e-2, noise_model=noise_model, m=10
            ))

            train_set = gpu(Float32.(rand(target_model, hparams.samples * hparams.epochs)))
            loader = gpu(Flux.DataLoader(
                train_set; batchsize=hparams.samples, shuffle=true, partial=false
            ))

            loss = sliced_invariant_statistical_loss_optimized_gpu_2(gen, loader, hparams)

            #loss = sliced_invariant_statistical_loss(gen, loader, hparams)
            #loss = sliced_invariant_statistical_loss_2(gen, loader, hparams)

            #loss = sliced_invariant_statistical_loss_multithreaded_2(gen, loader, hparams)

            plotlyjs()
            output_data = gen(Float32.(rand(noise_model, n_samples)))
            x = output_data[1, :]
            y = output_data[2, :]
            kde_result = kde((x, y))
            contour(
                kde_result.x,
                kde_result.y,
                kde_result.density;
                xlabel="X",
                ylabel="Y",
                zlabel="Index",
            )
            plot(
                kde_result.x,
                kde_result.y,
                kde_result.density;
                xlabel="X",
                ylabel="Y",
                zlabel="Index",
                st=:surface,
            )

            x_axis = -20:0.1:20
            y_axis = -20:0.1:20
            f(x_axis, y_axis) = pdf(target_model, [x_axis, y_axis])
            kde_result = kde((x_axis, y_axis))
            contour!(
                kde_result.x,
                kde_result.y,
                kde_result.density;
                xlabel="X",
                ylabel="Y",
                zlabel="Index",
            )
            plot!(
                x_axis,
                y_axis,
                f;
                title="Contour Plot of 2D Gaussian Distribution",
                xlabel="X-axis",
                ylabel="Y-axis",
                st=:surface,
                alpha=0.8,
            )
        end

        @test_experiments "N(0,1) to N(23,1)" begin
            gen = Chain(Dense(2, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 2))

            # Define the custom distribution type
            struct BivariatedUniform <: ContinuousMultivariateDistribution
                a_min::Float64
                a_max::Float64
                b_min::Float64
                b_max::Float64
            end

            Distributions.dim(::BivariatedUniform) = 2

            Base.length(::BivariatedUniform) = 2

            function Distributions.pdf(d::BivariatedUniform, x::AbstractArray{Float64})
                x_val, y_val = x[1], x[2]
                return pdf(Uniform(d.a_min, d.a_max), x_val) *
                       pdf(Uniform(d.b_min, d.b_max), y_val)  # Example pdf
            end

            function Distributions.rand(rng::AbstractRNG, d::BivariatedUniform)
                x = rand(rng, Uniform(d.a_min, d.a_max))
                y = rand(rng, Uniform(d.b_min, d.b_max))
                return [x, y]
            end

            function Distributions._rand!(
                rng::AbstractRNG, d::MultivariateDistribution, x::AbstractArray{Float64}
            )
                # Check if the dimensions of x match the dimensions of the distribution
                @assert size(x, 1) == length(d) "Dimension mismatch"

                # Iterate over each column (sample) in x
                for i in 1:size(x, 2)
                    # Generate a sample from the distribution d using the rng
                    sample = rand(rng, d)

                    # Fill the ith column of x with this sample
                    x[:, i] .= sample
                end

                return x
            end

            target_model = BivariatedUniform(-2, 2, -2, 2)

            hparams = HyperParamsSlicedISL(;
                K=10, samples=1000, epochs=100, η=1e-2, noise_model=noise_model, m=40
            )

            train_set = Float32.(rand(target_model, hparams.samples * hparams.epochs))
            loader = Flux.DataLoader(
                train_set; batchsize=hparams.samples, shuffle=true, partial=false
            )

            loss = sliced_invariant_statistical_loss(gen, loader, hparams)
        end
    end

    @test_experiments "N(0,1) to N(23,1)" begin
        gen = Chain(Dense(2, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 2))

        # Define the custom distribution type
        struct CoustomDistribution <: ContinuousMultivariateDistribution
            a_min::Float64
            a_max::Float64
            μ::Float64
            σ::Float64
        end

        Distributions.dim(::CoustomDistribution) = 2

        Base.length(::CoustomDistribution) = 2

        function Distributions.pdf(d::CoustomDistribution, x::AbstractArray{Float64})
            x_val, y_val = x[1], x[2]
            return pdf(Uniform(d.a_min, d.a_max), x_val) * pdf(Normal(d.μ, d.σ), y_val)  # Example pdf
        end

        function Distributions.rand(rng::AbstractRNG, d::CoustomDistribution)
            x = rand(rng, Uniform(d.a_min, d.a_max))
            y = rand(rng, Normal(d.μ, d.σ))
            return [x, y]
        end

        function Distributions._rand!(
            rng::AbstractRNG, d::CoustomDistribution, x::AbstractArray{Float64}
        )
            # Ensure that the dimensions of x are compatible with the distribution
            @assert size(x, 1) == 2 "Dimension mismatch"

            # Iterate over each column (sample) in x
            for i in 1:size(x, 2)
                # Generate a sample for each dimension of the distribution
                x[1, i] = rand(rng, Uniform(d.a_min, d.a_max))  # First dimension
                x[2, i] = rand(rng, Normal(d.μ, d.σ)) # Second dimension
            end

            return x
        end

        μ = 2.0
        σ = 1.0
        target_model = CoustomDistribution(-2, 2, μ, σ)

        hparams = HyperParamsSlicedISL(;
            K=10, samples=1000, epochs=100, η=1e-2, noise_model=noise_model, m=40
        )

        train_set = Float32.(rand(target_model, hparams.samples * hparams.epochs))
        loader = Flux.DataLoader(
            train_set; batchsize=hparams.samples, shuffle=true, partial=false
        )

        loss = sliced_invariant_statistical_loss(gen, loader, hparams)
    end

    @test_experiments "N(0,1) to N(23,1)" begin
        gen = Chain(Dense(2, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 2))

        # Define the custom distribution type
        struct CoustomDistribution <: ContinuousMultivariateDistribution
            α::Float64
            β::Float64
            μ::Float64
            σ::Float64
        end

        Distributions.dim(::CoustomDistribution) = 2

        Base.length(::CoustomDistribution) = 2

        function Distributions.pdf(d::CoustomDistribution, x::AbstractArray{Float64})
            x_val, y_val = x[1], x[2]
            return pdf(Cauchy(d.α, d.β), x_val) * pdf(Normal(d.μ, d.σ), y_val)  # Example pdf
        end

        function Distributions.rand(rng::AbstractRNG, d::CoustomDistribution)
            x = rand(rng, Cauchy(d.α, d.β))
            y = rand(rng, Normal(d.μ, d.σ))
            return [x, y]
        end

        function Distributions._rand!(
            rng::AbstractRNG, d::CoustomDistribution, x::AbstractArray{Float64}
        )
            # Ensure that the dimensions of x are compatible with the distribution
            @assert size(x, 1) == 2 "Dimension mismatch"

            # Iterate over each column (sample) in x
            for i in 1:size(x, 2)
                # Generate a sample for each dimension of the distribution
                x[1, i] = rand(rng, Cauchy(d.α, d.β))  # First dimension
                x[2, i] = rand(rng, Normal(d.μ, d.σ)) # Second dimension
            end

            return x
        end

        target_model = CoustomDistribution(2.0, 2.0, 2.0, 1.0)

        hparams = HyperParamsSlicedISL(;
            K=10, samples=1000, epochs=100, η=1e-2, noise_model=noise_model, m=40
        )

        train_set = Float32.(rand(target_model, hparams.samples * hparams.epochs))
        loader = Flux.DataLoader(
            train_set; batchsize=hparams.samples, shuffle=true, partial=false
        )

        loss = sliced_invariant_statistical_loss(gen, loader, hparams)
    end

    @test_experiments "N(0,1) to N(23,1)" begin
        gen = Chain(Dense(2, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 2))

        # Define the custom distribution type
        struct CoustomDistribution <: ContinuousMultivariateDistribution
            α::Float32
            β::Float32
            a_min::Float32
            a_max::Float32
        end

        Distributions.dim(::CoustomDistribution) = 2

        Base.length(::CoustomDistribution) = 2

        function Distributions.pdf(d::CoustomDistribution, x::AbstractArray{Float64})
            x_val, y_val = x[1], x[2]
            return pdf(Uniform(d.a_min, d.a_max), x_val) * pdf(Cauchy(d.α, d.β), y_val)
        end

        function Distributions.rand(rng::AbstractRNG, d::CoustomDistribution)
            x = rand(rng, Uniform(d.a_min, d.a_max))
            y = rand(rng, Cauchy(d.α, d.β))
            return [x, y]
        end

        function Distributions._rand!(
            rng::AbstractRNG, d::CoustomDistribution, x::AbstractArray{Float64}
        )
            # Ensure that the dimensions of x are compatible with the distribution
            @assert size(x, 1) == 2 "Dimension mismatch"

            # Iterate over each column (sample) in x
            for i in 1:size(x, 2)
                # Generate a sample for each dimension of the distribution
                x[1, i] = rand(rng, Uniform(d.a_min, d.a_max))  # First dimension
                x[2, i] = rand(rng, Cauchy(d.α, d.β)) # Second dimension
            end

            return x
        end

        target_model = CoustomDistribution(0.0, 10.0, -1.0, 1.0)

        hparams = HyperParamsSlicedISL(;
            K=10, samples=1000, epochs=100, η=1e-2, noise_model=noise_model, m=10
        )

        train_set = Float32.(rand(target_model, hparams.samples * hparams.epochs))
        loader = Flux.DataLoader(
            train_set; batchsize=hparams.samples, shuffle=true, partial=false
        )

        loss = sliced_invariant_statistical_loss(gen, loader, hparams)
    end

    @test_experiments "N(0,1) to N(23,1)" begin
        gen = Chain(Dense(2, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 2))

        # Define the custom distribution type
        struct CoustomDistribution <: ContinuousMultivariateDistribution
            α₁::Float32
            β₁::Float32
            α₂::Float32
            β₂::Float32
        end

        Distributions.dim(::CoustomDistribution) = 2

        Base.length(::CoustomDistribution) = 2

        function Distributions.pdf(d::CoustomDistribution, x::AbstractArray{Float64})
            x_val, y_val = x[1], x[2]
            return pdf(Cauchy(d.α₁, d.β₁), x_val) * pdf(Cauchy(d.α₂, d.β₂), y_val)
        end

        function Distributions.rand(rng::AbstractRNG, d::CoustomDistribution)
            x = rand(rng, Cauchy(d.α₁, d.β₁))
            y = rand(rng, Cauchy(d.α₂, d.β₂))
            return [x, y]
        end

        function Distributions._rand!(
            rng::AbstractRNG, d::CoustomDistribution, x::AbstractArray{Float64}
        )
            # Ensure that the dimensions of x are compatible with the distribution
            @assert size(x, 1) == 2 "Dimension mismatch"

            # Iterate over each column (sample) in x
            for i in 1:size(x, 2)
                # Generate a sample for each dimension of the distribution
                x[1, i] = rand(rng, Cauchy(d.α₁, d.β₁))  # First dimension
                x[2, i] = rand(rng, Cauchy(d.α₂, d.β₂)) # Second dimension
            end

            return x
        end

        target_model = CoustomDistribution(1.0, 1.0, 1.0, 1.0)

        hparams = HyperParamsSlicedISL(;
            K=10, samples=1000, epochs=100, η=1e-2, noise_model=noise_model, m=20
        )

        train_set = Float32.(rand(target_model, hparams.samples * hparams.epochs))
        loader = Flux.DataLoader(
            train_set; batchsize=hparams.samples, shuffle=true, partial=false
        )

        loss = sliced_invariant_statistical_loss_distributed(gen, loader, hparams)
    end
end;
