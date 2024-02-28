using ISL
using HypothesisTests
using Flux
using Distributions
using Random
using Test

include("../examples/time_series_predictions/ts_utils.jl")       # Include time series utility functions

# Set a tolerance value for approximate comparisons
tol::Float64 = 1e-5

# Test the '_sigmoid' function
@testset "sigmoid" begin
    @test all(_sigmoid([2.6 2.3], 2.0) .< [0.5 0.5])
    @test !all(_sigmoid([2.6 2.3], 2.4) .< [0.5 0.5])
    @test all(_sigmoid([2.6 2.3], 2.7) .> [0.5, 0.5])
    @test all(_sigmoid([2.6f0 2.3f0], 2.0f0) .< [0.5f0, 0.5f0])
    @test !all(_sigmoid([2.6 2.3], 2.4) .< [0.5f0 0.5f0])
    @test all(_sigmoid([2.6f0 2.3f0], 2.7f0) .> [0.5f0, 0.5f0])
end;

# Test the 'ψₘ' function
@testset "ψₘ" begin
    @test ψₘ(1.0, 1) == 1.0
    @test ψₘ(1.5, 1) < 1
    @test ψₘ(0.5, 1) < 1
    #@test isapprox(ψₘ([1.0f0, 2.0f0, 0.0f0], 1), [1.0, 0.0, 0.0], atol=tol)
    #@test isapprox(ψₘ([1.0, 2.0, 0.0], 1), [1.0, 0.0, 0.0], atol=tol)
end;

# Test the 'ϕ' function
@testset "ϕ" begin
    @test ϕ([1.0f0 2.0f0 3.1f0 3.9f0], 2.4f0) > 1.0f0
    @test ϕ([1.0f0 2.0f0 3.1f0 3.9f0], 2.4f0) < 3.0f0
    @test ϕ([1.0f0 2.0f0 3.1f0 3.9f0], 3.4f0) > 2.0f0
    @test ϕ([1.0 2.0 3.1 3.9], 2.4) > 1.0
    @test ϕ([1.0 2.0 3.1 3.9], 2.4) < 3.0
    @test ϕ([1.0 2.0 3.1 3.9], 3.4) > 2.0
end;

# Test the 'γ' function
@testset "γ" begin
    #@test isapprox(γ([1.0, 2.0, 3.1, 3.9], 3.6, 3), [0.0, 0.0, 0.0, 0.9997, 0.0], atol=tol)
    @test isapprox(
        γ([1.0f0 2.0f0 3.1f0 3.9f0], 3.6f0, 3), [0.0, 0.0, 0.0, 0.92038, 0.0], atol=tol
    )
    @test isapprox(γ([1.0 2.0 3.1 3.9], 3.6, 3), [0.0, 0.0, 0.0, 0.92038, 0.0], atol=tol)
end;

@testset "generate_aₖ" begin
    #@test isapprox(
    #    generate_aₖ([1.0, 2.0, 3.1, 3.9], 3.6), [0.0, 0.0, 0.0, 0.9997, 0.0], atol=tol
    #)
    @test isapprox(
        generate_aₖ([1.0f0 2.0f0 3.1f0 3.9f0], 3.6f0),
        [0.0, 0.0, 0.0, 0.92038, 0.0],
        atol=tol,
    )
    @test isapprox(
        generate_aₖ([1.0 2.0 3.1 3.9], 3.6), [0.0, 0.0, 0.0, 0.92038, 0.0], atol=tol
    )
end;

# Test the 'scalar_diff' function
@testset "scalar_diff 1" begin
    yₖ = [1.0 2.0 3.0 4.0]
    data = 0.5:0.5:4.5
    aₖ = zeros(5)
    for y in data
        aₖ += generate_aₖ(yₖ, y)
    end
    @test isapprox(scalar_diff(aₖ), 3.1929, atol=tol)
end;

@testset "scalar_diff 2" begin
    # Test Case 1: A vector with identical elements
    q1 = [1.0, 1.0, 1.0]
    expected1 = sum((q1 .- (1.0 / length(q1))) .^ 2)
    @test scalar_diff(q1) ≈ expected1

    # Test Case 2: A vector with increasing elements
    q2 = [0.5, 1.0, 1.5]
    expected2 = sum((q2 .- (1.0 / length(q2))) .^ 2)
    @test scalar_diff(q2) ≈ expected2

    # Test Case 3: A vector with random elements
    q3 = [0.1, 0.2, 0.3, 0.4]
    expected3 = sum((q3 .- (1.0 / length(q3))) .^ 2)
    @test scalar_diff(q3) ≈ expected3
end;

@testset "Jensen-Shannon Divergence Tests" begin
    # Test 1: Identical distributions
    p1 = [0.25, 0.25, 0.25, 0.25]
    q1 = [0.25, 0.25, 0.25, 0.25]
    @test jensen_shannon_divergence(p1, q1) ≈ 0.0 atol = 1e-3

    # Test 2: Symmetry
    p2 = [0.1, 0.4, 0.5]
    q2 = [0.3, 0.4, 0.3]
    @test jensen_shannon_divergence(p2, q2) ≈ jensen_shannon_divergence(q2, p2) atol = 1e-3

    # Test 3: Non-identical distributions
    p3 = [0.1, 0.9]
    q3 = [0.9, 0.1]
    @test jensen_shannon_divergence(p3, q3) > 0.0
end

# Test the 'jensen_shannon_divergence' function
@testset "jensen shannon divergence" begin
    @test jensen_shannon_divergence([1.0, 2.0], [1.0, 2.0]) == 0.0
    @test jensen_shannon_divergence([1.0, 2.0], [1.0, 3.0]) > 0.0
    @test jensen_shannon_divergence([1.0, 2.0], [1.0, 3.0]) <
        jensen_shannon_divergence([1.0, 2.0], [1.0, 4.0])
    @test jensen_shannon_divergence([1.0, 3.0], [1.0, 2.0]) ==
        jensen_shannon_divergence([1.0, 2.0], [1.0, 3.0])
    @test jensen_shannon_divergence([0.0, 3.0], [1.0, 3.0]) > 0.0
end;

@testset "jensen shannon ∇" begin
    yₖ = [1.0 2.0 3.0 4.0]
    data = 0.5:0.5:4.5
    aₖ = zeros(5)
    for y in data
        aₖ += generate_aₖ(yₖ, y)
    end
    @test isapprox(jensen_shannon_∇(aₖ ./ sum(aₖ)), 0.0, atol=tol)
end;

# Test the 'jensen_shannon_∇' function
@testset "invariant_statistical_loss" begin
    @testset "learning Normal(4.0f0, 2.0f0)" begin
        nn = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
        hparams = ISLParams(;
            samples=1000, K=10, epochs=2000, η=1e-2, transform=Normal(0.0f0, 1.0f0)
        )

        target_model = Normal(4.0f0, 2.0f0)

        train_set = Float32.(rand(target_model, hparams.samples))
        loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

        loss = invariant_statistical_loss(nn, loader, hparams)

        validation_set = Float32.(rand(target_model, hparams.samples))
        data = vec(nn(rand(hparams.transform, hparams.samples)'))

        @test length(loss) == hparams.epochs
        @test all(loss .>= 0)
        @test loss[end] <= loss[1]

        @test pvalue(HypothesisTests.ApproximateTwoSampleKSTest(validation_set, data)) >
            0.01
        #@test Pingouin.anderson(data, Normal(4.0f0, 2.0f0))[1] == true
        #hist1 = fit(Histogram, train_set, (-2:0.1:8))
        #hist2 = fit(Histogram, data, (-2:0.1:8))
        #@test js_divergence(hist1.weights, hist2.weights)/hparams.samples < 0.03
    end

    @testset "learning uniform distribution (-2,2)" begin
        nn = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
        hparams = ISLParams(;
            samples=1000, K=10, epochs=2000, η=1e-2, transform=Normal(0.0f0, 1.0f0)
        )

        target_model = Uniform(-2, 2)

        train_set = Float32.(rand(target_model, hparams.samples))
        loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

        invariant_statistical_loss(nn, loader, hparams)

        validation_set = Float32.(rand(target_model, hparams.samples))
        data = vec(nn(rand(hparams.transform, hparams.samples)'))

        @test pvalue(HypothesisTests.ApproximateTwoSampleKSTest(validation_set, data)) >
            0.01
    end

    #Testing the Capability of ISL to Learn 1D Distributions
    @testset "learning Cauchy distribution" begin
        nn = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
        hparams = ISLParams(;
            samples=1000, K=10, epochs=2000, η=1e-2, transform=Normal(0.0f0, 1.0f0)
        )

        target_model = Cauchy(1.0f0, 3.0f0)

        train_set = Float32.(rand(target_model, hparams.samples))
        loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

        invariant_statistical_loss(nn, loader, hparams)

        validation_set = Float32.(rand(target_model, hparams.samples))
        data = vec(nn(rand(hparams.transform, hparams.samples)'))

        @test pvalue(HypothesisTests.ApproximateTwoSampleKSTest(validation_set, data)) >
            0.01
    end

    @testset "learning Bimodal Normal Distribution" begin
        nn = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
        hparams = ISLParams(;
            samples=1000, K=10, epochs=2000, η=1e-2, transform=Normal(0.0f0, 1.0f0)
        )

        target_model = MixtureModel(Normal[Normal(5.0f0, 2.0f0), Normal(-1.0f0, 1.0f0)])

        train_set = Float32.(rand(target_model, hparams.samples))
        loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

        invariant_statistical_loss(nn, loader, hparams)

        validation_set = Float32.(rand(target_model, hparams.samples))
        data = vec(nn(rand(hparams.transform, hparams.samples)'))

        @test pvalue(HypothesisTests.ApproximateTwoSampleKSTest(validation_set, data)) >
            0.01
    end

    @testset "learning modal auto_adaptative_block_learning Normal(4.0f0, 2.0f0)" begin
        nn = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))
        hparams = AutoISLParams(;
            max_k=10, samples=1000, epochs=1000, η=1e-2, transform=Normal(0.0f0, 1.0f0)
        )

        target_model = Normal(4.0f0, 2.0f0)

        train_set = Float32.(rand(target_model, hparams.samples))
        loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

        loss = auto_invariant_statistical_loss(nn, loader, hparams)

        validation_set = Float32.(rand(target_model, hparams.samples))
        data = vec(nn(rand(hparams.transform, hparams.samples)'))

        @test length(loss) == hparams.epochs
        @test all(loss .>= 0)
        @test loss[end] <= loss[1]

        @test pvalue(HypothesisTests.ApproximateTwoSampleKSTest(validation_set, data)) >
            0.01
    end

    @testset "testing get_window_of_Aₖ" begin
        # Mock model function in Julia
        function mock_model(x)
            # This mock model is adjusted to return predictable values
            return x * 2
        end

        mock_data = [100.0, 100.0, 100.0]
        K = 2
        transform = Normal(0.0f0, 1.0f0)
        result = get_window_of_Aₖ(transform, mock_model, mock_data, K::Int64)
        expected = [0, 0, 3]
        @test result == expected
    end

    @testset "test_convergence_to_uniform" begin
        # Distributions
        uniform_distribution = [25, 25, 25, 25]  # Uniform
        non_uniform_distribution = [5, 5, 20, 70]  # Non-uniform
        approx_uniform_distribution = [20, 30, 20, 30]  # Approx-uniform
        limit_uniform_distribution_negative = [15, 35, 20, 30]  # Limit-uniform negative
        limit_uniform_distribution_positive = [17, 33, 20, 30]  # Limit-uniform positive

        # Test assertions
        @test convergence_to_uniform(uniform_distribution)
        @test !convergence_to_uniform(non_uniform_distribution)
        @test convergence_to_uniform(approx_uniform_distribution)
        @test !convergence_to_uniform(limit_uniform_distribution_negative)
        @test convergence_to_uniform(limit_uniform_distribution_positive)
    end

    @testset "test_get_better_K" begin
        function mock_model_1(x)
            # This mock model is adjusted to return predictable values
            return 2 * x
        end

        mock_data = [100.0, 100.0, 100.0]

        hparams = AutoISLParams(;
            max_k=100, samples=1000, epochs=1000, η=1e-2, transform=noise_model
        )
        expected_K = 2  # Expected K value for these inputs

        # Call to get_better_K with the mock model, mock data, starting K value, and hyperparameters
        result_K = ISL.get_better_K(mock_model_1, mock_data, 2, hparams)

        @test result_K == expected_K

        result_K = ISL.get_better_K(mock_model_1, mock_data, 10, hparams)
        expected_K = 10  # Expected K value for these inputs
        @test result_K == expected_K
    end

    # Test function
    @testset "ts_invariant_statistical_loss_one_step_prediction Tests" begin

        # Define AR model parameters
        ar_hparams = ARParams(;
            ϕ=[0.5f0, 0.3f0, 0.2f0],  # Autoregressive coefficients
            x₁=rand(Normal(0.0f0, 1.0f0)),  # Initial value from a Normal distribution
            proclen=2000,  # Length of the process
            noise=Normal(0.0f0, 0.2f0),  # Noise in the AR process
        )

        # Define the recurrent and generative models
        recurrent_model = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))
        generative_model = Chain(Dense(11, 16, relu), Dense(16, 1, identity))

        # Generate training and testing data
        n_series = 200  # Number of series to generate
        loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(
            n_series, ar_hparams
        )

        # --- Training Configuration ---

        # Define hyperparameters for time series prediction
        ts_hparams = HyperParamsTS(;
            seed=1234,
            η=1e-3,  # Learning rate
            epochs=n_series,
            window_size=1000,  # Size of the window for prediction
            K=10,  # Hyperparameter K (if it has a specific use, add a comment)
        )

        # Train model and calculate loss
        loss = ts_invariant_statistical_loss_one_step_prediction(
            recurrent_model, generative_model, loaderXtrain, loaderYtrain, ts_hparams
        )
        @test !isempty(loss) # Check that losses are returned
        @test all(loss .>= 0) # Assuming loss cannot be negative; adjust as necessary

        # --- Testing Configuration ---
        prediction = Vector{Float32}()
        Flux.reset!(recurrent_model)
        n_average = 100  # Number of predictions to average
        s = 0
        loaderX = collect(loaderXtrain)[1]
        loadertestX = collect(loaderXtest)[1]
        for data in loaderX
            s = recurrent_model([data])
            y, _ = average_prediction(generative_model, s, n_average)
            append!(prediction, y[1])
        end

        ideal = vcat(loaderX, loadertestX)
        t = 1:length(ideal)

        prediction = Vector{Float32}()
        std_prediction = Vector{Float32}()
        for data in loadertestX
            y, std = average_prediction(generative_model, s, n_average)
            s = recurrent_model([y[1]])
            append!(prediction, y[1])
            append!(std_prediction, std)
        end

        nd = ND(loadertestX, prediction)
        rmse = RMSE(loadertestX, prediction)
        qlρ = QLρ(loadertestX, prediction; ρ=0.9)

        @test nd >= 0.0 and nd <= 1.0
        @test rmse >= 0.0 and rmse <= 25.0
        @test qlρ >= 0.0 and nd <= 2.0
    end
end;
