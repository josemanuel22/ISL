using ISL
using Random
using CSV
using DataFrames
using Distributions
using HypothesisTests

include("../utils.jl")

# Implementing the Hill estimator for the tail index
# The Hill estimator is defined as the average of the logarithms of the top k order statistics divided by the logarithm of the k-th order statistic
function hill_estimator(y)
    ysort = sort(y) # sort the returns
    CT = 100        # set the threshold
    iota = 1 / mean(log.(ysort[1:CT] / ysort[CT + 1]))
    return iota
end

function HillEstimator(data::Vector{Float64})
    """
    Returns the Hill Estimators for some 1D data set.
    """
    # Sort data in such a way that the smallest value is first and the largest value comes last:
    Y = sort(data)
    n = length(Y)

    Hill_est = zeros(Float64, n - 1)

    for k in 1:(n - 1)  # k = 1,...,n-1
        summ = 0.0
        for i in 1:k   # i = 1, ..., k
            if Y[n + 1 - i] <= 0.0 || Y[n - k] <= 0.0
                continue
            end
            summ += log(Y[n + 1 - i]) - log(Y[n - k])
        end

        Hill_est[k] = (1 / k) * summ
    end

    kappa = 1.0 ./ Hill_est
    return kappa
end

###HAZ UNA BERNOULLI Si no
struct GPD <: ContinuousUnivariateDistribution
    ξ::Float32
end

Distributions.dim(::GPD) = 1

Base.length(::GPD) = 1

function Distributions.pdf(d::GPD, x::AbstractArray{Float32})
    x_val = x[1]
    return pdf((((Uniform(0.0f0, 1.0f0))^(-d.ξ)) ./ d.ξ), x_val)
end

function Distributions.rand(rng::AbstractRNG, d::GPD)
    x = (rand(rng, Uniform(0.0f0, 1.0f0))^(-d.ξ)) - 1.0f0 ./ d.ξ
    if rand() < 0.5
        return -1.0f0 * x
    else
        return x
    end
end

function Distributions.rand!(rng::AbstractRNG, d::GPD, x::AbstractVector{Float32})
    # Iterate over each element of the array
    for i in 1:length(x)
        # Generate a random value according to the GPD distribution
        generated_val = (rand(rng, Uniform(0.0f0, 1.0f0))^(-d.ξ)) - 1.0f0 ./ d.ξ

        # Store the generated value in the current position of the provided array
        #append!(x, generated_val)
        if rand() < 0.5
            x[i] = -1.0f0 * generated_val
        else
            x[i] = generated_val
        end
    end

    return x  # Return the modified array, now filled with generated values
end

function generate_Zξ(ξ::Float64)
    U = rand(-20.0f0, 20.0f0) # Generates a random number U from Uniform(0, 1)
    Zξ = (U^(-ξ) - 1) / ξ
    return Zξ
end

struct GPD2 <: ContinuousUnivariateDistribution
    ξ::Float32
end

Distributions.dim(::GPD2) = 1

Base.length(::GPD2) = 1

function Distributions.pdf(d::GPD2, x::AbstractArray{Float32})
    x_val = x[1]
    return pdf((((Uniform(-1.0f0, 1.0f0))^(-d.ξ) - 1.0f0) / d.ξ), x_val)
end

function Distributions.rand(rng::AbstractRNG, d::GPD2)
    U = rand(rng) # Generates a random number U from Uniform(0, 1)
    Zξ = (U^(-d.ξ) - 1) / d.ξ
    return Float32(Zξ)
end

function Distributions.rand!(rng::AbstractRNG, d::GPD2, x::AbstractVector{Float32})
    # Iterate over each element of the array
    for i in 1:length(x)
        # Generate a random value according to the GPD distribution
        generated_val = rand(rng, d)

        # Store the generated value in the current position of the provided array
        #append!(x, generated_val)
        x[i] = generated_val
    end

    return x  # Return the modified array, now filled with generated values
end

@test_experiments "Origin N(0,1)" begin
    # Initialize the noise model as a normal distribution N(0,1)
    noise_model = Uniform(0.0f0, 1.0f0)
    noise_model = Normal(0.0f0, 1.0f0)
    noise_model = LogNormal(1.0f0, 1.0f0)
    noise_model = GPD(1.0f0)
    noise_model = GPD2(1.0f0)
    noise_model = Gumbel(1.0f0, 1.0f0)
    noise_model = GeneralizedPareto(1.0f0, 1.0f0, 1.0f0)
    t_dist = TDist(3)

    @test_experiments "N(0,1) to N(23,1)" begin
        gen = Chain(
            Dense(1, 100, relu), Dense(100, 100, relu), Dense(100, 100, relu), Dense(100, 1)
        )

        #target_model = MixtureModel([Cauchy(0.2f0, 0.70f0), Cauchy(2.0f0, 0.85f0)])
        target_model = MixtureModel([Cauchy(-1.0f0, 0.70f0), Cauchy(1.0f0, 0.85f0)])
        #target_model = MixtureModel([Cauchy(-10.0f0, 1.70f0), Cauchy(1.0f0, 0.85f0)])
        #target_model = Cauchy(1.0f0, 2.0f0)

        # Parameters for automatic invariant statistical loss
        #hparams = ISLParams(; samples=1000, K=20, epochs=600, η=1e-3, transform=noise_model)
        hparams = ISLParams(;
            samples=1000, K=20, epochs=1000, η=1e-3, transform=noise_model
        )
        #Estos parametros dan buenos resutlados
        #hparams = ISLParams(;
        #    samples=1000, K=20, epochs=1000, η=1e-2, transform=noise_model
        #)
        #hparams = AutoISLParams(;
        #    max_k=10, samples=10000, epochs=200, η=1e-2, transform=noise_model
        #)

        # Preparing the training set and data loader
        train_set = Float32.(rand(target_model, hparams.samples))
        loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)

        # Training using the automatic invariant statistical loss
        loss = invariant_statistical_loss(gen, loader, hparams)
        #loss = auto_invariant_statistical_loss(gen, loader, hparams)

        plot_range = -200:1.0:200  # Adjust step for smoother or coarser plots
        #plot_range = -10:0.1:10  # Adjust step for smoother or coarser plots

        plotlyjs()

        n_samples = 1000000
        x = rand(noise_model, n_samples)
        ŷ = gen(x')

        histogram(
            ŷ';
            bins=plot_range,
            normalize=:pdf,
            yscale=:log,
            z=2,
            ylims=(10^-6, 1),
            color=get(ColorSchemes.rainbow, 0.2),
            legend=false,
            #label="Generated distribution",
            alpha=0.7,
        )
        plot!(
            x -> pdf(target_model, x),  # Function to plot
            plot_range;  # Range over which to plot the function
            lw=2,  # Line width
            z=-1,
            #yscale=:log10,  # Logarithmic y-scale
            linecolor=:redsblues,  # Line color
            #xlims=(-10, 10),  # Limits for the x-axis
            #ylims=(10^-6, 1),  # Limits for the y-axis, set to the specified log scale range
            xlabel="x",  # Label for the x-axis
            ylabel="pdf",  # Label for the y-axis
            legend=false,
            #label="Target distribution",  # Label for the plot
        )
    end

    @test_experiments "N(0,1) to N(23,1)" begin
        struct MultivariateHeavyTailed <: ContinuousMultivariateDistribution
            μ₁::Float32
            σ₁::Float32
            μ₂::Float32
            σ₂::Float32
        end

        Distributions.dim(::MultivariateHeavyTailed) = 2

        Base.length(::MultivariateHeavyTailed) = 2

        #=
        function Distributions.pdf(d::MultivariateHeavyTailed, x::AbstractArray{Float32})
            x_val, y_val = x[1], x[2]
            A = Cauchy(d.μ₁, d.σ₁)
            B = Cauchy(d.μ₂, d.σ₂)
            return pdf(A, x_val) * pdf(sign(A - B) * abs(A - B)^(0.5f0), y_val)
        end
        =#

        function Distributions.rand(rng::AbstractRNG, d::MultivariateHeavyTailed)
            A = Cauchy(d.μ₁, d.σ₁)
            B = Cauchy(d.μ₂, d.σ₂)
            x = rand(rng, A) + rand(rng, B)
            y = sign(rand(rng, A) - rand(rng, B)) * abs(rand(A) - rand(B))^(0.5f0)
            return [x, y]
        end

        function Distributions.rand!(
            rng::AbstractRNG, d::MultivariateHeavyTailed, x::AbstractVector{Float64}
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

        noise_model = GPD(0.7f0)

        target_model = MultivariateHeavyTailed(0.5, 1.0f0, 1.0f0, 1.0f0)

        gen = Chain(
            Dense(1, 256), relu, Dense(256, 256), relu, Dense(256, 256), relu, Dense(256, 2)
        )

        #target_model = Cauchy(0.5f0, 1.0f0)
        hparams = HyperParamsSlicedISL(;
            K=10, samples=1000, epochs=20, η=1e-3, noise_model=noise_model, m=10
        )

        # Preparing the training set and data loader
        train_set = Float32.(rand(target_model, hparams.samples * hparams.epochs))
        loader = Flux.DataLoader(
            train_set; batchsize=hparams.samples, shuffle=true, partial=false
        )

        # Training using the automatic invariant statistical loss
        #invariant_statistical_loss(gen, loader, hparams)
        total_loss = []
        for _ in 1:4
            #loss = marginal_invariant_statistical_loss(gen, loader, hparams)
            append!(total_loss, sliced_invariant_statistical_loss(gen, loader, hparams))
        end

        plotlyjs()
        x = rand(target_model, 100000)
        ecdf₁ = ecdf(x[1, :])
        ranges = (0:1.0:100000)
        plot(
            ranges,
            1 .- ecdf₁(ranges);
            xscale=:log10,
            yscale=:log10,
            xlabel="x₀",
            ylabel="P(X₀ > x₀)",
            #label="Target distribution",
            linecolor=:red,
            lw=2,
        )
        z = rand(noise_model, n_samples)
        ŷ = gen(z')
        ecdf₁ = ecdf(ŷ[1, :])
        ranges = (0:10.0:100000)
        plot!(sort(z), 1 .- ecdf₁(sort(z)))

        x = rand(target_model, 100000)
        ecdf₂ = ecdf(x[2, :])
        ranges = (0:1.0:100000)
        plot(
            ranges,
            1 .- ecdf₂(ranges);
            xscale=:log10,
            yscale=:log10,
            xlabel="x₁",
            ylabel="P(X₁ > x₁)",
            #label="Target distribution",
            linecolor=:red,
            lw=2,
        )
        z = rand(noise_model, n_samples)
        ŷ = gen(z')
        ecdf₂ = ecdf(ŷ[2, :])
        ranges = (0:10.0:100000)
        plot!(sort(z), 1 .- ecdf₂(sort(z)))

        x = rand(target_model, 10000)
        z = rand(noise_model, 10000)
        ŷ = gen(z')
        scatter(x[1, :], x[2, :])
        scatter!(ŷ[1, :], ŷ[2, :])
    end
end

function calculate_area(y, ŷ, n)
    fr_ecdf⁻¹ = ecdf(y)
    gr_ecdf⁻¹ = ecdf(ŷ[1, :])

    area = 0.0
    for i in 1:n
        term1 = log(fr_ecdf⁻¹(i / n))
        term2 = log(gr_ecdf⁻¹(i / n))
        area += abs(term1 - term2) * log((i + 1) / i)
    end
    return area
end

function KSD_1(noise_model, train_set, gen, n_sample, range)
    hist1 = fit(Histogram, train_set, range)

    data = vec(gen(rand(noise_model, n_sample)'))
    hist2 = fit(Histogram, data, range)
    return maximum(abs.(hist1.weights - hist2.weights)) / n_sample
end

@test_experiments "N(0,1) to N(23,1)" begin

    # Assuming the data is saved in 'typing_data.txt'
    file_path = "/Users/jmfrutos/Desktop/Keystrokes/files/metadata_participants.txt"

    # Read the data into a DataFrame
    df = CSV.read(file_path, DataFrame; delim='\t')
    # Select just the AVG_IKI column
    avg_iki_df = select(df, :AVG_IKI)

    # Calculate mean and standard deviation
    mean_val = mean(avg_iki_df.AVG_IKI)
    std_dev = std(avg_iki_df.AVG_IKI)

    # Normalize the data
    normalized_data = Float32.((avg_iki_df.AVG_IKI .- mean_val) ./ std_dev)

    # Create a DataFrame for the normalized data (optional)
    df_normalized = DataFrame(; AVG_IKI_normalized=normalized_data)

    # Assuming df_normalized is your DataFrame as per the last line of your code snippet

    # Shuffle the DataFrame rows
    shuffled_df = shuffle(df_normalized)

    # Calculate the number of rows for each set
    num_rows = nrow(shuffled_df)
    num_train = Int(floor(0.1 * num_rows)) # 10% for training
    num_val = Int(floor(0.1 * num_rows)) # 10% for validation
    num_test = num_rows - num_train - num_val # The rest for testing

    # Split the data
    train_set = vec(shuffled_df.AVG_IKI_normalized)[1:num_train]
    val_set = vec(shuffled_df.AVG_IKI_normalized)[(num_train + 1):(num_train + num_val)]
    test_set = vec(shuffled_df.AVG_IKI_normalized)[(num_train + num_val + 1):end]

    #noise_model = GPD(0.68f0) #Normal(0.0f0, 1.0f0)#GPD(0.68f0)
    noise_model = GeneralizedPareto(1.0f0, 1.0f0, 1.0f0)
    noise_model = Normal(0.0f0, 1.0f0)
    noise_model = Uniform(0.0f0, 1.0f0)
    noise_model = LogNormal(1.0f0, 1.0f0)

    gen = Chain(Dense(1, 32), relu, Dense(32, 32), relu, Dense(32, 32), relu, Dense(32, 1))

    hparams = ISLParams(; K=20, samples=1000, epochs=16, η=1e-2, transform=noise_model)

    hparams = AutoISLParams(;
        max_k=40, samples=16000, epochs=1, η=1e-2, transform=noise_model
    )

    # Preparing the training set and data loader
    train_set = Float32.(train_set)
    loader = Flux.DataLoader(
        train_set; batchsize=hparams.samples, shuffle=true, partial=false
    )

    # Training using the automatic invariant statistical loss
    #invariant_statistical_loss(gen, loader, hparams)
    total_loss = []
    ksds = []
    areas = []
    ranges = (-5:0.01:5)
    @showprogress for _ in 1:100
        #loss, ksd = auto_invariant_statistical_loss_1(gen, loader, hparams, test_set)
        #append!(total_loss, loss)
        #append!(ksds, ksd)
        #append!(total_loss, auto_invariant_statistical_loss_1(gen, loader, hparams))
        append!(total_loss, invariant_statistical_loss_1(gen, loader, hparams))
        #x = rand(noise_model, 10000)
        #ŷ = gen(x')
        x = rand(noise_model, 10000)
        ŷ = gen(x')
        append!(areas, calculate_area(test_set, ŷ, 10000000))
        append!(ksds, KSD_1(noise_model, test_set, gen, 100000, ranges))
        #ksd = HypothesisTests.ApproximateTwoSampleKSTest(train_set[1:10000], vec(ŷ))
        #append!(ksds, ksd.δp)
        #loss = invariant_statistical_loss_1(gen, loader, hparams)
    end

    calculate_area(test_set, ŷ, 10000)

    plotlyjs()

    ranges = (-2:0.01:5)
    histogram(
        normalized_data;
        range=ranges,
        #bins=10000,
        normalize=:pdf,
        label="AVG_IKI",
        alpha=0.7,
        #yscale=:log,
        xlabel="AVG_IKI",
        ylabel="Frequency",
        title="Histogram of AVG_IKI",
        xlims=(-10, 10),
    )

    x = rand(noise_model, 10000)
    ŷ = gen(x')

    #ranges = (-5:0.005:5)
    histogram!(
        ŷ';
        range=ranges,
        #bins=10000,
        label="Target distribution",
        normalize=:pdf,
        #yscale=:log,
        xlims=(-10, 10),
        norm=true,
        #alpha=0.9,
    )

    ranges = (-5:0.01:5)
    KSD(noise_model, target_model, gen, 1000000, ranges)
    KSD_1(noise_model, test_set, gen, 100000, ranges)
end

@test_experiments "N(0,1) to N(23,1)" begin

    # Assuming the data is saved in 'typing_data.txt'

    noise_model = GeneralizedPareto(0.66f0, 1.0f0, 1.0f0)

    file_path = "/Users/jmfrutos/Desktop/train_1.csv"

    df = CSV.read(file_path, DataFrame; delim=',')
    df_filled = coalesce.(df, 0)
    df_filled[!, :Sum] = sum.(eachrow(df_filled[:, Not(:Page)]))

    sum_col = df_filled.Sum

    # Calculate mean and standard deviation
    mean_val = mean(sum_col)
    std_dev = std(sum_col)

    # Normalize the data
    normalized_data = Float32.((sum_col .- mean_val) ./ std_dev)

    # Shuffle the DataFrame rows
    shuffled_df = shuffle(normalized_data)

    # Calculate the number of rows for each set
    num_rows = length(shuffled_df)
    num_train = Int(floor(0.1 * num_rows)) # 10% for training
    num_val = Int(floor(0.1 * num_rows)) # 10% for validation
    num_test = num_rows - num_train - num_val # The rest for testing

    # Split the data
    train_set = vec(shuffled_df)[1:num_train]
    val_set = vec(shuffled_df)[(num_train + 1):(num_train + num_val)]
    test_set = vec(shuffled_df)[(num_train + num_val + 1):end]

    gen = Chain(Dense(1, 32), relu, Dense(32, 32), relu, Dense(32, 32), relu, Dense(32, 1))

    hparams = ISLParams(; K=40, samples=1000, epochs=14, η=1e-1, transform=noise_model)

    # Preparing the training set and data loader
    train_set = Float32.(train_set)
    loader = Flux.DataLoader(
        train_set; batchsize=hparams.samples, shuffle=true, partial=false
    )

    total_loss = []
    ksds = []
    areas = []
    ranges = (-5:0.1:5)
    @showprogress for _ in 1:100
        #loss, ksd = auto_invariant_statistical_loss_1(gen, loader, hparams, test_set)
        #append!(total_loss, loss)
        #append!(ksds, ksd)
        #append!(total_loss, auto_invariant_statistical_loss_1(gen, loader, hparams))
        append!(total_loss, invariant_statistical_loss_1(gen, loader, hparams))
        #x = rand(noise_model, 10000)
        #ŷ = gen(x')
        x = rand(noise_model, 10000)
        ŷ = gen(x')
        append!(areas, calculate_area(test_set, ŷ, 10000000))
        append!(ksds, KSD_1(noise_model, test_set, gen, 100000, ranges))
        #ksd = HypothesisTests.ApproximateTwoSampleKSTest(train_set[1:10000], vec(ŷ))
        #append!(ksds, ksd.δp)
        #loss = invariant_statistical_loss_1(gen, loader, hparams)
    end

    plotlyjs()

    ranges = (-2:0.01:5)
    histogram(
        normalized_data;
        range=ranges,
        #bins=10000,
        normalize=:pdf,
        label="AVG_IKI",
        alpha=0.7,
        #yscale=:log,
        xlabel="AVG_IKI",
        ylabel="Frequency",
        title="Histogram of AVG_IKI",
        xlims=(-10, 10),
    )

    x = rand(noise_model, 10000)
    ŷ = gen(x')

    #ranges = (-5:0.005:5)
    histogram!(
        ŷ';
        range=ranges,
        #bins=10000,
        label="Target distribution",
        normalize=:pdf,
        #yscale=:log,
        xlims=(-10, 10),
        norm=true,
        #alpha=0.9,
    )
end
