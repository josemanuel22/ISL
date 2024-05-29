using ISL
using GAN

using Flux
using MLDatasets
using Images
using ImageTransformations  # For resizing images if necessary
using LinearAlgebra

using KernelDensity
using Interpolations
using Random

using Distributions
using LinearAlgebra

using NMoons, Plots
using StatsPlots  # For density plots

# `nmoons` is adapted from https://github.com/wildart/nmoons
function nmoons(
    ::Type{T},
    n::Int=100,
    c::Int=2;
    shuffle::Bool=false,
    ε::Real=0.1,
    d::Int=2,
    translation::Vector{T}=zeros(T, d),
    rotations::Dict{Pair{Int,Int},T}=Dict{Pair{Int,Int},T}(),
    seed::Union{Int,Nothing}=nothing,
) where {T<:Real}
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(Int(seed))
    ssize = floor(Int, n / c)
    ssizes = fill(ssize, c)
    ssizes[end] += n - ssize * c
    @assert sum(ssizes) == n "Incorrect partitioning"
    pi = convert(T, π)
    R(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    X = zeros(d, 0)
    for (i, s) in enumerate(ssizes)
        circ_x = cos.(range(zero(T), pi; length=s)) .- 1.0
        circ_y = sin.(range(zero(T), pi; length=s))
        C = R(-(i - 1) * (2 * pi / c)) * hcat(circ_x, circ_y)'
        C = vcat(C, zeros(d - 2, s))
        dir = zeros(d) - C[:, end] # translation direction
        X = hcat(X, C .+ dir .* translation)
    end
    y = vcat([fill(i, s) for (i, s) in enumerate(ssizes)]...)
    if shuffle
        idx = randperm(rng, n)
        X, y = X[:, idx], y[idx]
    end
    # Add noise to the dataset
    if ε > 0.0
        X += randn(rng, size(X)) .* convert(T, ε / d)
    end
    # Rotate dataset
    for ((i, j), θ) in rotations
        X[[i, j], :] .= R(θ) * view(X, [i, j], :)
    end
    return X, y
end

function sample_moons(n)
    X, _ = nmoons(Float64, n, 2; ε=0.05, d=2, translation=[0.25, -0.25])
    return Array(X')
end

struct CoustomDistribution <: ContinuousMultivariateDistribution end

Distributions.dim(::CoustomDistribution) = 2

Base.length(::CoustomDistribution) = 2

function Distributions.rand(rng::AbstractRNG, d::CoustomDistribution)
    r = rand()
    if r < 0.5
        x = rand(rng, MvNormal([0.0f0, 0.0f0], Diagonal(ones(2))))
    else
        x = rand(rng, MvNormal([10.0f0, 10.0f0], Diagonal(ones(2))))
    end
    return x
end

function Distributions._rand!(
    rng::AbstractRNG, d::CoustomDistribution, x::AbstractArray{Float64}
)
    # Ensure that the dimensions of x are compatible with the distribution
    @assert size(x, 1) == 2 "Dimension mismatch"

    # Iterate over each column (sample) in x
    for i in 1:size(x, 2)
        r = rand()
        if r < 0.5
            x[:, i] = rand(rng, MvNormal([0.0f0, 0.0f0], Diagonal(ones(2))))
        else
            x[:, i] = rand(rng, MvNormal([10.0f0, 0.0f0], Diagonal(ones(2))))
        end
    end
    return x
end

# Define a structure that represents the Two Moons distribution
struct TwoMoons <: ContinuousMultivariateDistribution
    n_dims::Int
    max_log_prob::Float64

    function TwoMoons()
        return new(2, 0.0)
    end
end

# Constructor for TwoMoons

# Required method: dimensionality of the distribution
Base.length(d::TwoMoons) = d.n_dims

# Required method: log probability function
function Distributions.logpdf(d::TwoMoons, z::AbstractMatrix{Float64})
    a = abs.(z[:, 1])
    norm_z = vec(norm.(eachrow(z)))
    log_prob = (
        -0.5 .* ((norm_z .- 2) ./ 0.2) .^ 2 .- 0.5 .* ((a .- 2) ./ 0.3) .^ 2 .+
        log.(1 .+ exp.(-4 .* a ./ 0.09))
    )
    return log_prob
end

# Sample from the Two Moons distribution using rejection sampling
function Distributions.rand(rng::AbstractRNG, d::TwoMoons, n_samples::Int=1)
    samples = []
    while length(samples) < n_samples
        # Sample from a normal distribution centered around the two moons
        z = randn(rng, d.n_dims) .+ (rand(rng) > 0.5 ? [4, 0] : [-4, 0])

        # Calculate the probability of acceptance (simplified)
        #accept_prob = exp(-0.5 * ((norm(z) - 2) / 0.2)^2)
        #-0.5 * ((a - 2) / 0.3)^2 + 1 + exp(-4 * a / 0.09)

        a = abs.(z[:, 1])  # Get absolute values of the first column (1-based indexing)

        # Calculate the log probability for each instance
        log_prob = (
            -0.5 * ((norm(z) - 2) / 0.2) .^ 2  # norm computed for each row, broadcasting the square operation
            .- 0.5 * ((a .- 2) / 0.3) .^ 2  # broadcasting the square operation
            + log.(1 .+ exp.(-4 .* a / 0.09))  # broadcasting the log and exp operations
        )

        accept_prob = exp.(log_prob)

        # Rejection sampling step
        if rand(rng) < sum(accept_prob)
            push!(samples, z)
        end
    end
    return hcat(samples...)
end

function Distributions.rand(rng::AbstractRNG, target::TwoMoons, num_samples::Int=1)
    z = zeros(num_samples, target.n_dims)
    for i in 1:num_samples
        sign = rand(rng, Bernoulli(0.5)) * 2 - 1  # Randomly pick -1 or 1
        mu_x = sign * 2
        z[i, 1] = mu_x + randn(rng) * 0.3  # x coordinate: center at -2 or 2 with some noise
        z[i, 2] = sin(pi * z[i, 1] / 2) + randn(rng) * 0.1  # y coordinate: sin transformation and noise
    end
    return z
end

struct RingMixture <: ContinuousMultivariateDistribution
    n_dims::Int
    max_log_prob::Float64
    n_rings::Int
    scale::Float64

    function RingMixture(n_rings::Int=2)
        return new(2, 0.0, n_rings, 1 / 4 / n_rings)
    end
end

function Distributions.logpdf(rm::RingMixture, z)
    # Initialize an empty array for distances. The equivalent size in Julia will adjust dynamically.
    d = Float64[]

    # Calculate distance for each ring
    for i in 1:(rm.n_rings)
        d_ = ((norm.(eachrow(z)) .- 2 / rm.n_rings * i) .^ 2) ./ (2 * rm.scale^2)
        push!(d, d_)
    end

    # Compute log-sum-exp for the likelihoods across all rings
    return logsumexp(-reduce(hcat, d); dims=2)
end

# Helper function to compute logsumexp safely
function logsumexp(x; dims=1)
    xmax = maximum(x; dims=dims)
    return sum(exp.(x .- xmax); dims=dims) .+ xmax
end

function Distributions.rand(rng::AbstractRNG, rm::RingMixture, n_samples::Int=1)
    samples = zeros(n_samples, 2)

    for i in 1:n_samples
        # Randomly choose a ring
        ring_index = rand(rng, 1:(rm.n_rings))

        # Determine the radius of the selected ring
        radius = 2 / rm.n_rings * ring_index

        # Add noise based on the scale
        noisy_radius = radius + rm.scale * randn(rng)

        # Random angle
        angle = 2π * rand(rng)

        # Convert polar coordinates to Cartesian coordinates
        samples[i, :] = [noisy_radius * cos(angle), noisy_radius * sin(angle)]
    end

    return samples
end

struct CircularGaussianMixture
    n_modes::Int
    scale::Float32

    # Constructor with calculation of the scale
    function CircularGaussianMixture(n_modes::Int=8)
        scale = Float32((2 / 3) * sin(π / n_modes))
        return new(n_modes, scale)
    end
end

# Log probability function
function log_prob(model::CircularGaussianMixture, z)
    d = zeros(eltype(z), size(z, 1), 0)
    for i in 0:(model.n_modes - 1)
        d_ =
            (
                (z[:, 1] - 2 * sin(2 * π / model.n_modes * i)) .^ 2 +
                (z[:, 2] - 2 * cos(2 * π / model.n_modes * i)) .^ 2
            ) / (2 * model.scale^2)
        d = hcat(d, d_[:, :])
    end
    log_p = -log(2 * π * model.scale^2 * model.n_modes) .+ logsumexp(-d; dims=2)
    return log_p
end

# Sampling function
function sample(model::CircularGaussianMixture, num_samples::Int=1)
    eps = randn(Float32, (num_samples, 2))
    phi = 2 * π / model.n_modes * rand(0:(model.n_modes - 1), num_samples)
    loc = hcat(2 * sin.(phi), 2 * cos.(phi))
    return eps * model.scale .+ loc
end

function genereated_moons()
    X, _ = nmoons(Float64, 10000, 2; ε=0.1, d=2, repulse=(-0.25, 0.25))
    return Float32.(X)
end

plotlyjs()
moons = genereated_moons()

moon = TwoMoons()
moon = RingMixture(2)
moons = Float32.(sample_moons(10000))

moons = Float32.(rand(moon, 10000))
#moons = moons'
#scatter(moons[1, :], moons[2, :]; markersize=1.0, legend=:none)
x = moons[2, :]
y = moons[1, :]
kdes = kde((x, y))
contour(
    kdes.x,
    kdes.y,
    kdes.density;
    title="Density Contour Plot",
    xlabel="X",
    ylabel="Y",
    color=:viridis,
    legend=false,
    fill=true,
)

heatmap(
    kdes.x,
    kdes.y,
    kdes.density;
    #title="Density Heatmap",
    #xlabel="X-axis",
    #ylabel="Y-axis",
    legend=false,
    color=:viridis,
    grid=false,
    axis=false,
    clims=(0.02, maximum(kdes.density)),
)

#=
z_dim = 2
hidden_dim = 20
model = Chain(
    Dense(z_dim, hidden_dim, tanh),
    Dropout(0.05),
    Dense(hidden_dim, hidden_dim, tanh),
    Dropout(0.05),
    Dense(hidden_dim, hidden_dim, tanh),
    Dropout(0.05),
    Dense(hidden_dim, 2),
    Dropout(0.05),
)
=#

z_dim = 2
hidden_dim = 32
model = Chain(
    Dense(z_dim, hidden_dim, tanh),
    Dense(hidden_dim, hidden_dim, tanh),
    Dense(hidden_dim, hidden_dim, tanh),
    Dense(hidden_dim, 2),
)

dscr = Chain(Dense(2, 128), relu, Dense(128, 128), relu, Dense(128, 1, σ))

target_model = Normal(0.0f0, 1.0f0)
hparams = HyperParamsVanillaGan(;
    data_size=1000,
    batch_size=1000,
    epochs=2e2,
    lr_dscr=1e-3,
    lr_gen=1e-3,
    latent_dim=2,
    dscr_steps=1,
    gen_steps=0,
    noise_model=noise_model,
    target_model=target_model,
)

kl_divs = []
for i in 1:1000
    train_vanilla_gan(dscr, model, hparams, train_loader)
    model_cpu = cpu(model)
    res = model_cpu(Float32.(rand(cpu(hparams.noise_model), 10000)))

    x = res[1, :]
    y = res[2, :]
    append!(kl_divs, kl_divergence_2d(res, moons))
end

hparams = HyperParamsWGAN(;
    noise_model=noise_model,
    target_model=target_model,
    data_size=128,
    batch_size=128,
    epochs=2e3,
    n_critic=1,
    lr_dscr=1e-2,
    latent_dim=2,
    #lr_gen = 1.4e-2,
    lr_gen=1e-2,
)

kl_divs = []
for i in 1:100
    train_wgan(dscr, model, hparams, train_loader)
    model_cpu = cpu(model)
    res = model_cpu(Float32.(rand(cpu(hparams.noise_model), 10000)))

    x = res[1, :]
    y = res[2, :]
    append!(kl_divs, kl_divergence_2d(res, moons))
end

device = cpu

model = device(model)

# Mean vector (zero vector of length dim)
mean_vector = device(zeros(z_dim))

# Covariance matrix (identity matrix of size dim x dim)
cov_matrix = device(Diagonal(ones(z_dim)))

# Create the multivariate normal distribution
#noise_model = device(MvNormal(mean_vector, cov_matrix))
noise_model = device(CoustomDistribution())

n_samples = 10000

# Create a data loader for training
batch_size = 1000
hparams = device(
    HyperParamsSlicedISL(;
        K=10, samples=batch_size, epochs=10, η=1e-2, noise_model=noise_model, m=5
    ),
)

#train_loader = DataLoader(moons; batchsize=batch_size, shuffle=true, partial=false)
train_loader = gpu(DataLoader(moons; batchsize=batch_size, shuffle=true, partial=false))

total_loss = []
kl_divs = []
@showprogress for _ in 1:1000
    append!(
        total_loss,
        #sliced_invariant_statistical_loss_optimized_gpu_2(model, train_loader, hparams),
        sliced_invariant_statistical_loss_optimized_2(model, train_loader, hparams),
    )
    model_cpu = cpu(model)
    res = model_cpu(Float32.(rand(cpu(hparams.noise_model), 10000)))

    x = res[1, :]
    y = res[2, :]
    append!(kl_divs, kl_divergence_2d(res, moons))
end

model_cpu = cpu(model)
res = model_cpu(Float32.(rand(cpu(hparams.noise_model), 10000)))
#res = Float32.(rand(hparams.noise_model, 1000)) ## Esta linea tienes que comentarla

x = res[1, :]
y = res[2, :]
data_matrix = hcat(y, x)  # Use hcat to form a matrix
#kdes = kde(data_matrix; bandwidth=(0.1, 0.1))
kdes = kde(data_matrix)
contourf(
    kdes.x,
    kdes.y,
    kdes.density;
    title="Density Contour Plot",
    xlabel="X",
    ylabel="Y",
    #color=:blues,
    color=:viridis,
    legend=false,
    #fill=true,
    linewidth=1,
    levels=8,
)

heatmap(
    kdes.x,
    kdes.y,
    kdes.density;
    #title="Density Heatmap",
    #xlabel="X-axis",
    #ylabel="Y-axis",
    color=:viridis,
    grid=false,
    clims=(0.02, maximum(kdes.density)),
    legend=false,
    axis=false,
    left_margin=0mm,  # Explicitly set all margins to zero
    right_margin=0mm,
    top_margin=0mm,
    bottom_margin=0mm,
    #aspect_ratio=:equal,
    #alpha=0.75,
    #xlims=(-3, 3),  # Set x-axis limits
    #ylims=(-4, 4),   # Set y-axis limits
)

scatter(moons[1, :], moons[2, :]; markersize=2.0, legend=:none, color=:blue)
scatter!(res[1, :], res[2, :]; markersize=2.0, legend=:none, color=:red)

function kl_divergence_2d(p_samples, q_samples)

    # Define the grid for evaluation based on the combined range of p_samples and q_samples
    min_x = min(minimum(p_samples[1, :]), minimum(q_samples[1, :]))
    max_x = max(maximum(p_samples[1, :]), maximum(q_samples[1, :]))
    min_y = min(minimum(p_samples[2, :]), minimum(q_samples[2, :]))
    max_y = max(maximum(p_samples[2, :]), maximum(q_samples[2, :]))

    x = p_samples[1, :]
    y = p_samples[2, :]

    kde_result = kde((x, y); boundary=((min_x, max_x), (min_y, max_y)))

    densities = kde_result.density  # Estimated density values
    x_values = kde_result.x  # Points at which density is estimated

    # Convert density values to probabilities
    # For continuous distributions, you'd integrate the density over x_values.
    # Here, we approximate this by summing the densities times the step size between points.
    dx = mean(diff(x_values))  # Average distance between points
    probabilities = densities * dx  # Approximate probability for each x_value
    p = probabilities / sum(probabilities)

    x = q_samples[1, :]
    y = q_samples[2, :]
    kde_result = kde((x, y); boundary=((min_x, max_x), (min_y, max_y)))

    densities = kde_result.density  # Estimated density values
    x_values = kde_result.x  # Points at which density is estimated

    # Convert density values to probabilities
    # For continuous distributions, you'd integrate the density over x_values.
    # Here, we approximate this by summing the densities times the step size between points.
    dx = mean(diff(x_values))  # Average distance between points
    probabilities = densities * dx  # Approximate probability for each x_value
    q = probabilities / sum(probabilities)

    epsilon = 1e-40
    q .+= epsilon
    p .+= epsilon

    kl = sum(p .* log.(p ./ q))

    return kl
end

using Flux
using Distributions
using Plots

abstract type Target end

mutable struct TwoMoons <: Target
    prop_scale::Float32
    prop_shift::Float32
    n_dims::Int
    max_log_prob::Float32

    TwoMoons() = new(6.0, -3.0, 2, 0.0)  # Default values for Two Moons distribution
end

function log_prob(z::AbstractArray)
    a = abs.(z[:, 1])
    norm_z = vec(sqrt.(sum(z .^ 2; dims=2)))
    return -0.5 .* ((norm_z .- 2) ./ 0.2) .^ 2 .- 0.5 .* ((a .- 2) ./ 0.3) .^ 2 .+
           log.(1 .+ exp.(-4 .* a ./ 0.09))
end

function rejection_sampling(model::TwoMoons, num_steps::Int)
    eps = rand(Float32, (num_steps, model.n_dims))
    z_ = model.prop_scale .* eps .+ model.prop_shift
    prob = rand(Float32, num_steps)
    prob_ = exp.(log_prob(z_) .- model.max_log_prob)
    accept = prob_ .> prob
    z = z_[accept, :]
    return z
end

function sample(model::TwoMoons, num_samples::Int)
    z = Array{Float32}(undef, 0, model.n_dims)  # Initialize z as an empty 2D array with 0 rows and model.n_dims columns
    while size(z, 1) < num_samples
        z_ = rejection_sampling(model, num_samples)
        ind = min(size(z_, 1), num_samples - size(z, 1))
        z = vcat(z, z_[1:ind, :])
    end
    return z
end

d = TwoMoons()
moons = Float32.(sample(d, 10000))

struct CircularGaussianMixture <: Target
    n_modes::Int
    scale::Float32
end

function CircularGaussianMixture(n_modes=8)
    scale = Float32(2 / 3 * sin(π / n_modes))
    return new(n_modes, scale)
end

function log_prob(cgm::CircularGaussianMixture, z)
    d = zeros(Float32, size(z, 1), 0)
    for i in 0:(cgm.n_modes - 1)
        d_ =
            (
                (z[:, 1] .- 2 * sin(2π / cgm.n_modes * i)) .^ 2 +
                (z[:, 2] .- 2 * cos(2π / cgm.n_modes * i)) .^ 2
            ) ./ (2 * cgm.scale^2)
        d = hcat(d, d_)
    end
    log_p = -log(2π * cgm.scale^2 * cgm.n_modes) .+ logsumexp(-d; dims=2)
    return log_p
end

function sample(cgm::CircularGaussianMixture, num_samples=1)
    eps = randn(Float32, num_samples, 2)
    phi = 2π / cgm.n_modes * rand(0:(cgm.n_modes - 1), num_samples)
    loc = hcat(2 * sin.(phi), 2 * cos.(phi))
    return eps .* cgm.scale .+ loc
end

d = CircularGaussianMixture(8, 0.1)
moons = Float32.(sample(d, 10000))

struct RingMixture <: Target
    n_rings::Int
    n_dims::Int
    max_log_prob::Float32
    scale::Float32
    prop_scale::Array{Float32}
    prop_shift::Array{Float32}

    function RingMixture(n_rings::Int)
        prop_scale = ones(Float32, 2) * 6.0
        prop_shift = ones(Float32, 2) * -3.0
        return new(n_rings, 2, 0.0, 1 / 4 / n_rings, prop_scale, prop_shift)
    end
end

Flux.@functor RingMixture

function log_prob(model::RingMixture, z)
    d = zeros(Float32, size(z, 1))
    for i in 1:(model.n_rings)
        d_ = ((norm.(eachrow(z)) .- 2 / model.n_rings * (i + 1)) .^ 2) / (2 * model.scale^2)
        d = hcat(d, d_)
    end
    return logsumexp(-d; dims=2)
end

function rejection_sampling(model::RingMixture, num_steps::Int=1)
    eps = rand(Float32, num_steps, model.n_dims)
    z_ = model.prop_scale .* eps' .+ model.prop_shift
    prob = rand(Float32, num_steps)
    prob_ = exp.(log_prob(model, z_) .- model.max_log_prob)
    accept = prob_ .> prob'
    println(size(accept))
    z = z_[accept, :]
    return z
end

function sample(model::RingMixture, num_samples::Int=1)
    z = Float32[]
    while length(z) < num_samples
        z_ = rejection_sampling(model, num_samples)
        ind = min(size(z_, 1), num_samples - length(z))
        z = vcat(z, z_[1:ind, :])
    end
    return z
end

d = RingMixture(2)
sample(d, 100)
