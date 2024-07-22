using ISL
using Flux
using LinearAlgebra
using Plots
using Distributions
using ProgressMeter

using GAN

mutable struct GridSampler
    n_row::Int
    n_col::Int
    edge::Float64
    sigma::Float64
    n_per_mode::Int
    centers::Matrix{Float64}
    n_mode::Int
    n_data::Int
    data::Matrix{Float64}

    function GridSampler(n_row=4, edge=1.0, sigma=0.02, n_per_mode=50)
        n_col = n_row
        return new(
            n_row,
            n_col,
            edge,
            sigma,
            n_per_mode,
            zeros(Float64, 0, 2),
            0,
            0,
            zeros(Float64, 0, 2),
        )
    end
end

function build!(sampler::GridSampler)
    function meshgrid(x, y)
        X = repeat(reshape(x, 1, :), length(y), 1)
        Y = repeat(reshape(y, :, 1), 1, length(x))
        return X, Y
    end

    mode = MvNormal(zeros(Float32, 2), sampler.sigma * I(2))
    x = LinRange(-4 * sampler.edge, 4 * sampler.edge, sampler.n_row)
    y = LinRange(-4 * sampler.edge, 4 * sampler.edge, sampler.n_col)
    X, Y = meshgrid(x, y)
    sampler.centers = hcat([[X[i], Y[i]] for i in 1:length(X)]...)

    sampler.data = Matrix{Float32}(undef, 2, 0)
    for i in (1:length(X))
        points = rand(mode, sampler.n_per_mode)
        points[1, :] .+= sampler.centers[1, i]
        points[2, :] .+= sampler.centers[2, i]
        sampler.data = hcat(sampler.data, points)
    end
end

grid_sampler = GridSampler(5, 1, 0.02, 1000)
grid_sampler = GridSampler(5, 0.2, 0.00028, 1000)
build!(grid_sampler)
#scatter(grid_sampler.data[1, :], grid_sampler.data[2, :])

mutable struct RingSampler
    n_gaussian::Int
    radius::Float64
    sigma::Float64
    n_per_mode::Int
    centers::Matrix{Float32}
    n_mode::Int
    n_data::Int
    data::Matrix{Float32}
    0.00
    function RingSampler(n_gaussian=8, radius=1.0, sigma=1e-3, n_per_mode=50)
        return new(
            n_gaussian,
            radius,
            sigma,
            n_per_mode,
            zeros(Float64, 0, 2),
            0,
            0,
            zeros(Float64, 0, 2),
        )
    end
end

function build!(sampler::RingSampler)
    sampler.centers = hcat(
        [
            [
                sampler.radius * cos(2π * i / sampler.n_gaussian),
                sampler.radius * sin(2π * i / sampler.n_gaussian),
            ] for i in 0:(sampler.n_gaussian - 1)
        ]...,
    )
    sampler.n_mode = sampler.n_gaussian
    sampler.n_data = sampler.n_mode * sampler.n_per_mode
    sampler.data = Matrix{Float32}(undef, 2, 0)

    mode = MvNormal(zeros(Float32, 2), sampler.sigma * I(2))
    for i in (1:size(sampler.centers)[2])
        points = rand(mode, sampler.n_per_mode)
        points[1, :] .+= sampler.centers[1, i]
        points[2, :] .+= sampler.centers[2, i]
        sampler.data = hcat(sampler.data, points)
    end
end

grid_sampler = RingSampler(8, 1.0, 1e-3, 1000)
build!(grid_sampler)
#scatter(grid_sampler.data[1, :], grid_sampler.data[2, :])

z_dim = 2
hidden_dim = 25
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

z_dim = 22
z_dim = 2
hidden_dim = 128
model = Chain(
    Dense(z_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, 2),
)

dscr = Chain(Dense(2, 128), relu, Dense(128, 128), relu, Dense(128, 1, σ))
#=
hparams = HyperParamsVanillaGan(;
    data_size=128,
    batch_size=128,
    epochs=1e2,
    lr_dscr=1e-3,
    lr_gen=1e-3,
    latent_dim=32,
    dscr_steps=1,
    gen_steps=0,
    noise_model=noise_model,
    target_model=target_model,
)
=#

hparams = HyperParamsVanillaGan(;
    data_size=128,
    batch_size=128,
    epochs=1e2,
    lr_dscr=1e-3,
    lr_gen=1e-3,
    latent_dim=2,
    dscr_steps=1,
    gen_steps=0,
    noise_model=noise_model,
    target_model=target_model,
)

counts = []
hparams = HyperParamsVanillaGan(;
    data_size=128,
    batch_size=128,
    epochs=3e4,
    lr_dscr=1e-3,
    lr_gen=1e-3,
    latent_dim=2,
    dscr_steps=0,
    gen_steps=0,
    noise_model=noise_model,
    target_model=target_model,
)
for i in 1:100
    train_vanilla_gan(dscr, model, hparams, loader)
    z = rand(noise_model, 10000)
    yₖ = model(z)
    push!(counts, count_within_3std(grid_sampler, yₖ))
end

countss = []
counts_max = []
@showprogress for _ in 1:1000
    #for i in 1:1000
    #=
    z_dim = 2
    model = Chain(
        Dense(z_dim, hidden_dim, tanh),
        Dense(hidden_dim, hidden_dim, tanh),
        Dense(hidden_dim, hidden_dim, tanh),
        Dense(hidden_dim, 2),
    )
    =#
    #@load "modelo_aux.bson" model
    #dscr = Chain(Dense(2, 128), relu, Dense(128, 128), relu, Dense(128, 1, σ))
    #=

    for i in 1:10
        hparams = HyperParamsSlicedISL(;
            K=10, samples=1000, epochs=8, η=1e-3, noise_model=noise_model, m=10
        )
        sliced_invariant_statistical_loss_optimized_2(model, loader, hparams)
    end
    =#
    #=
    hparams = HyperParamsMMD1D(;
        noise_model=noise_model,
        target_model=target_model,
        data_size=1000,
        batch_size=1,
        num_gen=2,
        num_enc_dec=1,
        epochs=1000,
        lr_dec=1.0e-3,
        lr_enc=1.0e-3,
        lr_gen=1.0e-3,
    )

    train_mmd_gan_1d(enc, dec, gen, hparams)
    =#
    hparams = HyperParamsVanillaGan(;
        data_size=1000,
        batch_size=1000,
        epochs=1,
        lr_dscr=1e-5,
        lr_gen=1e-5,
        latent_dim=2,
        dscr_steps=0,
        gen_steps=1,
        noise_model=noise_model,
        target_model=target_model,
    )

    train_vanilla_gan(dscr, model, hparams, loader)
    z = rand(noise_model, 10000)
    yₖ = model(z)
    push!(countss, count_within_3std(grid_sampler, yₖ))
end

sHQ = [x[1] for x in countss]
modes = [x[2] for x in countss]
# Set font to Times New Roman
default(; fontfamily="Times New Roman")
plot(
    sHQ / 100.0;
    color=:red,
    label="%HQ",
    xlabel="Steps",
    ylabel="%HQ",
    #legend=false,
    yaxis=:left,
    linewidth=2,
    ylims=[0, 100],
    #legend=:outertopright
)
plot!(; legend=:topright, legendfont=font(8, "Times New Roman"), lw=2)
plot!(
    twinx(),
    modes;
    #legend=false,
    seriestype=:steppost,
    color=:blue,
    label="# modes",
    yaxis=:right,
    ylabel="# modes",
    linestyle=:dash,
    ylims=[0, 8],
    linewidth=2,
    #legend=:outertopright
)
# Adjust legend position and style
# Adjust legend position and style
plot!(; legend=:topright, legendfont=font(8, "Times New Roman"), lw=2)

annotate!(800, 90, Plots.text("sHQ", :red, 10, "Times New Roman"))
annotate!(800, 85, Plots.text("# modes", :blue, 10, "Times New Roman"))
annotate!(
    800,
    80,
    Plots.line(800, 90; series=2, label="", linewidth=2, linestyle=:dash, color=:blue),
)

hparams = HyperParamsWGAN(;
    noise_model=noise_model,
    target_model=target_model,
    data_size=128,
    batch_size=128,
    epochs=1e3,
    n_critic=1,
    lr_dscr=1e-4,
    latent_dim=2,
    #lr_gen = 1.4e-2,
    lr_gen=1e-4,
)

counts = []
for i in 1:10
    train_wgan(dscr, model, hparams, loader)
    z = rand(noise_model, 10000)
    yₖ = model(z)
    push!(counts, count_within_3std(grid_sampler, yₖ))
end

device = cpu
model = device(model)

# Mean vector (zero vector of length dim)
mean_vector_1 = device(zeros(z_dim))
mean_vector_2 = device(ones(z_dim))

# Covariance matrix (identity matrix of size dim x dim)
cov_matrix_1 = device(Diagonal(ones(z_dim)))
cov_matrix_2 = device(Diagonal(ones(z_dim)))

# Create the multivariate normal distribution
noise_model = device(MvNormal(mean_vector_1, cov_matrix_1))
noise_model = device(
    MixtureModel([
        MvNormal(mean_vector_1, cov_matrix_1), MvNormal(mean_vector_2, cov_matrix_2)
    ]),
)

hparams = HyperParamsSlicedISL(;
    K=10, samples=1000, epochs=8, η=1e-3, noise_model=noise_model, m=10
)

# Preparing the training set and data loader
train_set = Float32.(grid_sampler.data)
loader = Flux.DataLoader(train_set; batchsize=hparams.samples, shuffle=true, partial=false)

#scatter(train_set[1, :], train_set[2, :])

total_loss = []
counts = []
@showprogress for _ in 1:10
    append!(
        total_loss, sliced_invariant_statistical_loss_optimized_2(model, loader, hparams)
    )
    #loss = sliced_invariant_statistical_loss(gen, loader, hparams)
    z = rand(noise_model, 10000)
    yₖ = model(z)
    push!(counts, count_within_3std(grid_sampler, yₖ))
end

z = rand(noise_model, 1000)
yₖ = model(z)

#scatter(train_set[1, :], train_set[2, :])
scatter!(yₖ[1, :], yₖ[2, :]; ylim=(-7, 7), xlim=(-7, 7))
scatter!(yₖ[1, :], yₖ[2, :])

function count_within_3std(sampler, yₖ)
    count = 0
    threshold_squared = (3 * sqrt(sampler.sigma))^2
    modes_covered = falses(size(sampler.centers)[2])

    # Iterate over all data points
    for i in 1:size(yₖ, 2)
        point = yₖ[:, i]
        within_3std = false

        # Check if the point is within 3 std devs of any mode center
        for j in 1:size(sampler.centers)[2]
            center = sampler.centers[:, j]
            if ((point[1] - center[1])^2 + (point[2] - center[2])^2) <= threshold_squared
                within_3std = true
                modes_covered[j] = true
                break
            end
        end

        if within_3std
            count += 1
        end
    end

    return count, sum(modes_covered)
end
