using ISL
using Flux
using LinearAlgebra
using Plots
using Distributions
using ProgressMeter

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

grid_sampler = GridSampler(2, 1.0, 0.02, 1000)
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

z_dim = 4
hidden_dim = 100
model = Chain(
    Dense(z_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, tanh),
    Dense(hidden_dim, 2),
)

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

noise_model = MixtureModel(map(u -> Normal(u, 1.0), [-2.0, 0.0, 3.0]))

hparams = HyperParamsSlicedISL(;
    K=10, samples=2000, epochs=2, η=1e-2, noise_model=noise_model, m=10
)

# Preparing the training set and data loader
train_set = Float32.(grid_sampler.data)
loader = Flux.DataLoader(train_set; batchsize=hparams.samples, shuffle=true, partial=false)

#scatter(train_set[1, :], train_set[2, :])

total_loss = []
@showprogress for _ in 1:100
    append!(
        total_loss, sliced_invariant_statistical_loss_optimized_2(model, loader, hparams)
    )
    #loss = sliced_invariant_statistical_loss(gen, loader, hparams)
end

z = rand(noise_model, 500)
yₖ = model(z)

#scatter(train_set[1, :], train_set[2, :])
scatter!(yₖ[1, :], yₖ[2, :]; ylim=(-5, 5), xlim=(-5, 5))
scatter!(yₖ[1, :], yₖ[2, :])
