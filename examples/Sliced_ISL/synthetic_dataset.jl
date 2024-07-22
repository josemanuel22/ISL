using ISL
using KernelDensity
using Random
using LinearAlgebra
using Distances
using MultivariateStats

include("../utils.jl")

function generated_syntetic_data(; d=2, D=100)
    D = 100 # For example, D is the dimension of the vector x
    d = 2 # For example, d is the dimension of the vector z and matrix A has dimensions dxd

    # Identity matrices scaled by the variance for z and ε
    Id_d = 10.0f0 * Matrix{Float32}(I, d, d)
    Id_D = Matrix{Float32}(I, D, D)

    # Create the distributions for z, A, and ε
    dist_z = MvNormal(Id_d)
    dist_A = MvNormal(Id_D)  # Here the covariance matrix is the identity matrix
    dist_ε = MvNormal(0.01f0 .* Id_D)

    # Sample from the distributions
    z = rand(dist_z) # z is a vector
    A = rand(dist_A, d) # A is a matrix
    ε = rand(dist_ε)   # ε is a vector
    return A * z + ε
end

gen_1 = Chain(Dense(10, 1000, relu), Dense(1000, 100, identity))
gen_2 = Chain(Dense(10, 1000, relu), Dense(1000, 100, identity))

noise_model = MvNormal(zeros(Float32, 10), 1.0f0 * I(10))
hparams = HyperParamsSlicedISL(;
    K=10, samples=1000, epochs=10, η=1e-4, noise_model=noise_model, m=10
)
train_set = hcat([generated_syntetic_data(; d=2, D=500) for _ in 1:10000]...)
batchsize = 1000
loader = Flux.DataLoader(train_set; batchsize=batchsize, shuffle=true, partial=false)

total_loss_1 = []
total_loss_2 = []
js_values_1 = []
js_values_2 = []
@showprogress for _ in 1:1000
    loss = marginal_invariant_statistical_loss_optimized(gen_1, loader, hparams)
    append!(total_loss_1, loss)
    js = get_js(gen_1, train_set)
    println(js)
    append!(js_values_1, get_js(gen_1, train_set))
    loss = sliced_invariant_statistical_loss_optimized_2(gen_2, loader, hparams)
    append!(total_loss_2, loss)
    js = get_js(gen_2, train_set)
    println(js)
    append!(js_values_2, js)
end

function get_js(gen, train_set)
    total = 0.0
    for _ in 1:100
        M = fit(PCA, train_set; maxoutdim=100)
        train_set_reduce = predict(M, train_set)
        x = train_set_reduce[1, :]
        y = train_set_reduce[2, :]
        kde_result = kde((x, y))

        densities = kde_result.density  # Estimated density values
        x_values = kde_result.x  # Points at which density is estimated

        # Convert density values to probabilities
        # For continuous distributions, you'd integrate the density over x_values.
        # Here, we approximate this by summing the densities times the step size between points.
        dx = mean(diff(x_values))  # Average distance between points
        probabilities = densities * dx  # Approximate probability for each x_value
        p = probabilities / sum(probabilities)

        output_data = gen(Float32.(rand(noise_model, 10000)))
        output_data = predict(M, output_data)
        x = output_data[1, :]
        y = output_data[2, :]
        kde_result = kde((x, y))

        densities = kde_result.density  # Estimated density values
        x_values = kde_result.x  # Points at which density is estimated

        # Convert density values to probabilities
        # For continuous distributions, you'd integrate the density over x_values.
        # Here, we approximate this by summing the densities times the step size between points.
        dx = mean(diff(x_values))  # Average distance between points
        probabilities = densities * dx  # Approximate probability for each x_value
        q = probabilities / sum(probabilities)

        total += js_divergence(p, q)
    end

    return total / 100.0
end

plot(
    js_values_1;
    label="Marginal",
    xlabel="Epoch",
    ylabel="JS-Divergence",
    legend=:topright,
    linecolor=:redsblues,
    fontfamily="Times New Roman",
)
plot!(js_values_2; label="Slicing", linecolor=get(ColorSchemes.rainbow, 0.2))
