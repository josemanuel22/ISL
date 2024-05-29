using InvertibleNetworks
using LinearAlgebra
#using PyPlot
using Plots
using Flux
using Random

import Flux.Optimise: ADAM, update!
Random.seed!(1234)

n_train = 10000;
X_train = sample_banana(n_train);

scatter(X_train[1, 1, 1, 1:1000], X_train[1, 1, 2, 1:1000]; alpha=0.4, legend=:topright)

function loss(G, X)
    batch_size = size(X)[end]

    Z, lgdet = G.forward(X)

    l2_loss = 0.5 * norm(Z)^2 / batch_size  #likelihood under Normal Gaussian training
    dZ = Z / batch_size                   #gradient under Normal Gaussian training

    G.backward(dZ, Z)  #sets gradients of G wrt output and also logdet terms

    return (l2_loss, lgdet)
end

nx = 1
ny = 1

#network architecture
n_in = 2 #put 2d variables into 2 channels
n_hidden = 16
levels_L = 1
flowsteps_K = 10

G = NetworkGlow(n_in, n_hidden, levels_L, flowsteps_K;)

#G = G |> gpu

#training parameters
batch_size = 1000
maxiter = cld(n_train, batch_size)

lr = 9.0f-4
opt = ADAM(lr)

loss_l2_list = zeros(maxiter)
loss_lgdet_list = zeros(maxiter)

kldivs = []
@showprogress for _ in 1:1000
    for j in 1:(maxiter - 1)
        Base.flush(Base.stdout)
        idx = ((j - 1) * batch_size + 1):(j * batch_size)

        X = X_train[:, :, :, idx]
        #x = x |> gpu

        losses = loss(G, X) #sets gradients of G

        loss_l2_list[j] = losses[1]
        loss_lgdet_list[j] = losses[2]

        (j % 1 == 0) && println(
            "Iteration=",
            j,
            "/",
            maxiter,
            "; f l2 = ",
            loss_l2_list[j],
            "; f lgdet = ",
            loss_lgdet_list[j],
            "; f nll objective = ",
            loss_l2_list[j] - loss_lgdet_list[j],
        )

        for p in get_params(G)
            update!(opt, p.data, p.grad)
        end
    end
    num_test_samples = 1000
    Z_test = randn(Float32, nx, ny, n_in, num_test_samples)

    X_test = G.inverse(Z_test)

    X_train_reshaped = reshape(X_train[1, 1, 1:2, 1:1000], (2, 1000))
    X_test_reshaped = reshape(X_test, (2, 1000))
    append!(kldivs, kl_divergence_2d(X_test_reshaped, X_train_reshaped))
end

num_test_samples = 1000;
Z_test = randn(Float32, nx, ny, n_in, num_test_samples);

X_test = G.inverse(Z_test);

scatter(X_train[1, 1, 1, 1:1000], X_train[1, 1, 2, 1:1000]; alpha=0.4, legend=:topright)
scatter!(X_test[1, 1, 1, 1:1000], X_test[1, 1, 2, 1:1000]; alpha=0.4, color=:orange)

X_train_reshaped = reshape(X_train[1, 1, 1:2, 1:1000], (2, 1000))
X_test_reshaped = reshape(X_test, (2, 1000))
kl_divergence_2d(X_test_reshaped, X_train_reshaped)

x = X_test[1, 1, 1, 1:1000]
y = X_test[1, 1, 2, 1:1000]
data_matrix = hcat(y, x)  # Use hcat to form a matrix
#kdes = kde(data_matrix; bandwidth=(0.1, 0.1))
kdes = kde(data_matrix)
heatmap(
    kdes.x,
    kdes.y,
    kdes.density;
    #title="Density Heatmap",
    #xlabel="X-axis",
    #ylabel="Y-axis",
    color=:viridis,
    grid=false,
    clims=(0, maximum(kdes.density)),
    legend=false,
    axis=false,
    #aspect_ratio=:equal,
    #alpha=0.75,
    #xlims=(-3, 3),  # Set x-axis limits
    #ylims=(-4, 4),   # Set y-axis limits
)

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
