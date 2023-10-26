using ThreadsX
using Plots

"""
proxi_cost_function(mesh, model, target, K, n_samples)

This function computes the cost function of a model with respect to a target function.

Args:
    mesh (Vector{LinRange{Float64, Int64}}): Meshgrid of model parameters.
    model (Function): The model function to be evaluated.
    target (Function): The target function.
    K (Int): Number of samples.
    n_samples (Int): Number of samples for Monte Carlo integration.

Returns:
    Vector{Float64}: Vector of computed losses for each combination of mesh parameters.
"""
function proxi_cost_function(
    mesh::Vector{LinRange{Float64,Int64}},
    model::Function,
    target::Function,
    K::Int,
    n_samples::Int,
)::Vector{Float64}
    μ₁::Float64 = 0.0
    σ₁::Float64 = 1.0
    losses::Vector{Float64} = []
    ms, bs = mesh

    for mᵢ in ms
        for bᵢ in bs
            loss = 0.0
            aₖ = zeros(K + 1)
            aₖ = ThreadsX.sum(1:n_samples) do _
                x = rand(Normal(μ₁, σ₁), K)
                yₖ = model.(x', m=mᵢ, b=bᵢ)
                y = target(rand(Normal(μ₁, σ₁)))
                generate_aₖ(yₖ, y)
            end
            loss = scalar_diff(aₖ ./ sum(aₖ))
            push!(losses, loss)
        end
    end
    return losses
end;

"""
Calculate the real theoretical cost function of a model with respect to a target function.

This function computes the cost function of a model with varying parameters `m` and `b`
over a specified mesh. It evaluates the model function `model` with different values
of `m` and `b` on the mesh points and calculates the cost based on the difference
between the model's output and the target function `target`. The cost is calculated
for multiple parameter combinations specified by `K` and `n_samples`.

# Arguments
- `mesh::Vector{LinRange{Float64,Int64}}`: A vector containing two `LinRange` objects
  representing the range of values for the parameters `m` and `b`.
- `model::Function`: The model function to be evaluated. It should take `x` as input and
  accept additional keyword arguments `m` and `b`.
- `target::Function`: The target function to compare the model with.
- `K::Int`: The number of parameter combinations to consider.
- `n_samples::Int`: The number of samples to use for each parameter combination.

# Returns
- `losses::Vector{Float64}`: A vector containing the calculated loss for each parameter
  combination.

# Example
```julia
mesh = [LinRange(0.0, 1.0, 10), LinRange(0.0, 2.0, 10)]
model(x; m, b) = m * x + b
target(x) = 2.0 * x + 1.0
K = 5
n_samples = 100
losses = real_cost_function(mesh, model, target, K, n_samples)
```
"""
function real_cost_function(
    mesh::Vector{LinRange{Float64,Int64}},
    model::Function,
    target::Function,
    K::Int,
    n_samples::Int,
)::Vector{Float64}
    losses::Vector{Float64} = []
    ms, bs = mesh
    for mᵢ in ms
        for bᵢ in bs
            m(x) = model.(x; m=mᵢ, b=bᵢ)
            windows = get_window_of_Aₖ(m, target, K, n_samples)
            aₖ = [count(x -> x == i, windows) for i in 0:K]
            loss = scalar_diff(aₖ ./ sum(aₖ))
            push!(losses, loss)
        end
    end
    return losses
end;

if abspath(PROGRAM_FILE) == @__FILE__
    #model to learn
    model(x; m, b) = m * x + b
    m = 3
    b = 5
    real_model(x) = model(x; m=m, b=b)

    #Generating Traning Set
    μ = 0
    stddev = 1

    res = 100
    n_samples = 1000
    K = 2
    ms = LinRange(m - 50, m + 50, res)
    bs = LinRange(b - 50, b + 50, res)

    losses_proxi = proxi_cost_function([ms, bs], model, real_model, K, n_samples)
    losses_real = real_cost_function([ms, bs], model, real_model, K, n_samples)

    #Plot the cost function longitudinal slice.
    #plot(bs, losses_proxi)

    #Plot the cost function surface.
    plot(
        ms,
        bs,
        reshape(losses_proxi, (res, res));
        st=:surface,
        title=string("N=", n_samples, " K=", K, " res=", res),
    )
    xlabel!("m")
    ylabel!("b")
end;
