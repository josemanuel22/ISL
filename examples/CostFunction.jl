using ThreadsX

"""
    proxi_cost_function(mesh, model, target, K, n_samples)

Compute the cost function of a model with respect to a target function.
"""
function proxi_cost_function(
    mesh::Vector{LinRange{Float64, Int64}},
    model::Function,
    target::Function,
    K::Int,
    n_samples::Int)::Vector{Float64}

    μ₁::Float64 = 0.; σ₁::Float64 = 1.
    losses::Vector{Float64} = []
    ms, bs = mesh

    for mᵢ in ms
        for bᵢ in bs
            loss = 0.
            aₖ = zeros(K+1)
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
    real_cost_function(mesh, model, target, K, n_samples)

Compute the cost function of a model with respect to a target function.
"""
function real_cost_function(
    mesh::Vector{LinRange{Float64, Int64}},
    model::Function, target::Function,
    K::Int, n_samples::Int)::Vector{Float64}

    losses::Vector{Float64} = []
    ms, bs = mesh
    for mᵢ in ms
        for bᵢ in bs
            m(x) = model(x; m=mᵢ, b=bᵢ)
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
    model1(x; m, b) = m * x + b
    m = 3; b = 5
    truthh(x) =  model1(x; m=m, b=b)

    #Generating Traning Set
    μ = 0
    stddev = 1

    res = 100; n_samples = 1000; K = 2;
    losses = []
    ms = LinRange(m-5, m+5, res)
    bs = LinRange(b-5, b+5, res)

    losses_proxi = proxi_cost_function([ms, bs], model1, truthh, K, n_samples)
    losses_real = real_cost_function([ms, bs], model1, truthh, K, n_samples)

    plot(bs, losses)
<<<<<<< HEAD
    Plots.plot(ms, bs, reshape(losses_proxi, (res,res)),st=:surface, title=string("N=",n_samples," K=", K," res=",res))
=======
    Plots.plot(ms, bs, reshape(losses, (res,res)),st=:surface, title=string("N=",n_samples," K=", K," res=",res))
>>>>>>> main
    xlabel!("m")
    ylabel!("b")
end;
