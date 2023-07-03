
function proxi_cost_function(mesh, model::Function, target::Function, K::Int, n_samples::Int)
    μ::Float64 = 0.; stddev::Float64 = 1.

    losses::Vector{Float64} = []
    ms, bs = mesh
    for mᵢ in ms
        for bᵢ in bs
            loss = 0.
            aₖ = zeros(K+1)
            aₖ = ThreadsX.sum(1:n_samples) do _
                x = rand(Normal(μ, stddev), K)
                yₖ = model.(x', m=mᵢ, b=bᵢ)
                y = target(rand(Normal(μ, stddev)))
                generate_aₖ(yₖ, y)
            end
            loss = scalar_diff(aₖ ./ sum(aₖ))
            push!(losses, loss)
        end
    end
    return losses
end;

function real_cost_function(mesh, model::Function, target::Function, K::Int, n_samples::Int)
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

#model to learn
model1(x; m, b) = m * x + b
m = 3; b = 5
truthh(x) =  model1(x; m=m, b=b)

#Generating Traning Set
μ = 0
stddev = 1

res=100; n_samples = 1000; K = 2
losses = []
ms = LinRange(m-5, m+5, res)
bs = LinRange(b-5, b+5, res)
@showprogress for mᵢ in ms
    for bᵢ in bs
        loss = 0.
        aₖ = zeros(K+1)
        aₖ = ThreadsX.sum(1:n_samples) do _
            x = rand(Normal(μ, stddev), K)
            yₖ = model1.(x', m=mᵢ, b=bᵢ)
            y = truthh(rand(Normal(μ, stddev)))
            generate_aₖ(yₖ, y)
        end
        loss = jensen_shannon_∇(aₖ ./ sum(aₖ))
        push!(losses, loss)
    end
end

for mᵢ in ms
    for bᵢ in bs
        model2(x) =  model1(x; m=mᵢ, b=bᵢ)
        windows = get_window_of_Aₖ(model1, truthh, K, n_samples)
        aₖ = [count(x -> x == i, windows) for i in 0:K]
        loss = scalar_diff(aₖ ./ sum(aₖ))
        scalar_diff(aₖ ./ sum(aₖ))
        push!(losses, loss)
    end
end


plot(bs, losses)
Plots.plot(ms,bs,reshape(losses, (res,res)),st=:surface, title=string("N=",n_samples," K=", K," res=",res))
xlabel!("m")
ylabel!("b")