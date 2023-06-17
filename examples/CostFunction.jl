#model to learn
model1(x; m, b) = m * x + b
m = 3; b = 5
truthh(x) =  model1(x; m=m, b=b)

#Generating Traning Set
μ = 0
stddev = 1

η = 0.1; res=400; n_samples = 1000
losses = []
l = CustomLoss(2)
ms = LinRange(m-10, m+10, res)
bs = LinRange(b-10, b+10, res)
@showprogress for mᵢ in ms
    for bᵢ in bs
        loss = 0.
        aₖ = zeros(l.K+1)
        for _ in 1:n_samples
            x = rand(Normal(μ, stddev), l.K)
            yₖ = model1.(x', m=mᵢ, b=bᵢ)
            y = truthh(rand(Normal(μ, stddev)))
            aₖ += generate_aₖ(l, yₖ, y)
        end
        loss = scalar_diff(l, aₖ ./ sum(aₖ))
        push!(losses, loss)
    end
end

@showprogress for mᵢ in ms
    for bᵢ in bs
        model2(x) =  model1(x; m=mᵢ, b=bᵢ)
        windows = get_window_of_Aₖ(model2, truthh, l.K, n_samples)
        aₖ = [count(x -> x == i, windows) for i in 0:l.K]
        loss = scalar_diff(l, aₖ ./ sum(aₖ))
        push!(losses, loss)
    end
end


plot(ms, losses)
Plots.plot(ms,bs,reshape(losses, (res,res)),st=:surface, title=string("N=",n_samples," K=", l.K," res=",res))
xlabel!("m")
ylabel!("b")
