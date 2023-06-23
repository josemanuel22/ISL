model = Chain(
    Dense(1, 10),
    Dense(10, 1)
)|> gpu

#model to learn
line(x; m, b) = m * x + b
model1(x) =  line(x; m=3, b=5)
model2(x) =  line(x; m=-2, b=2)
realModel(x, ϵ) =  ϵ < 0.1 ? model1(x) : model2(x)

#Learning with custom loss
η = 0.1; num_epochs = 100; n_samples = 1000
optim = Flux.setup(Flux.Adam(η), model)
losses = []
l = CustomLoss(2)
@showprogress for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        aₖ = zeros(l.K+1)
        for _ in 1:n_samples
            x = rand(Normal(μ, stddev), l.K)
            yₖ = m(x')
            y = realModel(rand(Normal(μ, stddev)), rand(Float64))
            aₖ += generate_aₖ(l, yₖ, y)
        end
        #Flux.mse(yₖ, y)
        scalar_diff(l, aₖ ./ sum(aₖ))
    end
    Flux.update!(optim, model, grads[1])
    push!(losses, loss)
end;

#Learning with classical mse loss
η = 0.1; num_epochs = 1000;
optim = Flux.setup(Flux.Adam(η), model)
losses_mse = []
for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        x = rand(Normal(μ, stddev), l.K)
        yₖ = m(x')
        y = realModel(rand(Normal(μ, stddev)), rand(Float64))
        Flux.mse(yₖ, y)
    end
    Flux.update!(optim, model, grads[1])
    push!(losses_mse, loss)
end;

x = rand(Normal(μ, stddev), 1000)
ϵ = rand(Float64, 1000)
y = realModel.(x, ϵ)
ŷ = model(x')
Plots.plot(x, y, seriestype = :scatter)
Plots.plot!(x, ŷ', seriestype = :scatter)
