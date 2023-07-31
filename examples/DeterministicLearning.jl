using ProgressMeter

model = Chain(
    Dense(1, 10, tanh),
    Dense(10, 1)
)|> gpu

#model to learn
line(x; m, b) = m * x + b
m = 3; b = 5
truthh(x) =  line(x; m=m, b=b)

#Generating Traning Set
μ = 0.f0
stddev = 1.f0

η = 0.1; num_epochs = 2000; n_samples = 1000; K = 2;
optim = Flux.setup(Flux.Adam(η), model)
losses = []
@showprogress for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        aₖ = zeros(K+1)
        for _ in 1:n_samples
            x = rand(Normal(μ, stddev), K)
            yₖ = m(x')
            y = truthh(rand(Normal(μ, stddev)))
            aₖ += generate_aₖ(yₖ, y)
        end
        #jensen_shannon_∇(aₖ ./ sum(aₖ))
        scalar_diff(aₖ ./ sum(aₖ))
    end
    Flux.update!(optim, model, grads[1])
    push!(losses, loss)
end

#Learning with classical mse loss
η = 0.1; num_epochs = 10000;
optim = Flux.setup(Flux.Adam(η), model)
losses_mse = []
for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        x = rand(Normal(μ, stddev), 1)
        ŷ = m(x')
        y = truthh.(x)
        Flux.mse(ŷ, y)
    end
    Flux.update!(optim, model, grads[1])
    push!(losses_mse, loss)
end;

x = rand(Normal(μ, stddev), 10000)
y = truthh.(x)
ŷ = model(x')
Plots.plot(x, y, seriestype = :scatter)
Plots.plot!(x, ŷ', seriestype = :scatter)
