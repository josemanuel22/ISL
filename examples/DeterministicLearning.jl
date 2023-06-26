model = Chain(
    Dense(1, 10),
    Dense(10, 1)
)|> gpu

#model to learn
line(x; m, b) = m * x + b
m = 3; b = 5
truthh(x) =  line(x; m=m, b=b)

#Generating Traning Set
μ = 0
stddev = 1

η = 0.01; num_epochs = 50; n_samples = 1000; K = 2;
optim = Flux.setup(Flux.Adam(η), model)
losses = []
l = CustomLoss(5)
for epoch in 1:num_epochs
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

x = rand(Normal(μ, stddev), 1000)
y = truthh.(x)
ŷ = model(x')
Plots.plot(x, y, seriestype = :scatter)
Plots.plot!(x, ŷ', seriestype = :scatter)