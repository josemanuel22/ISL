model = Chain(
    Dense(1, 10),
    Dense(10, 1)
)|> gpu

#model to learn
line(x; m, b) = m * x + b
model1(x) =  line(x; m=3, b=5)
model2(x) =  line(x; m=-2, b=2)

realModel(x, ϵ) =  ϵ < 0.1 ? model1(x) : model2(x)

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
        scalar_diff(l, aₖ ./ sum(aₖ))
    end
    Flux.update!(optim, model, grads[1])
    push!(losses, loss)
end

