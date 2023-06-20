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

η = 0.01; num_epochs = 50; n_samples = 1000
optim = Flux.setup(Flux.Adam(η), model)
losses = []
l = CustomLoss(5)
@showprogress for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        aₖ = zeros(l.K+1)
        for _ in 1:n_samples
            x = rand(Normal(μ, stddev), l.K)
            yₖ = m(x')
            y = truthh(rand(Normal(μ, stddev)))
            aₖ += generate_aₖ(l, yₖ, y)
        end
        #jensen_shannon_∇(l, aₖ ./ sum(aₖ))
        scalar_diff(l, aₖ ./ sum(aₖ))
    end
    Flux.update!(optim, model, grads[1])
    push!(losses, loss)
end

x = rand(Normal(μ, stddev), 1000)
y = truthh.(x)
ŷ = model(x')
Plots.plot(x, y, seriestype = :scatter)
Plots.plot!(x, ŷ', seriestype = :scatter)