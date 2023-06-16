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

η = 0.1; num_epochs = 100; n_samples = 1000
optim = Flux.setup(Flux.Adam(η), model)
losses = []
l = CustomLoss(2)
@time for epoch in 1:num_epochs
        aₖ = zeros(l.K+1)
        loss, grads = Flux.jacobian(model) do m  
            for x in 1:n_samples
                x = rand(Normal(μ, stddev), l.K)
                yₖ = m(x')
                y = truthh(rand(Normal(μ, stddev)))
                aₖ += generate_aₖ(l, yₖ, y)
            end
            print(scalar_diff(l, aₖ ./ sum(aₖ)))
            scalar_diff(l, aₖ ./ sum(aₖ))
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
end
