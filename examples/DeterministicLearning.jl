model = Chain(
    Dense(1, 10),
    Dense(10, 1)
)|> gpu

loss = CustomLoss(2)

losses = []
@time for epoch in 1:num_epochs
        loss, grads = Flux.withgradient(model) do m
            for (x,y) in loader
                ŷ = m(x)
                aₖ += loss->generate_aₖ(ŷ, y)
            end
            loss = loss->scalar_diff(aₖ./sum(aₖ))
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
end

