using Flux

struct NeuralNetworkSruct
    linear_relu_stack::Chain
end

function NeuralNetwork()
    linear_relu_stack = Chain(
        Dense(1, 10),
        x -> tanh.(x),
        Dense(10, 1)
    )|> gpu 
    return NeuralNetworkSruct(linear_relu_stack)
end

function (nn::NeuralNetworkSruct)(x)
    return nn.linear_relu_stack(x)
end

#model to learn
m = 3; b = 5
target(m,b) = m * x.pow(1) + b 

model = NeuralNetwork()

lr = 0.01
optim = Flux.setup(Flux.Adam(lr), model)  # will store optimiser momentum, etc.

# Training loop, using the whole data set 1000 times:
n_epoch = 1_000
losses = []
@showprogress for epoch in 1:n_epoch
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end
