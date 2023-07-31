using AdaptativeBlockLearning

using Flux
using StatsBase
using Distributions: Normal, rand
using HypothesisTests: pvalue, ChisqTest

#model to learn
line(x; m, b) = m * x + b
m = 3;
b = 5;
truthh(x) = line(x; m=m, b=b)

model = gpu(Chain(Dense(1, 10), Dense(10, 1)))

#Generating Traning Set
mean = 0.0
std = 1.0
n_samples = 1000
train_set = [([x], truthh(x)) for x in rand(Normal(mean, std), n_samples)]


η = 0.1;
num_epochs = 1;

losses = []
optim = Flux.setup(Flux.Adam(η), model)
loader = Flux.DataLoader(train_set; batchsize=-1);
for epoch in 1:num_epochs
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            ŷ = m(x)
            Flux.mse(ŷ, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
end
