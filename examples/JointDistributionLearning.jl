using LoopVectorization

model = gpu(Chain(Dense(1 => 20, tanh), Dense(20 => 1)))

#model to learn
line(x; m, b) = m * x + b

m₁ = 3;
b₁ = 5;
m₂ = -2;
b₂ = 3;
model1(x) = line(x; m=m₁, b=b₁)
model2(x) = line(x; m=m₂, b=b₂)

epsi = 0.2
real_model(x, ϵ) = ϵ < 0.2 ? model1(x) : model2(x)

#Learning with custom loss
μ = 0;
stddev = 1;
η = 0.1;
num_epochs = 50;
n_samples = 1000;
K = 10;
optim = Flux.setup(Flux.Adam(η), model)
losses = []
@showprogress for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        aₖ = zeros(K + 1)
        println(typeof(aₖ))
        y = real_model(rand(Normal(μ, stddev)), rand(Float64))
        for _ in 1:n_samples
            aₖ += generate_aₖ(
                m(rand(Normal(μ, stddev), K)'),
                real_model(rand(Normal(μ, stddev)), rand(Float64)),
            )
        end
        scalar_diff(aₖ ./ sum(aₖ))
    end
    Flux.update!(optim, model, grads[1])
    push!(losses, loss)
end;

#Learning with classical mse loss
η = 0.1;
num_epochs = 300;
optim = Flux.setup(Flux.Adam(η), model)
losses_mse = []
@showprogress for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        x = rand(Normal(μ, stddev), 100)
        ŷ = m(x')
        mean_approx = mean(ŷ)
        std_approx = std(ŷ)
        y = rand(
            Normal(epsi * b₁ + (1 - epsi) * b₂, sqrt(epsi^2 * m₁^2 + (1 - epsi)^2 * m₂^2)),
            100,
        )
        KullbackLeibler(Normal(mean_approx, std_approx), Normal(mean(y), std(y)))
    end
    Flux.update!(optim, model, grads[1])
    push!(losses_mse, loss)
end;

#Learning with classical mse loss
η = 0.1;
num_epochs = 1000;
optim = Flux.setup(Flux.Adam(η), model)
losses_mse = []
for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        x = rand(Normal(μ, stddev), 1)
        ŷ = m(x')
        y = real_model(rand(Normal(μ, stddev)), rand(Float64))
        Flux.mse(ŷ, y)
    end
    Flux.update!(optim, model, grads[1])
    push!(losses_mse, loss)
end;

x = rand(Normal(μ, stddev), 1000)
ϵ = rand(Float64, 1000)
y = real_model.(x, ϵ)
ŷ = model(x')
Plots.plot(x, y; seriestype=:scatter)
Plots.plot!(x, ŷ'; seriestype=:scatter)
