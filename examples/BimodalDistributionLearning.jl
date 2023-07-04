using AdaptativeBlockLearning

model = Chain(
    Dense(1 => 40, tanh),
    Dense(40 => 1),
)|> gpu

μ₁=-1; σ₁ = 0.5; 
μ₂=1; σ₂ = 0.4;
μ₃=3; σ₃ = 0.4; 
realModel(ϵ) =  ϵ < 0.3 ? rand(Normal(μ₁, σ₁)) :
                ϵ < 0.7 ? rand(Normal(μ₂, σ₂)) :
                rand(Normal(μ₃, σ₃))

#train_loader = Flux.DataLoader((realModel.(rand(Float64, n_samples))), batchsize=64, shuffle=true)

#Learning with custom loss
losses = []

μ = 0; stddev = 1
η = 0.001; num_epochs = 10000; n_samples = 400; K = 20
optim = Flux.setup(Flux.Adam(η), model)

@showprogress for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        aₖ = zeros(K+1)
        for _ in 1:n_samples
            x = rand(Normal(μ, stddev), K)
            yₖ = m(x')
            y = realModel(rand(Float64))
            aₖ += @avx generate_aₖ(yₖ, y)
        end
        scalar_diff(aₖ ./ sum(aₖ))
    end
    Flux.update!(optim, model, grads[1])
    push!(losses, loss)
end;

#Learning with classical mse loss
η = 0.1; num_epochs = 100000;
optim = Flux.setup(Flux.Adam(η), model)
losses_mse = []
for epoch in 1:num_epochs
    loss, grads = Flux.withgradient(model) do m
        x = rand(Normal(μ, stddev), 1)
        ŷ = m(x')
        y = realModel(rand(Float64))
        Flux.mse(ŷ, y)
    end
    Flux.update!(optim, model, grads[1])
    push!(losses_mse, loss)
end;

x = rand(Normal(μ, stddev), 10000)
ϵ = rand(Float64, 10000)
y = realModel.(ϵ)
histogram(y, bins=-5:0.1:5, label = "distrubución real", title = "proxy error cuadrático")
ŷ = model(x')
histogram!(ŷ', bins=-5:0.1:5, label = "distrubución aproximada")
Plots.plot(x, y, seriestype = :scatter, label = "ϵ Normal($μ₁, $σ₁), (1-ϵ) Normal($μ₂, $σ₂)", title=string("N=",n_samples," K=", K, " mse"))
Plots.plot!(x, ŷ', label = "mse", seriestype = :scatter)

windows = get_window_of_Aₖ(model, realModel, K, n_samples)
aₖ = [count(x -> x == i, windows) for i in 0:K]
histogram(windows, bins=0:1:K)